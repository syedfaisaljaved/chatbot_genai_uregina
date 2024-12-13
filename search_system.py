from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qa import OllamaQA
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import json
import os
import torch
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import gc


class SearchSystem:
    def __init__(self, config: Config):
        print("\n=== Initializing Search System ===")
        self.config = config
        self.qa = OllamaQA(config)

        # GPU setup and optimization
        if torch.cuda.is_available():
            print(f"GPU detected: {torch.cuda.get_device_name(0)}")
            torch.cuda.empty_cache()
            self.device = 'cuda'
            # Set optimal GPU memory settings
            torch.cuda.set_per_process_memory_fraction(0.8)  # Reserve some GPU memory
        else:
            print("No GPU detected, using CPU")
            self.device = 'cpu'

        print("Initializing BGE embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={
                'device': self.device,
                'trust_remote_code': True
            }
        )

        print("Setting up Qdrant...")
        self.qdrant = QdrantClient(path="./qdrant_db")
        self.collection_name = "uregina_docs"

        # Optimize collection settings
        try:
            print("Creating new collection...")
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1024,  # BGE model dimension
                    distance=Distance.COSINE
                ),
                optimizers_config={
                    "default_segment_number": 8  # Optimize for 8 vCPUs
                }
            )
            self._load_initial_data()
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise

    def process_batch(self, chunks: List[str], start_idx: int) -> List[models.PointStruct]:
        """Process a batch of chunks with optimized memory handling"""
        try:
            # Generate embeddings for the batch
            with torch.cuda.amp.autocast():  # Use automatic mixed precision
                embeddings = self.embeddings.embed_documents(chunks)

            # Create points
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                points.append(models.PointStruct(
                    id=start_idx + i,
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "chunk_index": start_idx + i
                    }
                ))

            return points
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            return []
        finally:
            # Clean up GPU memory
            if self.device == 'cuda':
                torch.cuda.empty_cache()

    def _load_initial_data(self):
        try:
            print("\n=== Loading Initial Data ===")
            data_path = os.path.join(self.config.cache_dir, self.config.data_file)

            with open(data_path) as f:
                scraped_data = json.load(f)
            print(f"Loaded {len(scraped_data)} documents")

            # Optimize chunk size for performance
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            # Optimize batch sizes for g4dn.2xlarge
            embedding_batch_size = 64  # Larger batch size for 16GB GPU
            upload_batch_size = 100

            total_points = 0
            current_batch = []
            metadata_map = {}

            print("Processing documents and generating embeddings...")

            # Process documents with progress bar
            with tqdm(total=len(scraped_data), desc="Processing documents") as pbar:
                for doc_idx, doc in enumerate(scraped_data):
                    chunks = text_splitter.split_text(doc['text'])

                    # Store metadata
                    for _ in range(len(chunks)):
                        metadata_map[total_points + len(current_batch)] = {
                            "url": doc['url'],
                            "title": doc['title']
                        }

                    current_batch.extend(chunks)

                    # Process when batch is full or on last document
                    if len(current_batch) >= embedding_batch_size or doc_idx == len(scraped_data) - 1:
                        points = self.process_batch(current_batch, total_points)

                        # Add metadata
                        for point in points:
                            point.payload.update(metadata_map[point.id])

                        # Upload in optimal batches
                        for i in range(0, len(points), upload_batch_size):
                            batch = points[i:i + upload_batch_size]
                            self.qdrant.upsert(
                                collection_name=self.collection_name,
                                points=batch
                            )

                        total_points += len(points)
                        current_batch = []
                        metadata_map = {}

                        # Force garbage collection
                        gc.collect()
                        if self.device == 'cuda':
                            torch.cuda.empty_cache()

                    pbar.update(1)

            print(f"\nFinished processing all documents")
            print(f"Total points uploaded: {total_points}")

        except Exception as e:
            print(f"\nError in _load_initial_data: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            # Generate query embedding with GPU acceleration
            with torch.cuda.amp.autocast():
                query_embedding = self.embeddings.embed_query(query)

            # Search with optimized parameters
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k,
                score_threshold=0.7  # Add relevance threshold
            )

            if not results:
                return {
                    'answer': "I don't have that specific information about the University of Regina.",
                    'sources': [],
                    'success': True
                }

            relevant_docs = [{
                'text': hit.payload['text'],
                'url': hit.payload['url'],
                'title': hit.payload['title'],
                'score': hit.score  # Include relevance score
            } for hit in results]

            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            response = self.qa.get_response(context, query)

            return {
                'answer': response['answer'],
                'sources': [{'url': doc['url'], 'title': doc['title'], 'relevance': doc['score']}
                            for doc in relevant_docs],
                'success': response['success']
            }

        except Exception as e:
            print(f"Search error: {str(e)}")
            return {
                'answer': "I encountered an error while searching. Please try again.",
                'sources': [],
                'success': False
            }
        finally:
            # Clean up after search
            if self.device == 'cuda':
                torch.cuda.empty_cache()