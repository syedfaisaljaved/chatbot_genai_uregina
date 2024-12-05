from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qa import OllamaQA
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
import json
import os
import logging
from typing import Dict, Any, List
import torch
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial


class SearchSystem:
    def __init__(self, config: Config):
        print("\n=== Initializing Search System ===")
        self.config = config
        self.qa = OllamaQA(config)

        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        print("Setting up Qdrant...")
        self.qdrant = QdrantClient(path="./qdrant_db")
        self.collection_name = "uregina_docs"

        try:
            print("Attempting to get collection 'uregina_docs'...")
            collection_info = self.qdrant.get_collection(self.collection_name)
            count = collection_info.points_count
            print(f"Found existing collection with {count} documents")
            if count == 0:
                print("Collection empty, loading initial data...")
                self._load_initial_data()
        except Exception as e:
            print(f"Collection not found: {str(e)}")
            print("Creating new collection...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            self._load_initial_data()

    def process_batch(self, batch_data):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=25,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

        points = []
        for i, doc in enumerate(batch_data):
            chunks = text_splitter.split_text(doc['text'])
            chunk_embeddings = self.embeddings.embed_documents(chunks)

            for j, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                points.append(models.PointStruct(
                    id=f"doc_{i}_{j}",
                    vector=embedding,
                    payload={
                        "text": chunk,
                        "url": doc['url'],
                        "title": doc['title']
                    }
                ))
        return points

    def _load_initial_data(self):
        try:
            print("\n=== Loading Initial Data ===")
            data_path = os.path.join(self.config.cache_dir, self.config.data_file)
            print(f"Reading from: {data_path}")

            with open(data_path) as f:
                scraped_data = json.load(f)
            print(f"Successfully loaded JSON with {len(scraped_data)} documents")

            # Calculate optimal batch size and number of processes
            num_processes = cpu_count() - 1  # Leave one CPU free
            batch_size = max(1, len(scraped_data) // (num_processes * 4))  # Divide data into smaller batches

            # Create batches
            batches = [
                scraped_data[i:i + batch_size]
                for i in range(0, len(scraped_data), batch_size)
            ]

            print(f"Processing documents using {num_processes} processes...")

            # Process batches in parallel
            with Pool(processes=num_processes) as pool:
                # Use tqdm for progress bar
                results = list(tqdm(
                    pool.imap(self.process_batch, batches),
                    total=len(batches),
                    desc="Processing batches"
                ))

            # Flatten results and upload to Qdrant
            all_points = [point for batch_points in results for point in batch_points]

            # Upload to Qdrant in batches
            upload_batch_size = 100
            for i in range(0, len(all_points), upload_batch_size):
                batch = all_points[i:i + upload_batch_size]
                print(
                    f"Uploading batch {i // upload_batch_size + 1}/{(len(all_points) + upload_batch_size - 1) // upload_batch_size}")
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )

            final_count = self.qdrant.get_collection(self.collection_name).points_count
            print(f"\nFinished loading data. Collection now has {final_count} documents")

        except Exception as e:
            print(f"ERROR in _load_initial_data: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            print(f"\nSearching for query: {query}")
            collection_count = self.qdrant.get_collection(self.collection_name).points_count
            print(f"Collection size: {collection_count} documents")

            query_embedding = self.embeddings.embed_query(query)
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )

            found_docs = len(results)
            print(f"Found {found_docs} relevant documents")

            if found_docs == 0:
                return {
                    'answer': "I don't have that specific information about the University of Regina.",
                    'sources': [],
                    'success': True
                }

            relevant_docs = [{
                'text': hit.payload['text'],
                'url': hit.payload['url'],
                'title': hit.payload['title']
            } for hit in results]

            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            print(f"Context length: {len(context)} characters")

            response = self.qa.get_response(context, query)
            return {
                'answer': response['answer'],
                'sources': [{'url': doc['url'], 'title': doc['title']} for doc in relevant_docs],
                'success': response['success']
            }

        except Exception as e:
            print(f"Search error: {str(e)}")
            return {
                'answer': "I encountered an error while searching. Please try again.",
                'sources': [],
                'success': False
            }