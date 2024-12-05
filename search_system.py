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
from tqdm import tqdm


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
            collection_info = self.qdrant.get_collection(self.collection_name)
            count = collection_info.points_count
            print(f"Found existing collection with {count} documents")
            if count == 0:
                self._load_initial_data()
        except:
            print("Creating new collection...")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            )
            self._load_initial_data()

    def _load_initial_data(self):
        try:
            print("\n=== Loading Initial Data ===")
            data_path = os.path.join(self.config.cache_dir, self.config.data_file)

            with open(data_path) as f:
                scraped_data = json.load(f)
            print(f"Loaded {len(scraped_data)} documents")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,
                chunk_overlap=25,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            # Process documents sequentially with progress bar
            points = []
            for idx, doc in enumerate(tqdm(scraped_data, desc="Processing documents")):
                chunks = text_splitter.split_text(doc['text'])
                chunk_embeddings = self.embeddings.embed_documents(chunks)

                for j, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
                    points.append(models.PointStruct(
                        id=f"doc_{idx}_{j}",
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "url": doc['url'],
                            "title": doc['title']
                        }
                    ))

                # Upload in batches to save memory
                if len(points) >= 100:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    points = []

            # Upload remaining points
            if points:
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )

            count = self.qdrant.get_collection(self.collection_name).points_count
            print(f"Finished loading. Collection has {count} documents")

        except Exception as e:
            print(f"Error in _load_initial_data: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            query_embedding = self.embeddings.embed_query(query)
            results = self.qdrant.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
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
                'title': hit.payload['title']
            } for hit in results]

            context = "\n\n".join([doc['text'] for doc in relevant_docs])
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