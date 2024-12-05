from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qa import OllamaQA
from config import Config
from qdrant_client import QdrantClient
from qdrant_client.http import models
import json
import os
from typing import Optional


class QdrantSingleton:
    _instance: Optional[QdrantClient] = None

    @classmethod
    def get_instance(cls) -> QdrantClient:
        if cls._instance is None:
            # Remove existing database if it exists
            if os.path.exists("./qdrant_db"):
                import shutil
                shutil.rmtree("./qdrant_db")

            cls._instance = QdrantClient(path="./qdrant_db")
        return cls._instance


class SearchSystem:
    def __init__(self, config: Config):
        print("\n=== Initializing Search System ===")
        self.config = config
        self.qa = OllamaQA(config)

        print("Initializing embeddings model...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",  # Best performing free model
            model_kwargs={
                'device': 'cuda',  # Use GPU
                'trust_remote_code': True
            }
        )

        print("Setting up Qdrant...")
        self.qdrant = QdrantSingleton.get_instance()
        self.collection_name = "uregina_docs"

        try:
            print("Creating new collection...")
            self.qdrant.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE
                )
            )
            self._load_initial_data()
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            raise

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

            points = []
            total_points = 0
            total_batches_uploaded = 0

            for idx, doc in enumerate(scraped_data):
                print(
                    f"\rProcessing document {idx + 1}/{len(scraped_data)} ({((idx + 1) / len(scraped_data)) * 100:.1f}%)",
                    end="", flush=True)

                chunks = text_splitter.split_text(doc['text'])
                chunk_embeddings = self.embeddings.embed_documents(chunks)

                for chunk, embedding in zip(chunks, chunk_embeddings):
                    points.append(models.PointStruct(
                        id=total_points,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "url": doc['url'],
                            "title": doc['title']
                        }
                    ))
                    total_points += 1

                # Upload in batches
                if len(points) >= 100:
                    total_batches_uploaded += 1
                    print(f"\nUploading batch {total_batches_uploaded} ({len(points)} points)")
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=points
                    )
                    print(f"Successfully uploaded batch {total_batches_uploaded}")
                    print(f"Total points processed so far: {total_points}")
                    points = []

            # Upload remaining points
            if points:
                total_batches_uploaded += 1
                print(f"\nUploading final batch {total_batches_uploaded} ({len(points)} points)")
                self.qdrant.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Successfully uploaded final batch")

            print(f"\nFinished processing all documents")
            print(f"Total points uploaded: {total_points}")
            print(f"Total batches uploaded: {total_batches_uploaded}")

        except Exception as e:
            print(f"\nError in _load_initial_data: {str(e)}")
            raise

    def search(self, query: str, k: int = 3):
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