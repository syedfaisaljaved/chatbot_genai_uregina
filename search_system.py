from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qa import OllamaQA
from config import Config
import chromadb
from chromadb.config import Settings
import json
import os
import logging
from typing import Dict, Any, List
import torch


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

        print("Setting up ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        try:
            print("Attempting to get collection 'uregina_docs'...")
            self.collection = self.chroma_client.get_collection(name="uregina_docs")
            count = self.collection.count()
            print(f"Found existing collection with {count} documents")
            if count == 0:
                print("Collection empty, loading initial data...")
                self._load_initial_data()
        except Exception as e:
            print(f"Collection not found: {str(e)}")
            print("Creating new collection...")
            self.collection = self.chroma_client.create_collection(name="uregina_docs")
            self._load_initial_data()

    def _load_initial_data(self):
        try:
            print("\n=== Loading Initial Data ===")
            data_path = os.path.join(self.config.cache_dir, self.config.data_file)
            print(f"Reading from: {data_path}")

            with open(data_path) as f:
                scraped_data = json.load(f)
            print(f"Successfully loaded JSON with {len(scraped_data)} documents")

            # Reduce chunk size and batch size for better memory management
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=250,  # Reduced from 500
                chunk_overlap=25,  # Reduced from 50
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            print("Processing documents in smaller batches...")
            documents = []
            metadatas = []
            ids = []
            processed = 0

            # Process in smaller document batches
            doc_batch_size = 500  # Process 500 documents at a time
            for start_idx in range(0, len(scraped_data), doc_batch_size):
                end_idx = min(start_idx + doc_batch_size, len(scraped_data))
                current_batch = scraped_data[start_idx:end_idx]

                for i, doc in enumerate(current_batch):
                    processed += 1
                    if processed % 100 == 0:
                        print(
                            f"Processing document {processed}/{len(scraped_data)} ({(processed / len(scraped_data) * 100):.1f}%)")

                    chunks = text_splitter.split_text(doc['text'])
                    documents.extend(chunks)
                    metadatas.extend([{
                        "url": doc['url'],
                        "title": doc['title']
                    }] * len(chunks))
                    ids.extend([f"doc_{start_idx + i}_{j}" for j in range(len(chunks))])

                # Add to ChromaDB when batch is full
                if len(documents) >= 1000 or end_idx == len(scraped_data):
                    print(f"\nAdding {len(documents)} chunks to ChromaDB...")

                    # Add in smaller sub-batches
                    sub_batch_size = 100
                    for i in range(0, len(documents), sub_batch_size):
                        end = min(i + sub_batch_size, len(documents))
                        print(
                            f"Adding sub-batch {i // sub_batch_size + 1}/{(len(documents) + sub_batch_size - 1) // sub_batch_size}")

                        self.collection.add(
                            documents=documents[i:end],
                            metadatas=metadatas[i:end],
                            ids=ids[i:end]
                        )

                    # Clear processed chunks
                    documents = []
                    metadatas = []
                    ids = []

            final_count = self.collection.count()
            print(f"\nFinished loading data. Collection now has {final_count} documents")

        except Exception as e:
            print(f"ERROR in _load_initial_data: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            print(f"\nSearching for query: {query}")
            collection_count = self.collection.count()
            print(f"Collection size: {collection_count} documents")

            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            found_docs = len(results['documents'][0])
            print(f"Found {found_docs} relevant documents")

            if found_docs == 0:
                return {
                    'answer': "I don't have that specific information about the University of Regina.",
                    'sources': [],
                    'success': True
                }

            relevant_docs = [{
                'text': doc,
                'url': meta['url'],
                'title': meta['title']
            } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

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