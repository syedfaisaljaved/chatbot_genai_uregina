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

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            print("Processing documents...")
            documents = []
            metadatas = []
            ids = []

            for i, doc in enumerate(scraped_data):
                if i % 100 == 0:
                    print(f"Processing document {i}/{len(scraped_data)}")
                chunks = text_splitter.split_text(doc['text'])
                documents.extend(chunks)
                metadatas.extend([{
                    "url": doc['url'],
                    "title": doc['title']
                }] * len(chunks))
                ids.extend([f"doc_{i}_{j}" for j in range(len(chunks))])

            print(f"\nTotal chunks created: {len(documents)}")
            print("Adding documents to ChromaDB...")

            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                batch_num = i // batch_size + 1
                total_batches = (len(documents) + batch_size - 1) // batch_size
                print(f"Adding batch {batch_num}/{total_batches}")

                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )

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