from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import json
import os
import logging

from config import Config
from qa import OllamaQA
from typing import Dict, Any


class SearchSystem:
    def __init__(self, config: Config):
        self.config = config
        self.qa = OllamaQA(config)

        print("Initializing embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        print("Setting up ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        try:
            print("Attempting to get existing collection...")
            self.collection = self.chroma_client.get_collection(name="uregina_docs")
            print(f"Found existing collection with {self.collection.count()} documents")
        except:
            print("Creating new collection...")
            self.collection = self.chroma_client.create_collection(name="uregina_docs")
            self._load_initial_data()

    def _load_initial_data(self):
        try:
            data_path = os.path.join(self.config.cache_dir, self.config.data_file)
            print(f"Loading data from: {data_path}")

            if not os.path.exists(data_path):
                print(f"ERROR: File not found at {data_path}")
                return

            with open(data_path) as f:
                scraped_data = json.load(f)
            print(f"Loaded {len(scraped_data)} documents from JSON")

            if len(scraped_data) == 0:
                print("ERROR: No documents found in JSON file")
                return

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            documents = []
            metadatas = []
            ids = []

            for i, doc in enumerate(scraped_data):
                chunks = text_splitter.split_text(doc['text'])
                print(f"Document {i + 1}: Split into {len(chunks)} chunks")

                documents.extend(chunks)
                metadatas.extend([{
                    "url": doc['url'],
                    "title": doc['title']
                }] * len(chunks))
                ids.extend([f"doc_{i}_{j}" for j in range(len(chunks))])

            total_chunks = len(documents)
            print(f"Total chunks to add: {total_chunks}")

            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                print(f"Added batch {i // batch_size + 1}")

            print(f"Successfully added {self.collection.count()} documents to ChromaDB")

        except Exception as e:
            print(f"Error in _load_initial_data: {str(e)}")
            raise

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        try:
            print(f"\nSearching for: {query}")
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            if not results['documents'][0]:
                print("No results found")
                return {
                    'answer': "I don't have that specific information about the University of Regina.",
                    'sources': [],
                    'success': True
                }

            print(f"Found {len(results['documents'][0])} relevant documents")

            relevant_docs = [{
                'text': doc,
                'url': meta['url'],
                'title': meta['title']
            } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

            context = "\n\n".join([doc['text'] for doc in relevant_docs])
            print(f"Context length: {len(context)}")

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
