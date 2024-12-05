# search_system.py
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
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        # Configure ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )

        # Initialize collection
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize or create the collection if it doesn't exist"""
        try:
            self.collection = self.chroma_client.get_collection(name="uregina_docs")
        except Exception as e:
            logging.info(f"Creating new collection: {str(e)}")
            self.collection = self.chroma_client.create_collection(name="uregina_docs")
            self._load_initial_data()

    def _load_initial_data(self):
        """Load initial data into the collection"""
        try:
            with open(os.path.join(self.config.cache_dir, 'scraped_data.json')) as f:
                scraped_data = json.load(f)

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )

            documents = []
            metadatas = []
            ids = []

            for i, doc in enumerate(scraped_data):
                chunks = text_splitter.split_text(doc['text'])
                documents.extend(chunks)
                metadatas.extend([{
                    "url": doc['url'],
                    "title": doc['title']
                }] * len(chunks))
                ids.extend([f"doc_{i}_{j}" for j in range(len(chunks))])

            # Add documents in batches
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                self.collection.add(
                    documents=documents[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
                logging.info(f"Added batch {i // batch_size + 1}")

        except Exception as e:
            logging.error(f"Error loading initial data: {str(e)}")
            raise