from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import json
import os

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

        # Updated ChromaDB client configuration
        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )

        # Create or get existing collection
        try:
            self.collection = self.chroma_client.get_collection(
                name="uregina_docs",
                embedding_function=self.embeddings
            )
        except ValueError:
            self.collection = self.chroma_client.create_collection(
                name="uregina_docs",
                embedding_function=self.embeddings
            )

    def initialize(self):
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

        # Add documents in batches to prevent memory issues
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            self.collection.add(
                documents=documents[i:end_idx],
                metadatas=metadatas[i:end_idx],
                ids=ids[i:end_idx]
            )

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        results = self.collection.query(
            query_texts=[query],
            n_results=k
        )

        relevant_docs = [{
            'text': doc,
            'url': meta['url'],
            'title': meta['title']
        } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

        context = "\n\n".join([doc['text'] for doc in relevant_docs])
        response = self.qa.get_response(context, query)

        return {
            'answer': response['answer'],
            'sources': [{'url': doc['url'], 'title': doc['title']} for doc in relevant_docs],
            'success': response['success']
        }