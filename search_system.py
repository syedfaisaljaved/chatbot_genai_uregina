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

        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        try:
            self.collection = self.chroma_client.get_collection(name="uregina_docs")
        except:
            self.collection = self.chroma_client.create_collection(name="uregina_docs")
            self._load_initial_data()

    def _load_initial_data(self):
        # ... existing code ...
        pass

    def search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """
        Search for relevant documents and get response
        """
        try:
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=k
            )

            # Format relevant documents
            relevant_docs = [{
                'text': doc,
                'url': meta['url'],
                'title': meta['title']
            } for doc, meta in zip(results['documents'][0], results['metadatas'][0])]

            # Create context from relevant documents
            context = "\n\n".join([doc['text'] for doc in relevant_docs])

            # Get response from QA system
            response = self.qa.get_response(context, query)

            return {
                'answer': response['answer'],
                'sources': [{'url': doc['url'], 'title': doc['title']} for doc in relevant_docs],
                'success': response['success']
            }

        except Exception as e:
            logging.error(f"Search error: {str(e)}")
            return {
                'answer': "I encountered an error while searching. Please try again.",
                'sources': [],
                'success': False
            }