# search_system.py
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
import json
import os
from typing import Dict, Any

from config import Config
from qa import OllamaQA


class SearchSystem:
    def __init__(self, config: Config):
        self.config = config
        self.qa = OllamaQA(config)
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )

        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory="./chroma_db"
        ))
        self.collection = self.chroma_client.create_collection(
            name="uregina_docs",
            embedding_function=self.embeddings
        )

    def initialize(self):
        with open(os.path.join(self.config.cache_dir, 'web_content.json')) as f:
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

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        self.chroma_client.persist()

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
