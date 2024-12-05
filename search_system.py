from langchain_community.embeddings import HuggingFaceEmbeddings
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

        self.chroma_client = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings(
                allow_reset=True,
                anonymized_telemetry=False
            )
        )

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

    # Rest of the code remains the same