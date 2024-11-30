# config.py
from dataclasses import dataclass
import os


@dataclass
class Config:
    max_urls: int = 50
    model_name: str = "llama3.1:8b"
    cache_dir: str = "./data/"
    chunk_size: int = 500
    chunk_overlap: int = 50