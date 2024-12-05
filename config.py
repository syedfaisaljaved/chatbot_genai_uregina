# config.py
from dataclasses import dataclass


@dataclass
class Config:
    model_name: str = "llama3.1:8b"
    cache_dir: str = "./data/"
    chunk_size: int = 500
    chunk_overlap: int = 50
    data_file: str = "web_content.json"