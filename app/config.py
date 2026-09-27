from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    embedding_backend: str = os.getenv("EMBEDDING_BACKEND", "sentence_transformers")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5.6")
    chunk_size_words: int = int(os.getenv("CHUNK_SIZE_WORDS", "180"))
    chunk_overlap_words: int = int(os.getenv("CHUNK_OVERLAP_WORDS", "30"))
    top_k: int = int(os.getenv("TOP_K", "3"))
    data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
    documents_dir: Path = Path(os.getenv("DOCUMENTS_DIR", "docs"))
    auto_ingest_sample: bool = _as_bool(os.getenv("AUTO_INGEST_SAMPLE"), True)


settings = Settings()

