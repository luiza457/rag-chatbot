from __future__ import annotations

from pathlib import Path
from typing import Protocol

from app.config import Settings
from app.document_loader import load_directory, load_document
from app.embeddings import Embedder
from app.generation import AnswerGenerator
from app.models import Chunk, SearchResult


class VectorStore(Protocol):
    size: int

    def add(self, chunks: list[Chunk], vectors): ...
    def search(self, query_vector, top_k: int) -> list[SearchResult]: ...


class RAGService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: VectorStore,
        generator: AnswerGenerator,
    ) -> None:
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.generator = generator

    def ingest_chunks(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        self.store.add(chunks, vectors)
        if hasattr(self.store, "save"):
            self.store.save()
        return len(chunks)

    def ingest_file(self, path: Path) -> int:
        return self.ingest_chunks(
            load_document(
                path,
                max_words=self.settings.chunk_size_words,
                overlap_words=self.settings.chunk_overlap_words,
            )
        )

    def ingest_directory(self, path: Path) -> int:
        return self.ingest_chunks(
            load_directory(
                path,
                max_words=self.settings.chunk_size_words,
                overlap_words=self.settings.chunk_overlap_words,
            )
        )

    def ask(self, question: str, top_k: int | None = None) -> dict:
        k = top_k or self.settings.top_k
        query_vector = self.embedder.encode([question])[0]
        results = self.store.search(query_vector, k)
        return {
            "question": question,
            "answer": self.generator.generate(question, results),
            "mode": self.generator.mode,
            "sources": [result.to_dict() for result in results],
        }

