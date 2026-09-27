from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app.models import Chunk, SearchResult


class FaissVectorStore:
    """Cosine-similarity vector store backed by FAISS IndexFlatIP."""

    def __init__(self, dimension: int, data_dir: Path) -> None:
        self.dimension = dimension
        self.data_dir = data_dir
        self.index_path = data_dir / "index.faiss"
        self.metadata_path = data_dir / "chunks.json"
        self.chunks: list[Chunk] = []
        self._faiss = self._import_faiss()
        self.index = self._faiss.IndexFlatIP(dimension) # inner product 

    @staticmethod
    def _import_faiss():
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS is not installed. Run: pip install -r requirements.txt"
            ) from exc
        return faiss

    @property
    def size(self) -> int:
        return int(self.index.ntotal)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")
        if not chunks:
            return
        vectors = np.asarray(vectors, dtype="float32")
        if vectors.shape[1] != self.dimension:
            raise ValueError("Embedding dimension does not match FAISS index")
        self._faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if self.size == 0:
            return []
        query = np.asarray(query_vector, dtype="float32").reshape(1, -1)
        self._faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, self.size))
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(score))
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(
            json.dumps([chunk.to_dict() for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )

    def load(self) -> bool:
        if not self.index_path.exists() or not self.metadata_path.exists():
            return False
        index = self._faiss.read_index(str(self.index_path))
        if index.d != self.dimension:
            raise ValueError(
                "Saved index dimension differs from the configured embedding model"
            )
        chunks = [
            Chunk.from_dict(item)
            for item in json.loads(self.metadata_path.read_text(encoding="utf-8"))
        ]
        if index.ntotal != len(chunks):
            raise ValueError("FAISS index and chunk metadata are inconsistent")
        self.index = index
        self.chunks = chunks
        return True

    def reset(self) -> None:
        self.index = self._faiss.IndexFlatIP(self.dimension)
        self.chunks = []
        for path in (self.index_path, self.metadata_path):
            if path.exists():
                path.unlink()


class NumpyVectorStore:
    """Small in-memory test double with the same cosine-search behaviour."""

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.chunks: list[Chunk] = []
        self.vectors = np.empty((0, dimension), dtype="float32")

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype="float32")
        if len(chunks) != len(vectors) or vectors.shape[1] != self.dimension:
            raise ValueError("Chunk and vector dimensions do not match")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = np.vstack([self.vectors, vectors / norms])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchResult]:
        if not self.chunks:
            return []
        query = np.asarray(query_vector, dtype="float32").reshape(-1)
        norm = np.linalg.norm(query) or 1.0
        scores = self.vectors @ (query / norm)
        indices = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(self.chunks[int(index)], float(scores[index]))
            for index in indices
        ]

