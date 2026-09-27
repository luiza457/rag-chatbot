from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Semantic embeddings generated locally; document text is not sent externally."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dimension = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


class HashingEmbedder:
    """Dependency-free fallback for smoke tests; less semantic than the main model."""

    def __init__(self, dimension: int = 384) -> None:
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                column = value % self.dimension
                sign = 1.0 if (value >> 8) % 2 == 0 else -1.0
                matrix[row, column] += sign
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms


def build_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(model_name)
    if backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend}")

