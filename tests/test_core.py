from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.chunking import chunk_text
from app.embeddings import HashingEmbedder
from app.models import Chunk
from app.vector_store import NumpyVectorStore


class ChunkingTests(unittest.TestCase):
    def test_chunk_overlap_and_metadata(self):
        text = " ".join(f"word{i}" for i in range(25))
        chunks = chunk_text(text, "manual.txt", max_words=10, overlap_words=2)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].text.split()[-2:], chunks[1].text.split()[:2])
        self.assertEqual(chunks[0].source, "manual.txt")
        self.assertNotEqual(chunks[0].id, chunks[1].id)

    def test_invalid_overlap(self):
        with self.assertRaises(ValueError):
            chunk_text("hello", "x.txt", max_words=10, overlap_words=10)


class RetrievalTests(unittest.TestCase):
    def test_cosine_search_returns_closest_vector(self):
        store = NumpyVectorStore(dimension=3)
        chunks = [
            Chunk("a", "bearing vibration", "a.txt", None, 0),
            Chunk("b", "exhaust temperature", "b.txt", None, 0),
        ]
        store.add(chunks, np.array([[1, 0, 0], [0, 1, 0]], dtype="float32"))
        results = store.search(np.array([0.9, 0.1, 0], dtype="float32"), top_k=1)
        self.assertEqual(results[0].chunk.id, "a")

    def test_hashing_embedder_is_deterministic_and_normalised(self):
        embedder = HashingEmbedder(dimension=64)
        first = embedder.encode(["bearing vibration trend"])
        second = embedder.encode(["bearing vibration trend"])
        np.testing.assert_allclose(first, second)
        self.assertAlmostEqual(float(np.linalg.norm(first[0])), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()

