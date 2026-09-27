from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    chunk_index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict:
        return {
            "source": self.chunk.source,
            "page": self.chunk.page,
            "chunk_index": self.chunk.chunk_index,
            "score": round(self.score, 4),
            "text": self.chunk.text,
        }

