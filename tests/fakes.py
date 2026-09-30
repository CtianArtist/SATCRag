"""A deterministic stand-in for the embedding model, so tests need no model download."""
import hashlib
import re

import numpy as np

from src.embedders import ensure_fits

WORD_RE = re.compile(r"[a-z0-9]+")


class FakeEmbedder:
    """Bag-of-words vectors hashed into `dim` buckets; texts count one token per word plus 2 specials."""

    def __init__(self, dim: int = 64, max_tokens: int = 512, name: str = "fake/bag-of-words",
                 revision: str = "test"):
        self.dim, self.max_tokens, self.name, self.revision = dim, max_tokens, name, revision
        self.documents_embedded = 0

    def count_tokens(self, text: str) -> int:
        """One token per word, plus two special tokens."""
        return len(text.split()) + 2

    def vector(self, text: str) -> np.ndarray:
        """Unit-length hashed bag-of-words vector (never all zeros)."""
        v = np.zeros(self.dim)
        for word in WORD_RE.findall(text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        v[0] += 1e-3
        return (v / np.linalg.norm(v)).astype(np.float32)

    def embed_documents(self, texts: list[str], ids: list[str] | None = None) -> np.ndarray:
        """Check lengths like the real embedder, then embed."""
        ensure_fits(texts, self.count_tokens, self.max_tokens, ids)
        self.documents_embedded += len(texts)
        return np.stack([self.vector(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Check length like the real embedder, then embed."""
        ensure_fits([text], self.count_tokens, self.max_tokens)
        return self.vector(text)

    def identity(self) -> dict:
        """Cache-key identity, mirroring SentenceTransformerEmbedder.identity()."""
        return {"model": self.name, "revision": self.revision, "max_tokens": self.max_tokens,
                "dim": self.dim, "normalized": True, "pooling": "bag-of-words"}


def make_chunk(season: int, episode: int, index: int, start: int, end: int, text: str,
               speakers: list[str] | None = None, title: str | None = None) -> dict:
    """A minimal chunk record like the ones in data/processed/chunks/*.jsonl."""
    return {"chunk_id": f"s{season:02d}e{episode:02d}-test-{index:03d}", "season": season, "episode": episode,
            "episode_title": title, "source_row_start": start, "source_row_end": end,
            "speakers": speakers or [], "text": text}
