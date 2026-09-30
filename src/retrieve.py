"""Retrievers behind one interface: search(query, k, filters=None) -> list[SearchResult].

  bm25    lexical scoring (rank_bm25 Okapi) over the chunk text
  dense   exact cosine similarity between the query vector and cached chunk vectors
  hybrid  Reciprocal Rank Fusion of the bm25 and dense rankings (ranks, never raw scores)

Exact score ties break by a fixed pseudo-random order derived from each chunk id (SHA-1), so
unchanged inputs always give the same ranking and the tie-break never favours a season,
episode or retriever. (Hybrid ties are common: a chunk ranked r-th only by BM25 and one ranked
r-th only by dense score exactly the same.) Nothing here knows about evaluation items.
"""
import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.artifacts import ChunkConfig
from src.chunk import load_chunk_set
from src.embedders import Embedder
from src.index import get_dense_index

BM25_TOKEN_RE = re.compile(r"[a-z0-9]+")
SCORE_DECIMALS = 12   # scores are rounded before ranking so floating-point noise cannot reorder true ties


@dataclass(frozen=True)
class Filters:
    """Optional metadata restrictions; every listed speaker must appear in a chunk."""
    season: int | None = None
    episode: int | None = None
    speakers: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    """One retrieved chunk with the provenance needed to trace it back to the source rows."""
    chunk_id: str
    season: int
    episode: int
    episode_title: str | None
    source_row_start: int
    source_row_end: int
    speakers: tuple[str, ...]
    score: float
    rank: int
    method: str
    components: dict = field(default_factory=dict)   # hybrid only: {"bm25": rank, "dense": rank}
    text: str = field(default="", repr=False)

    def provenance(self) -> dict:
        """Everything except the chunk text, as a plain dict."""
        record = asdict(self)
        record.pop("text")
        record["speakers"] = list(self.speakers)
        return record


class Retriever(Protocol):
    """The common retrieval interface."""
    name: str

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[SearchResult]: ...


def tiebreak_key(chunk_id: str) -> int:
    """Fixed pseudo-random position of a chunk among exact ties (unrelated to file order)."""
    return int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest()[:15], 16)


class ChunkCorpus:
    """One chunk set in file order, with metadata filtering and deterministic ranking."""

    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.ids = [chunk["chunk_id"] for chunk in chunks]
        self.position = {chunk_id: i for i, chunk_id in enumerate(self.ids)}
        if len(self.position) != len(self.ids):
            raise ValueError("duplicate chunk ids in the chunk set")
        self.tiebreak = np.array([tiebreak_key(chunk_id) for chunk_id in self.ids], dtype=np.int64)

    def allowed(self, filters: Filters | None) -> np.ndarray:
        """Boolean mask of chunks that pass the filters (all True without filters)."""
        mask = np.ones(len(self.chunks), dtype=bool)
        if filters:
            for i, chunk in enumerate(self.chunks):
                mask[i] = ((filters.season is None or chunk["season"] == filters.season)
                           and (filters.episode is None or chunk["episode"] == filters.episode)
                           and set(filters.speakers) <= set(chunk["speakers"]))
        return mask

    def result(self, i: int, score: float, rank: int, method: str, components: dict | None = None) -> SearchResult:
        """Wrap chunk i as a SearchResult, copying its provenance fields unchanged."""
        chunk = self.chunks[i]
        return SearchResult(chunk["chunk_id"], chunk["season"], chunk["episode"], chunk.get("episode_title"),
                            chunk["source_row_start"], chunk["source_row_end"], tuple(chunk["speakers"]),
                            float(score), rank, method, components or {}, chunk["text"])

    def ranked(self, scores: np.ndarray, allowed: np.ndarray, k: int, method: str) -> list[SearchResult]:
        """Top-k allowed chunks by score; exact ties follow the fixed tie-break order."""
        scores = np.round(scores, SCORE_DECIMALS)
        candidates = np.flatnonzero(allowed)
        order = candidates[np.lexsort((self.tiebreak[candidates], -scores[candidates]))][:max(k, 0)]
        return [self.result(i, scores[i], rank, method) for rank, i in enumerate(order, 1)]


def bm25_tokenize(text: str) -> list[str]:
    """Lowercase runs of letters and digits; apostrophes split words ("Aidan's" -> aidan, s)."""
    return BM25_TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """Okapi BM25 over chunk text; only chunks with a positive score are returned.

    With IDF clamped at 0 (config.BM25_EPSILON = 0), a positive score means the chunk shares at
    least one informative query term. Chunks matching only zero-IDF words ("I", "you") would all
    tie at 0 and be ordered by file position, favouring early episodes, so they are left out.
    """
    name = "bm25"

    def __init__(self, corpus: ChunkCorpus, k1: float = config.BM25_K1, b: float = config.BM25_B,
                 epsilon: float = config.BM25_EPSILON):
        self.corpus = corpus
        self.index = BM25Okapi([bm25_tokenize(chunk["text"]) for chunk in corpus.chunks], k1=k1, b=b, epsilon=epsilon)
        self.params = {"k1": k1, "b": b, "epsilon": epsilon, "tokenizer": "lowercase [a-z0-9]+",
                       "idf_floor": round(epsilon * self.index.average_idf, 6)}

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[SearchResult]:
        """Rank chunks by BM25 score for the query's tokens."""
        scores = np.asarray(self.index.get_scores(bm25_tokenize(query)), dtype=np.float64)
        return self.corpus.ranked(scores, self.corpus.allowed(filters) & (scores > 0), k, self.name)


class DenseRetriever:
    """Exact cosine similarity against every chunk vector (no approximate search)."""
    name = "dense"

    def __init__(self, corpus: ChunkCorpus, vectors: np.ndarray, chunk_ids: list[str], embedder: Embedder):
        if chunk_ids != corpus.ids:
            raise ValueError("dense index rows do not match the chunk set's ids and order")
        if vectors.shape != (len(corpus.ids), embedder.dim):
            raise ValueError(f"index shape {vectors.shape} does not fit {len(corpus.ids)} chunks x {embedder.dim}")
        self.corpus, self.embedder = corpus, embedder
        self.vectors = vectors.astype(np.float64)

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[SearchResult]:
        """Rank chunks by cosine similarity (vectors are unit length, so a dot product)."""
        query_vector = np.asarray(self.embedder.embed_query(query), dtype=np.float64)
        scores = self.vectors @ query_vector
        return self.corpus.ranked(scores, self.corpus.allowed(filters), k, self.name)


def reciprocal_rank_fusion(rankings: dict[str, list[str]], rrf_k: int) -> dict[str, tuple[float, dict]]:
    """Fuse ranked id lists: score(id) = sum over lists of 1 / (rrf_k + rank). Returns id -> (score, ranks)."""
    fused: dict[str, tuple[float, dict]] = {}
    for method, ids in rankings.items():
        for rank, chunk_id in enumerate(ids, 1):
            score, ranks = fused.get(chunk_id, (0.0, {}))
            fused[chunk_id] = (score + 1.0 / (rrf_k + rank), {**ranks, method: rank})
    return fused


class HybridRetriever:
    """Reciprocal Rank Fusion of several retrievers' top-`candidates` rankings."""
    name = "hybrid"

    def __init__(self, corpus: ChunkCorpus, retrievers: list, rrf_k: int = config.RRF_K,
                 candidates: int = config.HYBRID_CANDIDATES):
        self.corpus, self.retrievers = corpus, retrievers
        self.params = {"rrf_k": rrf_k, "candidates": candidates, "fuses": [r.name for r in retrievers]}
        self.rrf_k, self.candidates = rrf_k, candidates

    def search(self, query: str, k: int, filters: Filters | None = None) -> list[SearchResult]:
        """Fuse the component rankings; exact ties follow the fixed tie-break order."""
        depth = max(k, self.candidates)
        rankings = {r.name: [res.chunk_id for res in r.search(query, depth, filters)] for r in self.retrievers}
        fused = reciprocal_rank_fusion(rankings, self.rrf_k)
        fused = {chunk_id: (round(score, SCORE_DECIMALS), ranks) for chunk_id, (score, ranks) in fused.items()}
        tiebreak = lambda chunk_id: self.corpus.tiebreak[self.corpus.position[chunk_id]]
        order = sorted(fused, key=lambda chunk_id: (-fused[chunk_id][0], tiebreak(chunk_id)))
        return [self.corpus.result(self.corpus.position[chunk_id], fused[chunk_id][0], rank, self.name,
                                   fused[chunk_id][1])
                for rank, chunk_id in enumerate(order[:max(k, 0)], 1)]


def build_retrievers(chunk_config: ChunkConfig, methods: tuple[str, ...] = ("bm25", "dense", "hybrid"),
                     embedder: Embedder | None = None) -> dict[str, Retriever]:
    """Load a chunk set and set up the requested retrievers (dense needs a prebuilt index)."""
    chunks, _ = load_chunk_set(chunk_config)
    corpus = ChunkCorpus(chunks)
    built: dict[str, Retriever] = {}
    if {"bm25", "hybrid"} & set(methods):
        built["bm25"] = BM25Retriever(corpus)
    if {"dense", "hybrid"} & set(methods):
        if embedder is None:
            raise ValueError("dense and hybrid retrieval need an embedder")
        index = get_dense_index(chunk_config, chunks, embedder, build=False)
        built["dense"] = DenseRetriever(corpus, index.vectors, index.chunk_ids, embedder)
    if "hybrid" in methods:
        built["hybrid"] = HybridRetriever(corpus, [built["bm25"], built["dense"]])
    return {method: built[method] for method in methods}
