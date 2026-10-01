"""SATC-RAG: a retrieval benchmark (BM25, dense and hybrid) over noisy TV-dialogue subtitles.

The package turns an ordered subtitle CSV into provenance-preserving dialogue lines, packs them
into token-sized chunks, retrieves chunks with BM25, exact dense search or Reciprocal Rank
Fusion, and scores retrieval against human-approved, row-level evidence. The command-line entry
point is `satc-rag` (see satc_rag.cli).
"""

__version__ = "0.1.0"
