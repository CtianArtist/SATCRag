# Sex and the City RAG

Retrieval-augmented question answering over the *Sex and the City* dialogue dataset
(Kaggle: `snapcrack/every-sex-and-the-city-script`).

The source is **ordered subtitle lines with speaker labels**, not screenplays: it has no
scene headings, locations, stage directions or voice-over markers, and the pipeline does not
invent them. Episodes are the only authoritative structure. Retrieval uses fixed token-based
chunks; heuristic scene groupings exist only as optional, clearly labeled metadata. The index
holds only source dialogue, with no generated summaries.

## Setup

Everything runs on CPU. Install the CPU-only PyTorch build first so nothing downloads CUDA:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Getting the data

This repository does not include the transcript corpus. Fetch it from Kaggle:

```bash
pip install kagglehub
python download_data.py
```

The script downloads version 3 of the dataset into `data/raw/` (git-ignored) and checks
`SATC_all_lines.csv` against the SHA-256 in `src/config.py`. `python -m src.parse` repeats the check,
because a different upstream file would shift source rows and invalidate eval targets. The pipeline
never modifies `data/raw/`.

The embedding model `BAAI/bge-m3` (about 2.3 GB) is downloaded on first use from its pinned revision
(`EMBEDDING_REVISION` in `src/config.py`). Everything derived from the corpus (lines, chunks, scenes,
embeddings) is rebuilt locally and not committed. Only the chunk-set manifests are committed; they
hold settings and hashes, so `git diff data/processed/chunks/` after a rebuild shows whether it
reproduced byte-identical chunks.

## Run each step

| Step | Command | Output |
|---|---|---|
| Parse and normalize | `python -m src.parse` | `data/processed/lines.jsonl`, `parse_report.json` |
| Inferred scenes (optional) | `python -m src.scenes [--show S4E13]` | `data/processed/scenes_inferred.jsonl` |
| Chunk all three sizes | `python -m src.chunk --all` | `data/processed/chunks/tok512o64-bge-m3.jsonl` + `.manifest.json`, etc. |
| Build dense indexes | `python -m src.index --all` (about an hour on a laptop CPU) | `index/dense/<chunk set>/bge-m3-<key>/` |
| Search | `python -m src.cli search "Aidan moves his stuff in" [--retriever bm25\|dense\|hybrid] [--size 256\|512\|1024] [-k 8] [--season N] [--episode N] [--speaker NAME] [--full]` | ranked chunks with provenance |
| Find a quote (for eval items) | `python -m src.cli find "I don't do plants"` | matches plus a paste-ready eval target |
| Retrieval benchmark | `python -m eval.run_eval [--eval-file eval/eval.json] [--sizes 256 512 1024] [--retrievers bm25 dense hybrid]` | `eval/results/<UTC time>.json` |
| Tests | `python -m pytest` (add `RUN_MODEL_TESTS=1` to also load the real model) | |

## Retrieval

All retrievers share one interface, `search(query, k, filters=None) -> list[SearchResult]`. Each
result carries `chunk_id`, season, episode, episode title, `source_row_start`/`source_row_end`,
speakers, score, rank, method and, for hybrid, each component's rank.

- **BM25:** `rank_bm25` Okapi over the chunk text (lowercased `[a-z0-9]+` tokens). Only chunks sharing a
  query token are returned. IDF is clamped at 0 (`BM25_EPSILON = 0.0`); see the comment in `src/config.py`.
- **Dense:** BGE-M3 (CLS pooling, normalized, 1024 dimensions) with exact cosine search over the cached
  vectors, with no approximate nearest-neighbour index. The same model embeds all three chunk sizes.
  Text over the configured context raises `TextTooLongError` rather than being truncated.
- **Hybrid:** Reciprocal Rank Fusion of the BM25 and dense top-30 rankings, scored as
  sum(1 / (60 + rank)). Raw scores are never added together.
- **Determinism:** exact ties break by a fixed pseudo-random order derived from the chunk id (SHA-1),
  so reruns are identical and the tie-break never favours a season or a retriever. Scores are rounded
  to 12 decimals first so floating-point noise cannot reorder true ties. Hybrid ties are common: a
  chunk ranked r-th only by BM25 and one ranked r-th only by dense score exactly the same. In about a
  third of queries a tie sits right at the Hit@5 or Hit@10 cut-off.
- **Filters:** optional exact metadata filters (season, episode, required speakers). The benchmark does not use them.

## Caching and reproducibility

- A chunk set is named by size, overlap and tokenizer (`tok512o64-bge-m3`). Its manifest records the
  tokenizer revision and the SHA-256 of `lines.jsonl` and of the chunk file. Loading refuses a chunk set
  whose file changed or whose `lines.jsonl` is newer.
- A dense index is keyed by a hash of exactly what was embedded (every chunk id and text, in order) plus
  the embedder identity (model, revision, context, pooling, dimensions). Any change produces a new key
  and folder, and a cached index is checked against its key and chunk ids on load. Retrieval never builds
  indexes; `src.index` does.
- Every benchmark run saves its configuration (hashes of the raw CSV, lines, chunk sets and eval file;
  embedder identity; BM25 and RRF parameters; k values; git state; library versions) next to its results.

## Data and provenance

- `data/meta/episodes.csv`: season, episode, title, taken from Wikipedia's *List of Sex
  and the City episodes* (retrieved 2026-09-29). Titles are metadata only and never appear in chunk text.
- `data/meta/speaker_aliases.json`: every speaker alias with the evidence for it. Ambiguous
  names (`Sam`, `Jack`) are only mapped in the listed episodes.
- `lines.jsonl`: one record per dialogue turn, in file order. Each record keeps `raw_text`
  and `clean_text`, `speaker_raw` and `speaker`, and the names of the rules applied (`text_rules`,
  `speaker_rules`, `row_fixes`).
  - `source_row` is the CSV's own per-episode row index (its unnamed first column). In S6E3 that
    column holds a tuple, so the index is rebuilt from row position, the same numbering every
    other episode uses.
  - `csv_record` is the 0-based record number in the CSV, and `line_id` looks like `s04e13-r017-0`.
  - A row holding two speaker turns (`- Hello? - Carrie, it's Stanford.`) is split. Because the
    dataset's label can belong to either turn, both get `speaker: null` (see
    `SPLIT_TURNS_KEEP_LABEL_ON_FIRST`).
- Chunks hold whole lines as `Speaker: text` (unknown speakers appear as `(unknown)`), with
  `source_row_start`/`source_row_end`, `line_ids`, `speakers`, `prev_chunk_id`/`next_chunk_id`
  and, when scene detection is on, `inferred_scenes`. Chunk sizes are counted in BGE-M3 tokens; the
  model sees each chunk's tokens plus 2 special tokens.

## Evaluation

Eval items point at source rows, so they stay valid whatever the chunk size or overlap:

```json
{"id": "q1", "type": "scene_fact", "question": "...", "expected_answer": "...",
 "season": 4, "episode": 13, "source_row_start": 17, "source_row_end": 22,
 "expected_quote": "I don't do plants"}
```

- A retrieved chunk is relevant when it is from the same season and episode and its row span overlaps
  the target span.
- Leave out the rows for episode-level questions.
- Use `"targets": [{...}, {...}]` for questions spanning several places.
- `expected_quote` is optional. The runner warns if it no longer lies inside its target span.
- **Metrics:** Hit@1/5/10 (any target reached), Recall@1/5/10 (fraction of targets reached; equal to Hit@k
  for single-target items) and MRR over the top 10.
- Results are saved per query (top-10 provenance with relevance flags) and as aggregates, overall and by
  question type. The `comparisons` section lists queries where BM25 and dense disagree, where hybrid
  recovered or lost a hit, and which queries are sensitive to chunk size.

`eval/smoke.json` is a five-item plumbing check, not a benchmark.
