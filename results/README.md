# Results

Experiment outputs. Everything here is **git-ignored except this file and the two `.gitkeep`
placeholders**, so nothing is published by accident.

```
results/
  runs/       one directory per evaluation run, written by `sexandrag eval` (never committed)
  reports/    write-ups and summary tables chosen for publication
  figures/    plots chosen for publication
```

## Run directories

Every `sexandrag eval` creates a new directory. An existing one is never reused or overwritten; if the
name is already taken, `-2`, `-3` and so on is appended.

```
results/runs/<UTC time>-<label>-<commit>[-dirty][-N]/
    e.g. results/runs/20261001T141500Z-dev-bf75352-dirty/
```

- `<UTC time>` is the start time, as `YYYYMMDDTHHMMSSZ`.
- `<label>` is `dev`, `test` (only possible with `--allow-heldout`), or the eval file's name.
- `<commit>` is the short git commit. `-dirty` means the working tree had uncommitted changes.

| File | Contents |
|---|---|
| `run.json` | Run id, start and finish times, wall time, the eval file with its SHA-256, split, item count and held-out flag, and the git commit and dirty state. |
| `config.json` | Every setting in effect, including BM25 and RRF parameters, k values and depth. Also: the retrievers and their parameters, the embedder identity (model, revision, pooling, context, dimension), each chunk set's identity and manifest hashes, the dense index key, and the SHA-256 of the eval file, raw CSV, `lines.jsonl` and frozen manifest. |
| `environment.json` | The package version, Python, platform and machine, and the versions of torch, sentence-transformers, transformers, tokenizers, huggingface_hub, numpy and rank-bm25. |
| `metrics.json` | Per (chunk size, retriever) cell: aggregates over all questions and per question type, plus seconds per query. Also the comparisons used for failure analysis, and timing. |
| `queries.jsonl` | One row per (cell, question), described below. |

Each row of `queries.jsonl` holds:

- the metrics `hit@k`, `recall@k`, `all_targets_hit@k`, and `rr` (reciprocal rank, averaged into
  MRR@10);
- the first relevant rank within the stored depth, and each target's first rank;
- the stored top 50: each result's rank, chunk id, season, episode, row span and score, each
  retriever's rank for hybrid results, and whether the result is relevant.

Run files store chunk ids and row spans, never chunk text, so a run can be inspected alongside the
local corpus without copying the corpus into the results. To see a chunk, use
`sexandrag inspect <chunk id>`.

## Publishing a result

Raw run directories stay local. To publish a result:

1. Write the summary you want to share (a table, a write-up, a figure) into `results/reports/` or
   `results/figures/`. Cite the run id it came from, and keep dialogue quotations out.
2. Add it explicitly: `git add -f results/reports/<file>`. The `-f` is the deliberate step: the
   directory is ignored, so a plain `git add -A` never picks it up.
3. `make check` still applies. The repository-hygiene check accepts a deliberately added report or
   figure, but it rejects raw run directories, arrays, model files and anything over 2 MB.

A development-set result can be published at any time. A held-out test result is published only
from the single final run described in [docs/evaluation.md](../docs/evaluation.md).
