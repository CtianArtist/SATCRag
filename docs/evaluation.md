# Evaluation

This document covers the benchmark schema, how relevance and the metrics are computed, the frozen
dev/test split, and the rules that protect the held-out test set.

## What is measured

**Retrieval, not answer generation.**

- **Retrieval evaluation** asks: for a question, do the top-ranked chunks contain the source lines
  that answer it? This is what the benchmark scores.
- **Answer-generation evaluation** asks: given retrieved chunks, does a language model write a correct
  and faithful answer? Nothing here measures it.

The benchmark isolates retrieval so that BM25, dense and hybrid retrieval, and the three chunk sizes,
can be compared on exactly the same ground truth. Generation will be evaluated separately, later,
on top of a frozen retrieval system.

## Relevance: row spans, never episodes

- **Spans.** Ground truth is a set of source-row spans. A span is an inclusive range of rows in one
  episode, where rows are the CSV's own per-episode row index, kept on every parsed line and chunk.
- **The rule.** A retrieved chunk is relevant when both of these hold:
  - it comes from the same season and episode as a scored target;
  - its row range, from `source_row_start` to `source_row_end`, overlaps that target's rows.
- **Why it works across chunk sizes.** Relevance depends only on rows, so one ground truth scores
  256-, 512- and 1,024-token chunks alike.
- **No episode-wide relevance.** A chunk from the right episode that misses the answer-bearing rows
  is a miss. A target without rows is rejected when the file is loaded.

## The item schema

An item is a question, its **scored evidence** and, optionally, **premise evidence**. The examples
below come from the invented fixture corpus in `tests/fixtures/synthetic/`. After the first example,
fields that do not concern evidence (type, style, expected answer) are left out for brevity.

### A single target, with a premise span

```json
{
  "id": "syn-fact-1",
  "question_type": "scene_fact",
  "style": "paraphrase",
  "question": "Where was the missing greyhound found?",
  "expected_answer": "Asleep under the bakery counter, next to the flour sacks (S1E2)",
  "season": 1, "episode": 2, "source_row_start": 13, "source_row_end": 13,
  "expected_quote": "asleep under the counter",
  "evidence": [
    {"season": 1, "episode": 2, "source_row_start": 0, "source_row_end": 0,
     "expected_quote": "a grey greyhound named Pickle", "role": "premise"}
  ]
}
```

### Several targets, all required (a sequence question's anchor and answer)

```json
{
  "id": "syn-seq-1",
  "question": "Right after the first batch comes out burnt, what does Theo tell Nora to do?",
  "targets": [
    {"season": 1, "episode": 1, "source_row_start": 12, "source_row_end": 14,
     "expected_quote": "The first batch is ruined", "role": "anchor"},
    {"season": 1, "episode": 1, "source_row_start": 15, "source_row_end": 16,
     "expected_quote": "Start again", "role": "answer"}
  ],
  "requires_all_targets": true
}
```

- **Why the anchor is scored.** A sequence question is only answerable when the retriever finds the
  event it refers to ("right after the first batch comes out burnt") as well as what followed it.
  So both spans are required, scored evidence.
- **What `role` does.** It documents what each span contributes. It does not change scoring.

### Alternatives: any one target is enough

```json
{
  "id": "syn-fact-3",
  "question": "What kind of oven does the new bakery have?",
  "targets": [
    {"season": 1, "episode": 1, "source_row_start": 2, "source_row_end": 2, "expected_quote": "wood-fired oven"},
    {"season": 1, "episode": 1, "source_row_start": 3, "source_row_end": 3, "expected_quote": "wood-fired"}
  ],
  "requires_all_targets": false
}
```

### The general form: required groups

`required_groups` combines both modes. The answer is fully supported when every group is satisfied.
This illustrative item needs one of two alternative mentions of the oven *and* the passage about the
first batch:

```json
{
  "id": "example-groups",
  "question": "What kind of oven does the bakery have, and what goes wrong with the first batch?",
  "required_groups": [
    {"mode": "any", "targets": [
      {"season": 1, "episode": 1, "source_row_start": 2, "source_row_end": 2, "expected_quote": "wood-fired oven"},
      {"season": 1, "episode": 1, "source_row_start": 3, "source_row_end": 3, "expected_quote": "wood-fired"}
    ]},
    {"mode": "all", "targets": [
      {"season": 1, "episode": 1, "source_row_start": 12, "source_row_end": 14,
       "expected_quote": "The first batch is ruined"}
    ]}
  ]
}
```

### Schema rules, all enforced when the file is loaded

- **Exactly one form of scored evidence per item:**
  - inline `season`, `episode`, `source_row_start` and `source_row_end`, for a single target;
  - a `targets` list with `requires_all_targets`: `true` makes one `all` group and `false` makes one
    `any` group. The flag is required whenever there are several targets, and refused when there is
    only one;
  - `required_groups`: a non-empty list of `{"mode": "all" | "any", "targets": [...]}`. An `any` group
    needs at least two alternatives.
- **Every target needs integer rows** with `0 <= start <= end`. `expected_quote` and `role` are
  optional strings.
- **Premise spans** go in `evidence`, and every one must have `"role": "premise"`. Scored spans do
  not belong there.
- **Required fields:** `id`, which must be unique within the file, and `question`. `question_type`
  and `style` are optional in the schema but must come from the known sets:
  - question types: `who_said`, `scene_fact`, `character`, `episode`, `sequence`, `arc`;
  - styles: `lexical`, `paraphrase`, `relationship`, `event`, `multi_evidence`, `metadata`.
- **All problems at once.** Every problem in a file is reported together, and nothing runs until the
  file is valid. Schema errors can never surface halfway through a benchmark run.
- **Stricter checks for benchmark items.** `satc-rag validate` requires a `question_type` and a
  `style`. It also checks that every scored and premise span names an existing episode and rows, and
  that each `expected_quote` lies inside its span.

## Scored evidence and premise evidence

- **Scored evidence** is the rows that carry the answer, including any anchor the question depends on.
  Only scored targets decide relevance and the metrics.
- **Premise evidence** is the rows that justify the question's wording, for example the line
  establishing that a dog went missing. Premise spans are validated, so a reviewer can check the
  premise against the source. They are never scored: retrieving a premise span earns nothing, and
  missing one costs nothing.

## What counts as evidence

- **Corpus evidence:** the dialogue text.
- **Corpus metadata evidence:** speaker labels. They are part of the indexed chunk text
  (`Speaker: line`), so the retriever sees them, and a "who said" answer may rest on them.
- **Never evidence:** inferred scenes, episode titles, general knowledge of the show, anything
  visible only on screen, and conclusions the lines merely suggest. Questions answerable only from
  metadata such as episode titles belong in `eval/probes.json`, not in the benchmark.
- **The cited lines must state every claim** in a question's premise and in its expected answer. If a
  detail is not in the cited rows, either the citation is widened or the detail is dropped.
- **Spans are as small as possible** while still supporting the answer.
- **`expected_quote` is a few words,** enough to check provenance. The eval files must not become a
  way to redistribute the scripts.

## Metrics

- **Requirements.** An item's scored evidence breaks down into independent requirements:
  - each target of an `all` group is one requirement;
  - an `any` group is one requirement, satisfied by any of its alternatives.
- **Hit@1/5/10:** any scored target is reached in the top k.
- **Recall@1/5/10:** the fraction of requirements reached in the top k. For a single requirement it
  equals Hit@k.
- **AllTargetsHit@5/10:** every requirement is reached in the top k. It applies only to items that
  need several requirements. For other items it is empty, and averages skip it; the count is reported
  as `n_all_targets`.
- **MRR@10:** 1 / rank of the first relevant chunk in the top 10, or 0 when there is none.
- **Worked example.** An arc question needs three passages, and two of them are in the top 10:
  Hit@10 = 1, Recall@10 = 2/3, AllTargetsHit@10 = 0.
  For the `required_groups` example above, suppose a top-5 list reaches row 3 but not rows 12 to 14.
  Then Hit@5 = 1, Recall@5 = 1/2 and AllTargetsHit@5 = 0.

### Reporting

- **Aggregates:** results are averaged over all questions and per question type, for each
  (chunk size, retriever) cell.
- **Stored per query:** each question's top 50 is kept, with relevance flags, the first relevant rank
  within that depth, and the first rank at which each target was reached. Only chunk ids and row spans
  are stored, never chunk text.
- **Comparisons for failure analysis,** at Hit@5:
  - per chunk size: which questions both retrievers hit, only BM25 hit, only dense hit, or both
    missed; and which ones hybrid recovered (hit when both others missed) or regressed (missed when
    either hit);
  - per retriever: which questions are sensitive to chunk size, and how the first relevant rank
    moves across sizes;
  - multi-evidence questions that found some but not all of their evidence.
- **Quote drift:** before scoring, the runner warns if any `expected_quote` has drifted outside its
  span, which would indicate parser drift.

Run outputs are described in [results/README.md](../results/README.md).

## The frozen benchmark

| File | Contents |
|---|---|
| `eval/frozen/benchmark.json` | the 60 approved questions |
| `eval/frozen/dev.json` | the development set: 20 questions |
| `eval/frozen/test.json` | the held-out test set: 40 questions |
| `eval/frozen/MANIFEST.json` | the SHA-256 and item count of every file; the corpus hashes the targets were written against; the source proposal's hash; the split seed, settings and dev ids; each item's strata; the intended use of each file |

- **Question types:** 10 questions of each of the six types.
- **Styles:**
  - 23 paraphrase;
  - 10 event;
  - 10 multi-evidence;
  - 9 lexical;
  - 8 relationship.

  The target mix was roughly lexical 15%, paraphrase 35 to 40%, event 20%, relationship 10 to 15%
  and multi-evidence 15 to 20%.
- **Evidence structure:** 100 scored spans and 24 premise spans.
  - 31 questions have a single target.
  - 18 need two targets, 9 need three, and 1 needs four.
  - 1 accepts either of two alternatives.
  - So 28 questions need more than one piece of evidence, and AllTargetsHit applies to them.
- **How the questions were built.**
  - Candidates were drafted from seeded random samples of the corpus, with proposed evidence.
  - No retriever was run to choose or keep any question.
  - A human reviewer approved every item, and valid candidates that did not fit the mix went to
    `eval/candidates_reserve.json`.
  - The reviewed proposal, `eval/benchmark_proposed.json`, is the source of the freeze. Its hash is
    in the manifest.

### How the split was made

`satc-rag freeze` made the split once, from item metadata only:

1. **Type quotas.** Every question type receives its proportional share of the 20 development places.
2. **Candidate splits.** Within those quotas, 20,000 random splits are drawn from seed `20261001`.
3. **Selection.** The first split whose development set best matches the whole benchmark is kept. It
   is judged on style, season, single versus multi-target, and lexical-overlap band (how many of a
   question's content words appear in its scored text).

No retrieval result was consulted, so the split cannot have been shaped toward any retriever.

### Checking the freeze

- **What is checked:** `satc-rag verify --only benchmark` re-checks every file hash, and confirms that
  the recorded seed and strata still reproduce the same split.
- **Where it runs:** in `make check` and in CI. It needs neither the corpus nor the model.
- **The freeze is permanent.** `freeze` refuses to overwrite an existing freeze, and it made the files
  read-only. The frozen files are never edited. A revised benchmark would be a new, separately frozen
  version, and its results would not be comparable with this one.

## Test-set discipline

**The order of work:**

1. Run the untouched baseline on the development set.
2. Analyse the failures on the development set.
3. Tune, if at all, on the development set only. That covers RRF parameters, hybrid candidate depth,
   BM25 preprocessing and reranker configuration.
4. Finalise the retrieval approach.
5. Only then, run the held-out test set once.

**The rules:**

- **The test set is never used for tuning.** Test-set failures are not inspected while decisions are
  being made, and results are not tuned repeatedly against held-out scores.
- **Frozen targets are never changed.**
- **Baseline parameters stay fixed until tuning starts on the development set:** BM25 k1 1.5 and b 0.75,
  hybrid candidate depth 30, RRF k 60.
- **No change is ever made because of how individual questions behave.** That covers preprocessing,
  chunking, BM25, dense retrieval and fusion. No code may special-case a question, episode, character
  or quote.

**How the code enforces it:**

- `satc-rag eval` and `python -m eval.run_eval` evaluate the development set.
- `--split test` is refused unless `--allow-heldout` is also given.
- The guard also refuses any `--eval-file` that touches the test set, however it is named:
  - the frozen test file itself, found by path;
  - a byte-identical copy of it, found by SHA-256;
  - any file containing a test item's id;
  - any file containing a test question under another id.
- **An allowed held-out run** logs a warning and is recorded with `"held_out": true` in its
  `run.json`. A `--split test` run is also labelled `test` in its run id.
- **`make check` and CI never run a retrieval benchmark,** on either split.

## Writing and checking eval items

- **Finding evidence:** `satc-rag find "a few words" [--season N --episode N]` locates a quote in the
  parsed lines and prints a paste-ready target.
- **Validating:** `satc-rag validate FILE --review data/processed/eval_review_<name>.md` checks the
  schema and the provenance, and prints the set's distribution:
  - question types, styles, seasons and characters;
  - target and premise counts, and span lengths;
  - question-to-evidence word overlap, and rare shared words.
- **The review sheet** shows each question beside its cited lines. It quotes the corpus, so it stays
  under `data/processed/`, which Git ignores.

## Other eval files

| File | Purpose |
|---|---|
| `eval/smoke.json` | Five plumbing-check items, drawn from episodes the benchmark does not use. Between them they exercise every evidence form: a single target, an `any` group, an `all` anchor and answer with a premise span, and `required_groups` mixing `any` and `all`. Not a benchmark. |
| `eval/probes.json` | Metadata and routing probes, such as questions answerable only from episode titles. Kept out of the retrieval benchmark. |
| `eval/candidates_reserve.json` | Valid approved-quality alternates that were not selected. |
| `eval/benchmark_proposed.json` | The reviewed proposal the freeze was made from. |
| `eval/drafts/candidates_v1.json` | The superseded first draft, kept for reference. |
