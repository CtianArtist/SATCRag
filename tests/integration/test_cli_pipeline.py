"""The whole pipeline through the real CLI on a tiny synthetic corpus: no SATC data, no model, no network."""

import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from sexandrag.cli import main
from tests.conftest import REPO_ROOT
from tests.support.fakes import FakeServices


def cli(root, *args, services=None):
    """Run the CLI against a project root; return the exit code."""
    return main(["--root", str(root), *args], services=services or FakeServices())


@pytest.fixture
def project(synthetic_root):
    """The synthetic project parsed, chunked, indexed and with its small benchmark frozen and split."""
    services = FakeServices()
    assert cli(synthetic_root, "parse", services=services) == 0
    assert cli(synthetic_root, "chunk", "--all", services=services) == 0
    assert cli(synthetic_root, "index", "--all", services=services) == 0
    assert (
        cli(synthetic_root, "freeze", str(synthetic_root / "eval" / "synthetic_benchmark.json"), services=services) == 0
    )
    return synthetic_root, services


def run_dirs(root):
    runs = root / "results" / "runs"
    return sorted(runs.iterdir()) if runs.is_dir() else []


def test_the_pipeline_builds_every_artifact(project):
    root, _ = project
    assert (root / "data" / "processed" / "lines.jsonl").is_file()
    for config_id in ("tok40o8-regex", "tok80o16-regex"):
        assert (root / "data" / "processed" / "chunks" / f"{config_id}.manifest.json").is_file()
        assert len(list((root / "index" / "dense" / config_id).glob("*/vectors.npy"))) == 1
    assert {p.name for p in (root / "eval" / "frozen").iterdir()} == {
        "benchmark.json",
        "dev.json",
        "test.json",
        "MANIFEST.json",
    }


def test_rerunning_index_reuses_the_cache_without_the_model(project, capsys):
    root, _ = project
    services = FakeServices()
    assert cli(root, "index", "--all", services=services) == 0
    assert services.embedder_loads == 0
    assert "cached and valid" in capsys.readouterr().out


def test_search_prints_ranked_chunks_with_provenance(project, capsys):
    root, _ = project
    assert cli(root, "search", "greyhound slipped his collar near the fountain", "--method", "hybrid", "-k", "3") == 0
    output = capsys.readouterr().out
    assert "s01e02-tok40o8" in output
    assert "rows" in output
    assert "[bm25 #" in output


def test_bm25_search_never_loads_the_model(project):
    root, _ = project
    services = FakeServices()
    assert cli(root, "search", "cobalt harbor", "--method", "bm25", services=services) == 0
    assert services.embedder_loads == 0


def test_inspect_shows_one_chunk_and_rejects_unknown_ids(project, capsys):
    root, _ = project
    assert cli(root, "inspect", "s01e01-tok40o8-000") == 0
    assert '"chunk_id": "s01e01-tok40o8-000"' in capsys.readouterr().out
    assert cli(root, "inspect", "s09e09-tok40o8-000") == 1
    assert cli(root, "inspect", "not-a-chunk-id") == 1
    assert "Traceback" not in capsys.readouterr().err


def test_eval_defaults_to_the_development_set(project):
    root, _ = project
    assert cli(root, "eval") == 0
    (run_dir,) = run_dirs(root)
    run = json.loads((run_dir / "run.json").read_text())
    assert run["eval"]["split"] == "dev"
    assert run["eval"]["held_out"] is False
    assert run["eval"]["n_items"] == 3
    assert {p.name for p in run_dir.iterdir()} == {
        "run.json",
        "config.json",
        "environment.json",
        "metrics.json",
        "queries.jsonl",
    }
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert set(metrics["cells"]) == {f"{size}/{m}" for size in (40, 80) for m in ("bm25", "dense", "hybrid")}
    assert "all_targets_hit@10" in metrics["cells"]["40/hybrid"]["aggregate"]["all"]
    config = json.loads((run_dir / "config.json").read_text())
    assert config["settings"]["retrieval"]["rrf_k"] == 60
    assert config["hashes"]["eval_file"]
    first_row = json.loads((run_dir / "queries.jsonl").read_text().splitlines()[0])
    assert "text" not in first_row["retrieved"][0]  # results never store corpus text


def test_the_test_split_is_refused_without_the_explicit_opt_in(project, capsys):
    root, services = project
    loads = services.embedder_loads
    assert cli(root, "eval", "--split", "test", services=services) == 1
    assert "--allow-heldout" in capsys.readouterr().err
    assert run_dirs(root) == []
    assert services.embedder_loads == loads


def test_the_test_split_runs_only_with_the_opt_in(project, capsys):
    root, _ = project
    assert cli(root, "eval", "--split", "test", "--allow-heldout") == 0
    (run_dir,) = run_dirs(root)
    assert json.loads((run_dir / "run.json").read_text())["eval"]["held_out"] is True
    assert "HELD-OUT TEST RUN" in capsys.readouterr().out


def test_a_copy_of_the_test_set_is_refused_too(project, tmp_path):
    root, _ = project
    copy = shutil.copyfile(root / "eval" / "frozen" / "test.json", tmp_path / "questions.json")
    assert cli(root, "eval", "--eval-file", str(copy)) == 1
    assert run_dirs(root) == []


def test_malformed_eval_items_are_rejected_at_load_before_any_model_work(project, tmp_path, capsys):
    root, services = project
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps([{"id": "x", "question": "Which episode?", "season": 1, "episode": 1}]))
    loads = services.embedder_loads
    assert cli(root, "eval", "--eval-file", str(bad), services=services) == 1
    assert "no source rows" in capsys.readouterr().err
    assert services.embedder_loads == loads
    assert run_dirs(root) == []


def test_repeat_runs_are_deterministic_and_never_overwrite_each_other(project):
    root, _ = project
    assert cli(root, "eval") == 0
    assert cli(root, "eval") == 0
    first, second = run_dirs(root)
    assert first != second

    def comparable(run_dir):
        rows = [json.loads(line) for line in (run_dir / "queries.jsonl").read_text().splitlines()]
        cells = json.loads((run_dir / "metrics.json").read_text())["cells"]
        return rows, {name: cell["aggregate"] for name, cell in cells.items()}

    assert comparable(first) == comparable(second)


def test_verify_passes_then_catches_corruption_and_missing_artifacts(project, capsys):
    root, _ = project
    only = ["verify", "--only", "corpus,metadata,lines,chunks,index,benchmark"]
    assert cli(root, *only) == 0
    capsys.readouterr()
    vectors = next((root / "index" / "dense" / "tok40o8-regex").glob("*/vectors.npy"))
    np.save(vectors, np.load(vectors) * 2)  # no longer unit length
    assert cli(root, *only) == 1
    assert "FAILED   index" in capsys.readouterr().out
    shutil.rmtree(vectors.parent)
    assert cli(root, *only) == 0  # a missing index is not a failure...
    assert cli(root, *only, "--strict") == 1  # ...unless --strict


def test_changed_lines_make_the_chunks_stale(project, capsys):
    root, _ = project
    lines = root / "data" / "processed" / "lines.jsonl"
    lines.write_text(lines.read_text() + lines.read_text().splitlines()[0] + "\n")
    assert cli(root, "search", "sourdough", "--method", "bm25") == 1
    assert "stale" in capsys.readouterr().err


def test_a_missing_corpus_is_an_actionable_error(synthetic_root, capsys):
    (synthetic_root / "data" / "raw" / "SATC_all_lines.csv").unlink()
    assert cli(synthetic_root, "parse") == 1
    err = capsys.readouterr().err
    assert "sexandrag download" in err
    assert "Traceback" not in err


def test_a_changed_corpus_is_refused_unless_explicitly_allowed(synthetic_root, capsys):
    raw = synthetic_root / "data" / "raw" / "SATC_all_lines.csv"
    raw.write_text(raw.read_text() + '99,1.0,1.0,Nora,"One more line.",\n')
    assert cli(synthetic_root, "parse") == 1
    assert "--allow-unverified-corpus" in capsys.readouterr().err
    assert cli(synthetic_root, "parse", "--allow-unverified-corpus") == 0


def test_config_is_shown_and_bad_config_fails_early(synthetic_root, capsys):
    assert cli(synthetic_root, "config", "show") == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["chunking"]["configs"] == [[40, 8], [80, 16]]
    (synthetic_root / "sexandrag.toml").write_text("[retrieval]\nrrf_k = 'sixty'\n")
    assert cli(synthetic_root, "config", "show") == 1
    assert "wrong type" in capsys.readouterr().err


def test_usage_errors_exit_2():
    with pytest.raises(SystemExit) as exit_info:
        main(["no-such-command"])
    assert exit_info.value.code == 2


class ExplodingServices(FakeServices):
    """Services whose model load fails with an unexpected (non-SexAndRag) error."""

    def embedder(self, settings):
        raise RuntimeError("simulated crash")


def test_unexpected_errors_are_summarised_without_a_traceback_unless_debug(project, capsys):
    root, _ = project
    assert cli(root, "search", "sourdough", "--method", "dense", services=ExplodingServices()) == 70
    err = capsys.readouterr().err
    assert "--debug" in err
    assert "Traceback" not in err
    with pytest.raises(RuntimeError, match="simulated crash"):
        main(["--root", str(root), "--debug", "search", "sourdough", "--method", "dense"], services=ExplodingServices())


def test_validate_scenes_and_find_commands(project, tmp_path, capsys):
    root, _ = project
    review = tmp_path / "review.md"
    assert cli(root, "validate", str(root / "eval" / "synthetic_benchmark.json"), "--review", str(review)) == 0
    assert review.is_file()
    assert "10/10 items valid" in capsys.readouterr().out
    assert cli(root, "scenes", "--show", "S1E1", "--limit", "5") == 0
    assert (root / "data" / "processed" / "scenes_inferred.jsonl").is_file()
    assert cli(root, "find", "wood-fired oven") == 0
    assert '"source_row_start": 2' in capsys.readouterr().out


def test_model_commands_work_offline_and_explain_a_missing_model(synthetic_root, capsys):
    config = synthetic_root / "sexandrag.toml"
    config.write_text(config.read_text() + '\n[paths]\nmodel_cache_dir = "empty-hf-cache"\n')
    assert cli(synthetic_root, "model", "info") == 0
    assert '"pooling_mode": "cls"' in capsys.readouterr().out
    assert cli(synthetic_root, "model", "verify") == 1
    assert "sexandrag model download" in capsys.readouterr().err
    assert cli(synthetic_root, "verify", "--only", "model") == 0  # missing is reported, not a failure...
    assert "MISSING  model" in capsys.readouterr().out
    assert cli(synthetic_root, "verify", "--only", "model", "--strict") == 1  # ...unless --strict


def test_parse_can_print_the_full_report(synthetic_root, capsys):
    assert cli(synthetic_root, "parse", "--report") == 0
    output = capsys.readouterr().out
    assert "Tuple-encoded rows unpacked: 10" in output
    assert "'Sam'" in output


def test_unknown_verify_components_and_malformed_episode_codes_are_rejected(project, capsys):
    root, _ = project
    assert cli(root, "verify", "--only", "corpus,moon") == 1
    assert "unknown verify component" in capsys.readouterr().err
    assert cli(root, "scenes", "--show", "episode-1") == 1
    assert "episode code" in capsys.readouterr().err


def test_a_second_freeze_never_overwrites_the_first(project):
    root, _ = project
    before = (root / "eval" / "frozen" / "MANIFEST.json").read_bytes()
    assert cli(root, "freeze", str(root / "eval" / "synthetic_benchmark.json")) == 1
    assert (root / "eval" / "frozen" / "MANIFEST.json").read_bytes() == before


def test_the_legacy_entry_point_still_means_development_only():
    done = subprocess.run(
        [sys.executable, "-m", "eval.run_eval", "--split", "test"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 1
    assert "--allow-heldout" in done.stderr
    helped = subprocess.run(
        [sys.executable, "-m", "eval.run_eval", "--help"], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert helped.returncode == 0
    assert "--allow-heldout" in helped.stdout
