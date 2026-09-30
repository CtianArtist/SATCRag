"""The one-time dev/test split (evaluation.split): quotas, determinism, balance."""

import random

import pytest

from sexandrag.errors import EvaluationError
from sexandrag.evaluation.split import choose_dev, imbalance, type_quotas


def synthetic_strata(n_per_type=10):
    """Sixty items across six types with rotating styles, seasons, target counts and overlap bands."""
    types = ("who_said", "scene_fact", "character", "episode", "sequence", "arc")
    strata = {}
    for t_index, qtype in enumerate(types):
        for i in range(n_per_type):
            strata[f"{qtype}-{i:02d}"] = {
                "question_type": qtype,
                "style": ("lexical", "paraphrase", "event")[(i + t_index) % 3],
                "seasons": [1 + (i + t_index) % 6],
                "targets": "multi" if qtype in ("sequence", "arc") else "single",
                "overlap": ("low", "medium", "high")[i % 3],
            }
    return strata


def test_type_quotas_are_proportional_and_fill_the_dev_set():
    quotas = type_quotas(synthetic_strata(), 20, random.Random(1))
    assert sum(quotas.values()) == 20
    assert sorted(quotas.values()) == [3, 3, 3, 3, 4, 4]  # 10 of 60 each -> 3.33 dev places per type


def test_the_split_meets_every_type_quota_and_partitions_the_items():
    strata = synthetic_strata()
    dev, _ = choose_dev(strata, seed=7, dev_size=20, candidates=200)
    assert len(dev) == len(set(dev)) == 20
    assert set(dev) <= set(strata)
    per_type = {
        t: sum(1 for i in dev if strata[i]["question_type"] == t) for t in {s["question_type"] for s in strata.values()}
    }
    assert set(per_type.values()) <= {3, 4}


def test_the_split_depends_only_on_the_seed_and_metadata():
    strata = synthetic_strata()
    first = choose_dev(strata, seed=7, dev_size=20, candidates=200)
    assert choose_dev(strata, seed=7, dev_size=20, candidates=200) == first
    assert choose_dev(dict(reversed(list(strata.items()))), seed=7, dev_size=20, candidates=200)[0] == first[0]
    assert choose_dev(strata, seed=8, dev_size=20, candidates=200)[0] != first[0]


def test_more_candidates_never_give_a_less_balanced_split():
    strata = synthetic_strata()
    assert choose_dev(strata, 3, 20, 500)[1] <= choose_dev(strata, 3, 20, 5)[1]


def test_imbalance_is_zero_for_a_perfectly_proportional_dev_set():
    strata = {
        f"q{i}": {"style": s, "seasons": [1], "targets": "single", "overlap": "low"} for i, s in enumerate("aabb")
    }
    assert imbalance(["q0", "q2"], strata) == 0.0
    assert imbalance(["q0", "q1"], strata) > 0.0


def test_an_impossible_dev_size_is_rejected():
    with pytest.raises(EvaluationError, match="dev set size"):
        choose_dev(synthetic_strata(1), seed=1, dev_size=6, candidates=10)
