"""Benchmark-item validation against a corpus (evaluation.validate)."""

from sexandrag.evaluation.schema import parse_item
from sexandrag.evaluation.validate import (
    by_episode,
    check_item,
    lexical_overlap,
    overlapping_targets,
    percentile,
    rare_words,
    shared_rare_words,
    validate_raw_items,
)
from tests.support.builders import episode_lines

EPISODES = by_episode(
    episode_lines(
        4,
        13,
        [
            ("Carrie", "Hello, Petey. It's just boxes, right?"),
            ("Carrie", "A plant! The man brought a living thing into my apartment."),
            ("Miranda", "You said yes to moving in, he moved in."),
        ],
    )
)
PREMISE = {
    "season": 4,
    "episode": 13,
    "source_row_start": 1,
    "source_row_end": 1,
    "expected_quote": "brought a living thing",
    "role": "premise",
}


def item(**fields):
    """A who_said item pointing at S4E13 unless overridden."""
    return {
        "id": "x",
        "question": "Who says the man brought a plant into her apartment?",
        "question_type": "who_said",
        "style": "paraphrase",
        "expected_answer": "Carrie",
        "season": 4,
        "episode": 13,
        **fields,
    }


def test_a_well_formed_item_passes():
    assert (
        check_item(item(source_row_start=1, source_row_end=1, expected_quote="brought a living thing"), EPISODES) == []
    )


def test_quote_outside_its_span_is_flagged_as_moved():
    problems = check_item(item(source_row_start=2, source_row_end=2, expected_quote="brought a living thing"), EPISODES)
    assert problems
    assert "moved" in problems[0]


def test_missing_rows_episode_quote_and_labels_are_flagged():
    assert "outside" in check_item(item(source_row_start=1, source_row_end=99, expected_quote="plant"), EPISODES)[0]
    assert "no expected_quote" in check_item(item(source_row_start=1, source_row_end=1), EPISODES)[0]
    assert any(
        "no episode" in p
        for p in check_item(item(episode=14, source_row_start=0, source_row_end=0, expected_quote="x"), EPISODES)
    )
    assert (
        "unknown question_type"
        in check_item(
            item(question_type="trivia", source_row_start=0, source_row_end=0, expected_quote="Petey"), EPISODES
        )[0]
    )
    no_labels = {
        k: v
        for k, v in item(source_row_start=0, source_row_end=0, expected_quote="Petey").items()
        if k not in ("question_type", "style")
    }
    assert check_item(no_labels, EPISODES) == [
        "missing question_type (one of who_said, scene_fact, character, episode, sequence, arc)",
        "missing style",
    ]


def test_episode_questions_need_scored_answer_spans_not_the_whole_episode():
    whole_episode = item(question_type="episode")  # no rows: any chunk of S4E13 would count
    assert "no source rows" in check_item(whole_episode, EPISODES)[0]
    scored = item(
        source_row_start=2, source_row_end=2, expected_quote="he moved in", question_type="episode", evidence=[PREMISE]
    )
    assert check_item(scored, EPISODES) == []
    bad_quote = dict(scored, evidence=[{**PREMISE, "expected_quote": "never said"}])
    assert "missing" in check_item(bad_quote, EPISODES)[0]


def test_lexical_overlap_counts_only_scored_text():
    target = parse_item(item(source_row_start=1, source_row_end=1, expected_quote="plant"))
    assert lexical_overlap(target, EPISODES) == 4 / 5  # says, man, brought, plant, apartment: all but 'says'
    premise_only = parse_item(
        item(source_row_start=2, source_row_end=2, expected_quote="he moved in", evidence=[PREMISE])
    )
    assert lexical_overlap(premise_only, EPISODES) == 0.0  # the plant line is premise, not target


def test_overlapping_targets_between_items_are_reported():
    a = parse_item(item(id="a", source_row_start=0, source_row_end=1, expected_quote="Petey"))
    b = parse_item(item(id="b", source_row_start=1, source_row_end=2, expected_quote="he moved in"))
    c = parse_item(item(id="c", source_row_start=2, source_row_end=2, expected_quote="he moved in"))
    assert overlapping_targets([a, b, c]) == [("a", "b"), ("b", "c")]


def test_duplicate_ids_are_reported():
    good = item(source_row_start=1, source_row_end=1, expected_quote="plant")
    assert validate_raw_items([good, good], EPISODES) == {"x": ["duplicate id"]}


def test_percentile_uses_nearest_rank():
    values = list(range(1, 11))
    assert (percentile(values, 0.1), percentile(values, 0.5), percentile(values, 0.9), percentile(values, 1.0)) == (
        1,
        5,
        9,
        10,
    )


def test_rare_shared_words_are_the_bm25_hooks():
    target = parse_item(item(source_row_start=1, source_row_end=1, expected_quote="plant"))
    assert shared_rare_words(target, EPISODES, rare_words(EPISODES)) == {"man", "brought", "plant", "apartment"}
