"""Optional heuristic scene grouping in src/scenes.py."""
import copy

from src.scenes import detect_scenes, scene_numbers
from tests.helpers import episode_lines


def alternating(names: tuple[str, str], n: int, topic: str) -> list[tuple[str, str]]:
    """n lines alternating between two speakers."""
    return [(names[i % 2], f"{topic} line {i}") for i in range(n)]


def assert_partition(scenes: list[dict], n_lines: int) -> None:
    """Scenes must cover every line exactly once, in order, and be marked inferred."""
    assert scenes[0]["line_start"] == 0 and scenes[-1]["line_end"] == n_lines
    for a, b in zip(scenes, scenes[1:]):
        assert a["line_end"] == b["line_start"]
    assert [s["scene"] for s in scenes] == list(range(1, len(scenes) + 1))
    assert all(s["inferred"] is True for s in scenes)


def test_speaker_turnover_starts_a_new_scene():
    lines = episode_lines(1, 1, alternating(("Carrie", "Big"), 12, "dinner")
                          + alternating(("Samantha", "Richard"), 12, "office"))
    scenes = detect_scenes(lines, window=6, max_similarity=0.25, min_lines=6)
    assert_partition(scenes, len(lines))
    assert [s["line_start"] for s in scenes] == [0, 12]
    assert scenes[1]["boundary_signals"] == ["speaker_turnover"]


def test_transition_phrase_starts_a_new_scene():
    talk = alternating(("Carrie", "Miranda"), 10, "brunch")
    lines = episode_lines(1, 1, talk + [("Carrie", "Later that night, I couldn't sleep.")] + talk)
    scenes = detect_scenes(lines, window=6, max_similarity=0.25, min_lines=6)
    assert [s["line_start"] for s in scenes] == [0, 10]
    assert "transition_phrase" in scenes[1]["boundary_signals"]


def test_no_signal_means_one_scene():
    lines = episode_lines(1, 1, alternating(("Carrie", "Miranda"), 30, "brunch"))
    assert len(detect_scenes(lines)) == 1


def test_scenes_respect_the_minimum_length():
    lines = episode_lines(1, 1, alternating(("A", "B"), 7, "x") + alternating(("C", "D"), 7, "y")
                          + alternating(("E", "F"), 7, "z"))
    scenes = detect_scenes(lines, window=3, max_similarity=0.25, min_lines=6)
    assert_partition(scenes, len(lines))
    assert all(s["n_lines"] >= 6 for s in scenes)


def test_detection_is_deterministic_and_does_not_touch_the_lines():
    lines = episode_lines(2, 5, alternating(("Carrie", "Big"), 15, "a") + alternating(("Charlotte", "Trey"), 15, "b"))
    before = copy.deepcopy(lines)
    assert detect_scenes(lines) == detect_scenes(lines)
    assert lines == before


def test_scene_numbers_label_every_line():
    lines = episode_lines(1, 1, alternating(("Carrie", "Big"), 12, "a") + alternating(("Samantha", "Richard"), 12, "b"))
    numbers = scene_numbers(detect_scenes(lines, window=6, max_similarity=0.25, min_lines=6), len(lines))
    assert numbers == [1] * 12 + [2] * 12
