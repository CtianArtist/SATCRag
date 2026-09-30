"""Speaker-label normalization and the alias file (sexandrag.speakers)."""

import json
from collections import Counter

import pytest

from sexandrag.errors import MetadataError
from sexandrag.speakers import AliasMap, build_case_map, load_aliases, normalize_speaker, tidy_label

ALIASES = AliasMap(
    global_aliases={"Crri": "Carrie", "Samanth": "Samantha", "Stanford Blatch": "Stanford"},
    scoped_aliases={(5, 5, "Jack"): "Jack Berger", (2, 7, "Sam"): "Samantha"},
)
CASE_MAP = build_case_map(Counter({"Carrie": 100, "CArrie": 1, "carrie": 1, "Sam": 17, "SAm": 8, "SAM": 1}))


def norm(label: str, season: int = 1, episode: int = 1):
    """Normalize with the test alias and case maps."""
    return normalize_speaker(label, season, episode, ALIASES, CASE_MAP)


def test_global_alias_fixes_a_typo():
    assert norm("Crri") == ("Carrie", ["Carrie"], ["alias"])


def test_casing_variants_map_to_the_clean_spelling():
    assert norm("CArrie")[0] == "Carrie"
    assert norm("carrie")[0] == "Carrie"
    assert "case_variant" in norm("CArrie")[2]


def test_jack_is_berger_only_in_the_listed_episodes():
    assert norm("Jack", 5, 5)[:2] == ("Jack Berger", ["Jack Berger"])
    assert norm("Jack", 1, 8)[0] == "Jack"


def test_sam_is_never_mapped_globally():
    assert norm("SAm", 1, 4)[0] == "Sam"
    assert norm("Sam", 2, 7)[0] == "Samantha"


def test_names_are_matched_whole_not_by_substring():
    assert norm("Carrie Fisher")[0] == "Carrie Fisher"
    assert norm("Stanford Blatch")[0] == "Stanford"


def test_multi_speaker_labels_list_every_name():
    speaker, speakers, rules = norm("Carrie, Samantha and Miranda")
    assert speaker == "Carrie & Samantha & Miranda"
    assert speakers == ["Carrie", "Samantha", "Miranda"]
    assert norm("Samanth and Charlotte")[1] == ["Samantha", "Charlotte"]
    assert "multi_speaker" in rules


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("Woman #1", "Woman 1"),
        ("Woman1", "Woman 1"),
        ("Woman # 1", "Woman 1"),
        ("2nd NYSE Official", "2nd NYSE Official"),
        ("(All)", "All"),
        ("  Dave ", "Dave"),
    ],
)
def test_generic_label_cleanup(label, expected):
    assert tidy_label(label)[0] == expected


def test_blank_label_means_unknown_speaker():
    assert norm("  ") == (None, [], ["whitespace"])


def test_the_committed_alias_file_loads_with_scoped_entries(repo_root):
    aliases = load_aliases(repo_root / "data" / "meta" / "speaker_aliases.json")
    assert aliases.scoped_aliases[(6, 3, "Jack")] == "Jack Berger"
    assert (1, 8, "Jack") not in aliases.scoped_aliases
    assert "Sam" not in aliases.global_aliases
    assert "Jack" not in aliases.global_aliases


@pytest.mark.parametrize(
    ("content", "problem"),
    [
        ("{not json", "not valid JSON"),
        ("[]", "JSON object"),
        (json.dumps({"global": [{"from": "A"}]}), "malformed alias entry"),
        (json.dumps({"episode_scoped": [{"from": "Sam", "to": "S", "episodes": ["season 2"]}]}), "bad episode code"),
        (json.dumps({"episode_scoped": [{"from": "Sam", "to": "S"}]}), "must list its episodes"),
    ],
)
def test_malformed_alias_files_are_rejected(tmp_path, content, problem):
    path = tmp_path / "aliases.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(MetadataError, match=problem):
        load_aliases(path)


def test_a_missing_alias_file_is_an_error(tmp_path):
    with pytest.raises(MetadataError, match="not found"):
        load_aliases(tmp_path / "absent.json")
