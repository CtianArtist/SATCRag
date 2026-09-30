"""Speaker-label normalization in src/speakers.py."""
from collections import Counter

import pytest

from src import config
from src.speakers import AliasMap, build_case_map, load_aliases, normalize_speaker, tidy_label

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
    assert norm("CArrie")[0] == "Carrie" and norm("carrie")[0] == "Carrie"
    assert "case_variant" in norm("CArrie")[2]


def test_jack_is_berger_only_in_the_listed_episodes():
    assert norm("Jack", 5, 5)[:2] == ("Jack Berger", ["Jack Berger"])
    assert norm("Jack", 1, 8)[0] == "Jack"


def test_sam_is_never_mapped_globally():
    assert norm("SAm", 1, 4)[0] == "Sam"               # S1E4: Samantha's waiter date
    assert norm("Sam", 2, 7)[0] == "Samantha"          # episode-scoped alias


def test_names_are_matched_whole_not_by_substring():
    assert norm("Carrie Fisher")[0] == "Carrie Fisher"
    assert norm("Stanford Blatch")[0] == "Stanford"


def test_multi_speaker_labels_list_every_name():
    speaker, speakers, rules = norm("Carrie, Samantha and Miranda")
    assert speaker == "Carrie & Samantha & Miranda"
    assert speakers == ["Carrie", "Samantha", "Miranda"]
    assert norm("Samanth and Charlotte")[1] == ["Samantha", "Charlotte"]
    assert "multi_speaker" in rules


@pytest.mark.parametrize("label, expected", [
    ("Woman #1", "Woman 1"), ("Woman1", "Woman 1"), ("Woman # 1", "Woman 1"),
    ("2nd NYSE Official", "2nd NYSE Official"), ("(All)", "All"), ("  Dave ", "Dave"),
])
def test_generic_label_cleanup(label, expected):
    assert tidy_label(label)[0] == expected


def test_blank_label_means_unknown_speaker():
    assert norm("  ") == (None, [], ["whitespace"])


def test_real_alias_file_loads_with_scoped_entries():
    aliases = load_aliases(config.SPEAKER_ALIASES_JSON)
    assert aliases.scoped_aliases[(6, 3, "Jack")] == "Jack Berger"
    assert (1, 8, "Jack") not in aliases.scoped_aliases
    assert "Sam" not in aliases.global_aliases and "Jack" not in aliases.global_aliases
