"""Conservative text cleaning and turn splitting (sexandrag.clean)."""

import pytest

from sexandrag.clean import clean_row_text


def texts(raw: str) -> list[str]:
    """Just the cleaned turn texts."""
    return [text for text, _ in clean_row_text(raw)]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("- It is?", "It is?"),
        ("-Yes? ", "Yes?"),
        ("- - That's yours.", "That's yours."),
        ("- 900", "900"),
        ("-900", "-900"),  # could be a minus sign: kept
        ("Do you know him? - ", "Do you know him?"),
        ("Oh, I --", "Oh, I --"),  # interrupted speech: kept
    ],
)
def test_subtitle_dashes(raw, expected):
    assert texts(raw) == [expected]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("It’s just boxes, “right”…", 'It\'s just boxes, "right"...'),
        ("You know¡¦", "You know..."),
        ("the book ¨Avenue B¨.", 'the book "Avenue B".'),
        ("–You have no idea.", "You have no idea."),
        ("That's the ''before'' picture.", 'That\'s the "before" picture.'),
    ],
)
def test_typography_is_normalized(raw, expected):
    assert texts(raw) == [expected]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("So what am l?", "So what am I?"),
        ("ls she married?", "Is she married?"),
        ("ln 50 years, men are obsolete.", "In 50 years, men are obsolete."),
        ("It's spelled D-l-V-O-R-C-E.", "It's spelled D-I-V-O-R-C-E."),
        ("Have you all had an AlDS test?", "Have you all had an AIDS test?"),
        ("There's a VlP room.", "There's a VIP room."),
        ("Lola, I really like all of it. Ill-timed lady.", "Lola, I really like all of it. Ill-timed lady."),
    ],
)
def test_ocr_fixes_are_narrow(raw, expected):
    assert texts(raw) == [expected]


def test_spacing_fixes():
    assert texts("She never missed a show.She was there.") == ["She never missed a show. She was there."]
    assert texts("NewYork City - dreary, gray.") == ["New York City - dreary, gray."]
    assert texts("at 7:00 a.m. sharp") == ["at 7:00 a.m. sharp"]


def test_orphan_closing_quote_at_row_start_is_removed():
    assert texts("\" No one has breakfast at Tiffany's.") == ["No one has breakfast at Tiffany's."]
    assert texts('"Hello," she said.') == ['"Hello," she said.']


def test_two_turns_are_split_and_marked():
    turns = clean_row_text("- Hello? - Carrie, it's Stanford.")
    assert [t for t, _ in turns] == ["Hello?", "Carrie, it's Stanford."]
    assert all("split_turns" in rules for _, rules in turns)


@pytest.mark.parametrize(
    "raw",
    [
        "The question remains-- Is this really a company we want to own?",  # no ' - ' turn marker
        "NewYork City - dreary, gray, miserable.",  # dash not after end punctuation
        "- Can I come? -  -",  # second piece is empty
    ],
)
def test_rows_without_two_real_turns_are_not_split(raw):
    assert len(clean_row_text(raw)) == 1


def test_punctuation_only_piece_is_dropped_before_counting_turns():
    assert texts("! - Is that a euphemism for tacky?") == ["Is that a euphemism for tacky?"]


@pytest.mark.parametrize("raw", ['"', ".", "- ", "   "])
def test_rows_without_letters_or_digits_have_no_turns(raw):
    assert clean_row_text(raw) == []


def test_clean_dialogue_is_left_untouched():
    raw = "In case rapists come in my bedroom window I see them in the mirror and then, phew, I'm gone."
    assert clean_row_text(raw) == [(raw, [])]
