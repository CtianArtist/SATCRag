"""Conservative text cleaning, one small named rule at a time, so every change is traceable.

Row rules run on the whole CSV text field; the result is then split into speaker turns
(only at a subtitle turn marker: sentence-final punctuation, then " - "); turn rules run on
each turn. No rule rewrites wording, grammar or style.
"""

import re
from collections.abc import Callable

Rule = tuple[str, Callable[[str], str]]

TYPOGRAPHY = [
    ("\u2019", "'"),  # right single quote
    ("\u2018", "'"),  # left single quote
    ("\u201c", '"'),  # left double quote
    ("\u201d", '"'),  # right double quote
    ("\u00a1\u00a6", "..."),  # mis-encoded ellipsis seen as "¡¦"
    ("\u2026", "..."),  # ellipsis character
    ("\u00a8", '"'),  # diaeresis used as a quote mark
    ("\u2013", "-"),  # en dash
    ("\u2014", "-"),  # em dash
    ("''", '"'),  # two apostrophes used as a double quote
]
OCR_L_TOKENS = {"l": "I", "ls": "Is", "ln": "In", "lt": "It", "lf": "If"}  # never English words
OCR_L_TOKEN_RE = re.compile(r"\b(?:l|ls|ln|lt|lf)\b")
CAPS_TOKEN_RE = re.compile(r"\b[A-Zl]{3,}\b")  # e.g. AlDS, VlP, HlV
MERGED_WORDS = {"NewYork": "New York"}
MISSING_SPACE_RE = re.compile(r"(?<=[a-z])([.?!])(?=[A-Z][a-z])")  # "show.She" -> "show. She"
ORPHAN_QUOTE_RE = re.compile(r'^"\s+')  # closing quote left at row start
TURN_SPLIT_RE = re.compile(r'(?<=[.?!"\-])\s+-\s*(?=\S)')  # "Hello? - Carrie, it's..."
LEADING_DASHES_RE = re.compile(r"^-[-\s]*")  # "- It is?", "- - That's yours."
TRAILING_DASH_RE = re.compile(r"\s+-\s*$")  # "Do you know him? -"; keeps "I --"
CONTENT_RE = re.compile(r"[^\W_]")  # any letter or digit


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace (including embedded line breaks) and trim the ends."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_typography(text: str) -> str:
    """Replace curly quotes, ellipsis characters, dash variants and quote look-alikes with ASCII."""
    for old, new in TYPOGRAPHY:
        text = text.replace(old, new)
    return text


def fix_l_as_i(text: str) -> str:
    """Fix OCR tokens where capital I was read as lowercase l ('am l?', 'ls she', 'ln 50 years')."""
    return OCR_L_TOKEN_RE.sub(lambda m: OCR_L_TOKENS[m.group(0)], text)


def fix_caps_l(text: str) -> str:
    """Fix all-caps tokens containing a stray lowercase l ('AlDS' -> 'AIDS', 'VlP' -> 'VIP')."""

    def repair(m: re.Match[str]) -> str:
        token = m.group(0)
        uppers = sum(c.isupper() for c in token)
        return token.replace("l", "I") if "l" in token and uppers >= 2 else token

    return CAPS_TOKEN_RE.sub(repair, text)


def fix_merged_words(text: str) -> str:
    """Split the few words the subtitles ran together ('NewYork' -> 'New York')."""
    for old, new in MERGED_WORDS.items():
        text = text.replace(old, new)
    return text


def fix_missing_space(text: str) -> str:
    """Restore the space lost after sentence punctuation ('show.She' -> 'show. She')."""
    return MISSING_SPACE_RE.sub(r"\1 ", text)


def strip_orphan_quote(text: str) -> str:
    """Drop a quote mark at the start of a row that is followed by a space (it closes the previous row)."""
    return ORPHAN_QUOTE_RE.sub("", text)


def strip_leading_dash(text: str) -> str:
    """Remove subtitle turn dashes at the start ('- It is?' -> 'It is?').

    A dash glued to a digit ('-900') could be a minus sign, so it stays.
    """
    match = LEADING_DASHES_RE.match(text)
    if not match or match.end() == len(text):
        return text
    glued_to_digit = text[match.end()].isdigit() and not any(c.isspace() for c in match.group(0))
    return text if glued_to_digit else text[match.end() :]


def strip_trailing_dash(text: str) -> str:
    """Remove a lone trailing subtitle dash; a double dash marking interrupted speech stays."""
    return TRAILING_DASH_RE.sub("", text)


ROW_RULES: list[Rule] = [
    ("whitespace", normalize_whitespace),
    ("typography", normalize_typography),
    ("ocr_l_as_I", fix_l_as_i),
    ("ocr_caps_l", fix_caps_l),
    ("merged_words", fix_merged_words),
    ("missing_space", fix_missing_space),
    ("orphan_quote", strip_orphan_quote),
]
TURN_RULES: list[Rule] = [
    ("leading_dash", strip_leading_dash),
    ("trailing_dash", strip_trailing_dash),
    ("whitespace", normalize_whitespace),
]


def apply_rules(text: str, rules: list[Rule]) -> tuple[str, list[str]]:
    """Apply (name, function) rules in order; return the text and the names of rules that changed it."""
    applied = []
    for name, rule in rules:
        new_text = rule(text)
        if new_text != text:
            applied.append(name)
            text = new_text
    return text, applied


def split_turns(text: str) -> list[str]:
    """Split a row at subtitle turn markers: sentence-final punctuation followed by ' - '."""
    return TURN_SPLIT_RE.split(text)


def has_content(text: str) -> bool:
    """True when the text contains at least one letter or digit."""
    return bool(CONTENT_RE.search(text))


def clean_row_text(raw: str) -> list[tuple[str, list[str]]]:
    """Clean one CSV text field into turns: [(clean_text, rules_applied), ...].

    Pieces without a letter or digit are discarded. If more than one turn remains, each
    turn's rule list ends with 'split_turns'. An empty list means the row had no content.
    """
    text, row_rules = apply_rules(raw, ROW_RULES)
    turns = []
    for piece in split_turns(text):
        clean, turn_rules = apply_rules(piece, TURN_RULES)
        if has_content(clean):
            turns.append((clean, list(dict.fromkeys(row_rules + turn_rules))))
    if len(turns) > 1:
        turns = [(clean, [*rules, "split_turns"]) for clean, rules in turns]
    return turns
