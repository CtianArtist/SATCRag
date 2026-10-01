"""Speaker-label normalization: small generic cleanup rules plus an explicit alias file.

Order for each label: tidy (whitespace, wrapping parentheses, 'Woman #1' numbering) ->
split multi-speaker labels ('Carrie, Miranda') -> per name: casing variant ->
episode-scoped alias -> global alias. The original label is never modified; callers keep it.
"""

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from satc_rag.errors import MetadataError

WRAPPED_RE = re.compile(r"^\((.*)\)$")  # "(All)" -> "All"
NUMBERED_RE = re.compile(r"^(.*?[A-Za-z])\s*#?\s*(\d+)$")  # "Woman #1", "Woman1" -> "Woman 1"
MULTI_SPLIT_RE = re.compile(r"\s*(?:,|&|\band\b)\s*")  # "Carrie and Miranda"
EPISODE_CODE_RE = re.compile(r"^S(\d+)E(\d+)$")


@dataclass
class AliasMap:
    """Explicit aliases from data/meta/speaker_aliases.json."""

    global_aliases: dict[str, str]
    scoped_aliases: dict[tuple[int, int, str], str]  # (season, episode, label) -> name


def parse_episode_code(code: str) -> tuple[int, int]:
    """Turn 'S5E8' into (5, 8)."""
    match = EPISODE_CODE_RE.match(code)
    if not match:
        raise MetadataError(f"bad episode code {code!r} in the speaker alias file", expected="like 'S5E8'")
    return int(match.group(1)), int(match.group(2))


def alias_entry(entry: object, where: str) -> tuple[str, str]:
    """The (from, to) pair of one alias entry, checked."""
    if not (isinstance(entry, dict) and isinstance(entry.get("from"), str) and isinstance(entry.get("to"), str)):
        raise MetadataError(f"malformed alias entry in {where}", expected='{"from": "...", "to": "..."}', actual=entry)
    return entry["from"], entry["to"]


def load_aliases(path: Path) -> AliasMap:
    """Read the alias file into lookup tables, rejecting a missing or malformed file."""
    if not path.is_file():
        raise MetadataError(f"speaker alias file not found: {path}", recovery="restore data/meta/ from the repository")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MetadataError(f"speaker alias file is not valid JSON: {path} ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise MetadataError(f"speaker alias file must hold a JSON object: {path}")
    global_aliases = dict(alias_entry(entry, "global") for entry in data.get("global", []))
    scoped = {}
    for entry in data.get("episode_scoped", []):
        source, target = alias_entry(entry, "episode_scoped")
        episodes = entry.get("episodes")
        if not isinstance(episodes, list) or not episodes:
            raise MetadataError(f"episode-scoped alias {source!r} must list its episodes", actual=episodes)
        for code in episodes:
            season, episode = parse_episode_code(code)
            scoped[(season, episode, source)] = target
    return AliasMap(global_aliases, scoped)


def tidy_label(label: str) -> tuple[str, list[str]]:
    """Apply the generic label cleanups; return (label, names of rules that changed it)."""
    rules = []
    tidy = re.sub(r"\s+", " ", label).strip()
    if tidy != label:
        rules.append("whitespace")
    unwrapped = WRAPPED_RE.sub(r"\1", tidy).strip()
    if unwrapped != tidy:
        rules.append("unwrap_parens")
    numbered = NUMBERED_RE.sub(r"\1 \2", unwrapped)
    if numbered != unwrapped:
        rules.append("numbered_extra")
    return numbered, rules


def split_label(label: str) -> list[str]:
    """Split a multi-speaker label ('Carrie, Samantha and Miranda') into names."""
    return [part for part in MULTI_SPLIT_RE.split(label) if part]


def casing_errors(label: str) -> int:
    """Count lowercase word starts and capitals inside words ('CArrie' -> 1, 'Carrie' -> 0)."""
    words = re.findall(r"[A-Za-z]+", label)
    return sum(w[0].islower() + sum(c.isupper() for c in w[1:]) for w in words)


def label_counts(raw_labels: Iterable[str]) -> Counter[str]:
    """Count every tidy single name in the corpus (multi-speaker labels count each name)."""
    counts: Counter[str] = Counter()
    for raw in raw_labels:
        tidy, _ = tidy_label(raw)
        for part in split_label(tidy):
            counts[tidy_label(part)[0]] += 1
    return counts


def build_case_map(counts: Counter[str]) -> dict[str, str]:
    """Map casing variants to one spelling: fewest casing errors, then most frequent, then alphabetical."""
    groups: defaultdict[str, dict[str, int]] = defaultdict(dict)
    for label, n in counts.items():
        groups[label.casefold()][label] = n
    case_map = {}
    for variants in groups.values():
        best = min(variants, key=lambda v: (casing_errors(v), -variants[v], v))
        case_map.update({v: best for v in variants if v != best})
    return case_map


def canonical_name(
    name: str, season: int, episode: int, aliases: AliasMap, case_map: dict[str, str]
) -> tuple[str, list[str]]:
    """Resolve one name: casing variant, then episode-scoped alias, then global alias."""
    name, rules = tidy_label(name)
    if name in case_map:
        name, rules = case_map[name], [*rules, "case_variant"]
    scoped = aliases.scoped_aliases.get((season, episode, name))
    if scoped:
        return scoped, [*rules, "episode_alias"]
    if name in aliases.global_aliases:
        return aliases.global_aliases[name], [*rules, "alias"]
    return name, rules


def normalize_speaker(
    label: str, season: int, episode: int, aliases: AliasMap, case_map: dict[str, str]
) -> tuple[str | None, list[str], list[str]]:
    """Return (display speaker, list of names, rules applied); a blank label gives (None, [], rules)."""
    tidy, rules = tidy_label(label)
    parts = split_label(tidy)
    if not parts:
        return None, [], rules
    if len(parts) > 1:
        rules.append("multi_speaker")
    names: list[str] = []
    for part in parts:
        name, part_rules = canonical_name(part, season, episode, aliases, case_map)
        rules += [r for r in part_rules if r not in rules]
        if name not in names:
            names.append(name)
    return " & ".join(names), names, rules
