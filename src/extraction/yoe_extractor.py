"""Extract years of experience from resume text via regex heuristics.

We report the maximum plausible YOE found. Resumes often mention several
numbers ("3 years at Acme", "8+ years overall") — the max captures the
candidate's overall seniority best.
"""
from __future__ import annotations

import re

_WORD_TO_NUM = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
}

# Order matters: more specific patterns first.
_PATTERNS: list[re.Pattern[str]] = [
    # "5-7 years", "5 to 7 years"
    re.compile(r"(\d{1,2})\s*(?:-|to|–)\s*\d{1,2}\s+(?:year|yr)s?", re.IGNORECASE),
    # "over 5 years", "more than 5 years", "above 5 years"
    re.compile(
        r"(?:over|above|more\s+than)\s+(\d{1,2})\s+(?:year|yr)s?",
        re.IGNORECASE,
    ),
    # "5+ years"
    re.compile(r"(\d{1,2})\s*\+\s*(?:year|yr)s?", re.IGNORECASE),
    # "5 years of experience"
    re.compile(
        r"(\d{1,2})\s+(?:year|yr)s?\s+(?:of\s+)?(?:experience|exp|work)",
        re.IGNORECASE,
    ),
    # "five years of experience" (word form)
    re.compile(
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|fifteen|twenty)\s+(?:year|yr)s?\s+(?:of\s+)?"
        r"(?:experience|exp)",
        re.IGNORECASE,
    ),
    # Generic "5 years" fallback (picks up job durations)
    re.compile(r"(\d{1,2})\s+(?:year|yr)s?\b", re.IGNORECASE),
]


def extract_yoe(text: str) -> int | None:
    """Return the maximum plausible YOE in *text*, or None if nothing matches."""
    if not text:
        return None

    values: list[int] = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(text):
            raw = match.group(1).lower()
            try:
                value = int(raw) if raw.isdigit() else _WORD_TO_NUM[raw]
            except (ValueError, KeyError):
                continue
            if 0 < value <= 50:
                values.append(value)

    return max(values) if values else None
