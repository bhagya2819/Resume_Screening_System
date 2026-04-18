"""Split cleaned resume text into canonical sections via heuristic headers.

Resumes vary wildly, so detection is tolerant:
- A line is treated as a section header if it is short AND matches one of the
  canonical patterns (case-insensitive) as either the whole line or the start.
- Everything between two header lines belongs to the first header's section.
- Lines before the first detected header go into the "header" section (usually
  contact info + name).
"""
from __future__ import annotations

import re
from typing import Final

_SECTION_PATTERNS: Final[dict[str, list[str]]] = {
    "summary": [
        r"summary", r"professional summary", r"profile", r"objective",
        r"career objective", r"about me",
    ],
    "experience": [
        r"experience", r"work experience", r"professional experience",
        r"employment", r"employment history", r"work history", r"career history",
    ],
    "education": [
        r"education", r"academic background", r"academic qualifications",
        r"qualifications", r"educational background",
    ],
    "skills": [
        r"skills", r"technical skills", r"core skills", r"key skills",
        r"core competencies", r"competencies", r"technologies",
        r"technical expertise", r"areas of expertise",
    ],
    "projects": [
        r"projects", r"personal projects", r"academic projects",
        r"notable projects", r"selected projects",
    ],
    "certifications": [
        r"certifications", r"certificates", r"licenses and certifications",
        r"professional certifications",
    ],
    "awards": [
        r"awards", r"honors", r"awards and honors", r"achievements",
        r"accomplishments",
    ],
    "publications": [r"publications", r"papers", r"research"],
    "languages": [r"languages"],
    "interests": [r"interests", r"hobbies", r"activities"],
}

_CANONICAL: Final[list[tuple[str, re.Pattern[str]]]] = [
    (
        name,
        re.compile(
            rf"^\s*(?:{'|'.join(patterns)})\s*[:\-]?\s*$",
            re.IGNORECASE,
        ),
    )
    for name, patterns in _SECTION_PATTERNS.items()
]

_MAX_HEADER_LEN = 60


def _match_header(line: str) -> str | None:
    if not line or len(line) > _MAX_HEADER_LEN:
        return None
    for name, pattern in _CANONICAL:
        if pattern.match(line):
            return name
    return None


def detect_sections(text: str) -> dict[str, str]:
    """Split cleaned resume text into canonical sections.

    Returns a dict mapping section name ("header", "summary", "experience", …)
    to its raw text. Absent sections are omitted.
    """
    sections: dict[str, list[str]] = {"header": []}
    current = "header"

    for line in text.splitlines():
        matched = _match_header(line)
        if matched is not None:
            current = matched
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)

    return {
        name: "\n".join(lines).strip()
        for name, lines in sections.items()
        if any(ln.strip() for ln in lines)
    }
