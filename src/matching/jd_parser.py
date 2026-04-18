"""Parse job descriptions from text, uploaded file, or structured form.

Normalizes every input mode to a JobRequirements dataclass that the scorer
and ranker consume.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from src.config import SUPPORTED_JD_EXTENSIONS
from src.extraction.education_extractor import (
    EDUCATION_TIERS,
    extract_education,
    highest_tier,
)
from src.extraction.skill_extractor import extract_skills
from src.extraction.yoe_extractor import extract_yoe
from src.parsing.docx_parser import parse_docx
from src.parsing.pdf_parser import parse_pdf
from src.parsing.text_cleaner import clean_text


class UnsupportedJDFormat(Exception):
    pass


@dataclass
class JobRequirements:
    raw_text: str = ""
    title: str | None = None
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    min_yoe: int | None = None
    required_degree: str | None = None  # one of EDUCATION_TIERS

    @property
    def all_skills(self) -> list[str]:
        seen: dict[str, str] = {}
        for s in (*self.required_skills, *self.preferred_skills):
            seen.setdefault(s.lower(), s)
        return list(seen.values())


# Matches a "Preferred Qualifications" / "Nice to have" / "Bonus" section header.
_PREFERRED_HEADER_RE = re.compile(
    r"^\s*(?:preferred(?:\s+qualifications)?|nice[\s-]*to[\s-]*have(?:s)?|"
    r"bonus(?:\s+points?)?|plus(?:es)?|desired|good\s+to\s+have)\b.*$",
    re.IGNORECASE | re.MULTILINE,
)


def _split_required_preferred(text: str) -> tuple[str, str]:
    """Return (required_section, preferred_section). Preferred is empty if no header."""
    match = _PREFERRED_HEADER_RE.search(text)
    if not match:
        return text, ""
    return text[: match.start()], text[match.end():]


def _normalize_degree_tier(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip().lower()
    if value in EDUCATION_TIERS:
        return value
    tier = highest_tier(extract_education(value))
    return tier


def parse_jd_from_text(
    text: str,
    *,
    title: str | None = None,
) -> JobRequirements:
    """Parse a free-text JD; auto-detects skills, YOE, and degree requirement.

    YOE and required degree are read from the required section only — pulling
    them from the preferred section would overstate the requirements (e.g.
    "PhD preferred" should not make a PhD the minimum).
    """
    cleaned = clean_text(text)
    required_section, preferred_section = _split_required_preferred(cleaned)

    required_skills = extract_skills(required_section)
    preferred_skills = [
        s for s in extract_skills(preferred_section) if s not in required_skills
    ]

    return JobRequirements(
        raw_text=cleaned,
        title=title,
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        min_yoe=extract_yoe(required_section),
        required_degree=highest_tier(extract_education(required_section)),
    )


def parse_jd_from_file(path: str | Path) -> JobRequirements:
    """Parse a JD from PDF/DOCX/TXT on disk."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext not in SUPPORTED_JD_EXTENSIONS:
        raise UnsupportedJDFormat(
            f"Unsupported JD format {ext!r}; expected one of "
            f"{sorted(SUPPORTED_JD_EXTENSIONS)}"
        )
    if ext == ".pdf":
        text = parse_pdf(path)
    elif ext == ".docx":
        text = parse_docx(path)
    else:  # .txt
        text = path.read_text(encoding="utf-8", errors="ignore")
    return parse_jd_from_text(text, title=path.stem)


def parse_jd_from_form(
    *,
    title: str | None = None,
    required_skills: list[str] | None = None,
    preferred_skills: list[str] | None = None,
    min_yoe: int | None = None,
    required_degree: str | None = None,
    description: str = "",
) -> JobRequirements:
    """Build JobRequirements from explicit form fields.

    The `description` field is still used for TF-IDF semantic matching and,
    if the user didn't populate the skill lists, as a fallback to extract
    them automatically.
    """
    required = list(required_skills or [])
    preferred = list(preferred_skills or [])
    cleaned = clean_text(description)

    if cleaned and not (required or preferred):
        required = extract_skills(cleaned)

    return JobRequirements(
        raw_text=cleaned,
        title=title,
        required_skills=required,
        preferred_skills=preferred,
        min_yoe=min_yoe,
        required_degree=_normalize_degree_tier(required_degree),
    )
