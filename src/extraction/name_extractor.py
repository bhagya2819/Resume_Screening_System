"""Extract candidate name from the resume header via spaCy PERSON NER."""
from __future__ import annotations

import re

from src.extraction.nlp_loader import get_nlp

_HEADER_CHAR_LIMIT = 600  # NER only needs the top of the resume


def _sanitize(raw: str) -> str:
    """Trim spaCy's PERSON span to a plausible human name.

    spaCy occasionally stretches the entity across a newline into the email
    or phone line; keep only up to the first newline, and drop anything past
    the first '@' or digit.
    """
    name = raw.split("\n", 1)[0].strip()
    name = name.split("@", 1)[0].strip()
    name = re.split(r"\d", name, maxsplit=1)[0].strip()
    name = name.strip(",;:-")
    return name


def extract_name(header_text: str) -> str | None:
    """Return the first PERSON entity found in the header region."""
    if not header_text:
        return None

    nlp = get_nlp()
    doc = nlp(header_text[:_HEADER_CHAR_LIMIT])
    for ent in doc.ents:
        if ent.label_ != "PERSON":
            continue
        name = _sanitize(ent.text)
        if name and 1 <= len(name.split()) <= 4:
            return name
    return None
