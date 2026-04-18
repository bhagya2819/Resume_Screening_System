"""Extract email and phone number from resume text via regex."""
from __future__ import annotations

import re

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)

# Matches common US and international formats: "(555) 123-4567", "+1-555-123-4567",
# "555.123.4567", "+91 98765 43210". Requires at least 10 digits total.
_PHONE_RE = re.compile(
    r"""
    (?:(?:\+|00)\d{1,3}[\s.\-]?)?        # country code
    (?:\(?\d{2,4}\)?[\s.\-]?)?           # area code / group
    \d{3,4}[\s.\-]?\d{3,4}               # local number
    """,
    re.VERBOSE,
)


def extract_email(text: str) -> str | None:
    if not text:
        return None
    match = _EMAIL_RE.search(text)
    return match.group(0) if match else None


def extract_phone(text: str) -> str | None:
    """Return the first plausible phone number (>= 10 digits) found."""
    if not text:
        return None
    for match in _PHONE_RE.finditer(text):
        candidate = match.group(0).strip()
        digits = re.sub(r"\D", "", candidate)
        if 10 <= len(digits) <= 15:
            return candidate
    return None
