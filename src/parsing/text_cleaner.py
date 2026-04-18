"""Clean raw resume text: unicode normalization, bullets, whitespace."""
from __future__ import annotations

import re
import unicodedata

BULLET_CHARS = "•●○◦▪▫∙◆◇■□►▶▷·‣⁃"
_BULLET_LINE_RE = re.compile(rf"^\s*[{re.escape(BULLET_CHARS)}\-\*]\s+")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(text: str) -> str:
    """Normalize unicode, strip bullets, collapse whitespace.

    Preserves line breaks since downstream section detection relies on them.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = _NON_PRINTABLE_RE.sub("", text)

    cleaned_lines: list[str] = []
    for line in text.splitlines():
        line = _BULLET_LINE_RE.sub("", line)
        line = _MULTI_SPACE_RE.sub(" ", line).strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()
