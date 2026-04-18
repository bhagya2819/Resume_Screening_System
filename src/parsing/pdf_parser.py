"""Extract text from PDF resumes using pdfplumber."""
from __future__ import annotations

from pathlib import Path

import pdfplumber


class PDFParseError(Exception):
    pass


def parse_pdf(path: str | Path) -> str:
    """Extract raw text from a PDF file.

    Pages are joined with a blank line so the section detector can still see
    page breaks as paragraph boundaries.
    """
    path = Path(path)
    try:
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as exc:
        raise PDFParseError(f"Failed to parse PDF {path}: {exc}") from exc
