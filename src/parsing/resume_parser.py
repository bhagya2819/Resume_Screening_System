"""Unified resume parser: dispatches on file extension."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import SUPPORTED_RESUME_EXTENSIONS
from src.parsing.docx_parser import parse_docx
from src.parsing.pdf_parser import parse_pdf
from src.parsing.section_detector import detect_sections
from src.parsing.text_cleaner import clean_text


class UnsupportedFormatError(Exception):
    pass


@dataclass
class ParsedResume:
    source_path: Path
    raw_text: str
    cleaned_text: str
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def filename(self) -> str:
        return self.source_path.name

    def section(self, name: str) -> str:
        return self.sections.get(name, "")


def parse_resume(path: str | Path) -> ParsedResume:
    """Parse a single resume end-to-end: extract, clean, detect sections."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_RESUME_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported resume format {ext!r}; expected one of "
            f"{sorted(SUPPORTED_RESUME_EXTENSIONS)}"
        )

    raw_text = parse_pdf(path) if ext == ".pdf" else parse_docx(path)
    cleaned = clean_text(raw_text)
    sections = detect_sections(cleaned)

    return ParsedResume(
        source_path=path,
        raw_text=raw_text,
        cleaned_text=cleaned,
        sections=sections,
    )
