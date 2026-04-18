from pathlib import Path

import pytest

from src.config import RAW_DATA_DIR
from src.parsing.resume_parser import (
    ParsedResume,
    UnsupportedFormatError,
    parse_resume,
)


KAGGLE_PDF_ROOT = RAW_DATA_DIR / "snehaanbhawal_resumes" / "data" / "data"


def _sample_pdfs(limit: int = 3) -> list[Path]:
    if not KAGGLE_PDF_ROOT.exists():
        return []
    return sorted(KAGGLE_PDF_ROOT.rglob("*.pdf"))[:limit]


def test_unsupported_extension_raises(tmp_path: Path):
    f = tmp_path / "resume.rtf"
    f.write_text("not supported")
    with pytest.raises(UnsupportedFormatError):
        parse_resume(f)


@pytest.mark.skipif(not _sample_pdfs(), reason="Kaggle dataset not downloaded")
@pytest.mark.parametrize("pdf_path", _sample_pdfs())
def test_parse_kaggle_pdf(pdf_path: Path):
    parsed = parse_resume(pdf_path)
    assert isinstance(parsed, ParsedResume)
    assert parsed.filename == pdf_path.name
    assert len(parsed.raw_text) > 100, "expected non-trivial text from a real resume"
    assert len(parsed.cleaned_text) > 0
    assert isinstance(parsed.sections, dict)
    # Every real resume should have a header region (contact info / name).
    assert "header" in parsed.sections


@pytest.mark.skipif(not _sample_pdfs(), reason="Kaggle dataset not downloaded")
def test_parse_kaggle_pdf_detects_common_sections():
    """At least one of experience/education/skills should be detected across samples."""
    samples = _sample_pdfs(limit=5)
    hit = False
    for pdf in samples:
        parsed = parse_resume(pdf)
        if {"experience", "education", "skills"} & parsed.sections.keys():
            hit = True
            break
    assert hit, "Expected at least one resume to surface a standard section"
