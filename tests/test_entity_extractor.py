from pathlib import Path

import pytest

from src.config import RAW_DATA_DIR
from src.extraction.entity_extractor import ResumeEntities, extract_entities
from src.parsing.resume_parser import parse_resume


KAGGLE_PDF_ROOT = RAW_DATA_DIR / "snehaanbhawal_resumes" / "data" / "data"


def _sample_pdfs(limit: int = 5) -> list[Path]:
    if not KAGGLE_PDF_ROOT.exists():
        return []
    # Mix of categories gives more confidence than 5 from the same folder.
    pdfs: list[Path] = []
    for category in sorted(KAGGLE_PDF_ROOT.iterdir())[:limit]:
        if category.is_dir():
            first = next(iter(sorted(category.glob("*.pdf"))), None)
            if first is not None:
                pdfs.append(first)
    return pdfs


def test_extract_entities_returns_dataclass(tmp_path: Path):
    # Build a minimal parsed resume from raw text.
    from src.parsing.resume_parser import ParsedResume
    from src.parsing.text_cleaner import clean_text
    from src.parsing.section_detector import detect_sections

    raw = """Jane Doe
jane.doe@example.com
(555) 123-4567

Summary
Senior Data Scientist with 6+ years of experience in Python and TensorFlow.

Experience
Senior Data Scientist at Acme — led ML projects.

Education
M.S. Computer Science from Stanford.

Skills
Python, TensorFlow, PyTorch, AWS, Docker"""
    cleaned = clean_text(raw)
    resume = ParsedResume(
        source_path=tmp_path / "jane.pdf",
        raw_text=raw,
        cleaned_text=cleaned,
        sections=detect_sections(cleaned),
    )
    ent = extract_entities(resume)

    assert isinstance(ent, ResumeEntities)
    assert ent.name == "Jane Doe"
    assert ent.email == "jane.doe@example.com"
    assert ent.phone is not None
    assert ent.yoe == 6
    assert "Python" in ent.skills
    assert "TensorFlow" in ent.skills
    assert any(d.tier == "masters" for d in ent.degrees)
    assert ent.highest_degree_tier == "masters"
    assert "Senior Data Scientist" in ent.titles


@pytest.mark.skipif(not _sample_pdfs(), reason="Kaggle dataset not downloaded")
def test_extract_entities_on_real_resumes():
    """Smoke-test over 5 real Kaggle resumes, one per category.

    The Kaggle dataset is anonymized (contact info is often stripped), so we
    only assert that skills — the load-bearing signal for ranking — are
    detected in at least one resume.
    """
    any_skill_found = False
    for pdf in _sample_pdfs():
        resume = parse_resume(pdf)
        ent = extract_entities(resume)
        assert ent.source_filename == pdf.name
        if ent.skills:
            any_skill_found = True
    assert any_skill_found, "expected skills in at least one real resume"
