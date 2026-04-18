from pathlib import Path

import pytest
from spacy.tokens import DocBin

from src.extraction.nlp_loader import get_nlp
from src.training.bootstrap_annotate import (
    LABELS,
    YOE_PATTERN,
    annotate_text,
    build_docbin,
    build_matchers,
)


def test_yoe_pattern_matches_common_forms():
    samples = ["5 years", "3+ years", "5-7 years", "10 yrs", "2 yr"]
    for s in samples:
        assert YOE_PATTERN.search(s), f"failed on {s!r}"


def test_annotate_text_labels_skills_degrees_titles_yoe():
    nlp = get_nlp()
    matchers = build_matchers(nlp)
    text = (
        "Senior Data Scientist with 5+ years of experience in Python, AWS, and "
        "TensorFlow. Holds a Master of Science from Stanford."
    )
    doc = annotate_text(text, nlp, matchers)
    by_label: dict[str, list[str]] = {lbl: [] for lbl in LABELS}
    for ent in doc.ents:
        by_label[ent.label_].append(ent.text)

    assert "Python" in by_label["SKILL"]
    assert "AWS" in by_label["SKILL"]
    assert "TensorFlow" in by_label["SKILL"]
    assert any("years" in y.lower() or "yr" in y.lower() for y in by_label["YOE"])
    assert any("Senior Data Scientist" in t for t in by_label["JOB_TITLE"])
    assert any("Master" in d for d in by_label["DEGREE"])


def test_annotate_text_resolves_overlaps_greedily():
    """'Senior Data Scientist' (JOB_TITLE) should win over 'Data Scientist' (JOB_TITLE)."""
    nlp = get_nlp()
    matchers = build_matchers(nlp)
    doc = annotate_text("Currently a Senior Data Scientist at Foo Corp.", nlp, matchers)
    titles = [ent.text for ent in doc.ents if ent.label_ == "JOB_TITLE"]
    assert "Senior Data Scientist" in titles
    assert "Data Scientist" not in titles


def test_annotate_text_empty_returns_empty_doc():
    nlp = get_nlp()
    matchers = build_matchers(nlp)
    doc = annotate_text("", nlp, matchers)
    assert list(doc.ents) == []


def test_build_docbin_writes_files_and_stats(tmp_path: Path):
    # Tiny inline corpus written as temp .txt files is not supported by parser
    # (we only accept .pdf/.docx). Instead, construct Doc objects directly via
    # the annotate_text path and test the split + serialization by calling
    # build_docbin with a single real PDF from the dataset if present.
    pytest.importorskip("pdfplumber")

    root = Path("data/raw/snehaanbhawal_resumes/data/data")
    if not root.exists():
        pytest.skip("Kaggle dataset not downloaded")
    pdfs = sorted(root.rglob("*.pdf"))[:5]
    if len(pdfs) < 2:
        pytest.skip("need at least 2 sample PDFs")

    stats = build_docbin(pdfs, tmp_path, train_ratio=0.6, seed=1)
    assert (tmp_path / "train.spacy").exists()
    assert (tmp_path / "dev.spacy").exists()
    assert stats.total_docs >= 1
    assert stats.train_docs >= 1
    assert stats.dev_docs >= 1

    # Reload and check the docs have entities
    nlp = get_nlp()
    loaded = list(DocBin().from_disk(tmp_path / "train.spacy").get_docs(nlp.vocab))
    assert any(doc.ents for doc in loaded)
