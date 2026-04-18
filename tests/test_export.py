from src.export import (
    export_ranked_to_csv,
    export_ranked_to_pdf,
    ranked_to_dataframe,
)
from src.extraction.education_extractor import EducationMatch
from src.extraction.entity_extractor import ResumeEntities
from src.matching.jd_parser import JobRequirements
from src.matching.ranker import rank_candidates


def _ranked_fixture():
    jd = JobRequirements(
        raw_text="Python data engineer",
        title="Data Engineer",
        required_skills=["Python", "Airflow"],
        preferred_skills=["Spark"],
    )
    c1 = ResumeEntities(
        source_filename="alice.pdf",
        cleaned_text="Python and Airflow and Spark at big tech",
        name="Alice Smith",
        email="alice@example.com",
        skills=["Python", "Airflow", "Spark", "Go"],
        degrees=[EducationMatch(name="M.S.", tier="masters")],
        yoe=7,
    )
    c2 = ResumeEntities(
        source_filename="bob.pdf",
        cleaned_text="Java developer with Spring Boot",
        name="Bob Jones",
        email="bob@example.com",
        skills=["Java", "Spring"],
        yoe=3,
    )
    return rank_candidates(jd, [c1, c2], min_score=0.0)


def test_ranked_to_dataframe_columns_and_sort():
    ranked = _ranked_fixture()
    df = ranked_to_dataframe(ranked)
    assert list(df.columns) == [
        "Rank",
        "Candidate",
        "Filename",
        "Email",
        "Phone",
        "Overall",
        "Skills %",
        "Semantic %",
        "Experience %",
        "Education %",
        "YOE",
        "Degree",
        "Matched Required",
        "Missing Required",
        "Matched Preferred",
    ]
    # Alice should rank first (more skills match)
    assert df.iloc[0]["Filename"] == "alice.pdf"
    assert df.iloc[0]["Rank"] == 1
    assert df.iloc[0]["Candidate"] == "Alice Smith"


def test_export_ranked_to_csv_is_parseable_utf8():
    ranked = _ranked_fixture()
    raw = export_ranked_to_csv(ranked)
    text = raw.decode("utf-8")
    assert "Alice Smith" in text
    assert "alice@example.com" in text
    # Header row present
    assert text.splitlines()[0].startswith("Rank,")


def test_export_ranked_to_pdf_produces_pdf_bytes():
    ranked = _ranked_fixture()
    pdf = export_ranked_to_pdf(ranked, jd_title="Data Engineer")
    assert isinstance(pdf, bytes)
    assert pdf.startswith(b"%PDF-"), "output should be a real PDF"
    assert b"%%EOF" in pdf


def test_export_ranked_to_pdf_handles_empty_list():
    pdf = export_ranked_to_pdf([], jd_title=None)
    assert pdf.startswith(b"%PDF-")
