from pathlib import Path

import pytest

from src.config import ROOT_DIR
from src.matching.jd_parser import (
    JobRequirements,
    UnsupportedJDFormat,
    parse_jd_from_file,
    parse_jd_from_form,
    parse_jd_from_text,
)


def test_parse_jd_from_text_extracts_skills_and_yoe():
    jd = """Senior Python Developer

Required:
- 5+ years of experience
- Python, Django, PostgreSQL
- Bachelor's Degree in Computer Science

Preferred:
- Kubernetes and AWS
- Kafka
"""
    req = parse_jd_from_text(jd)
    assert "Python" in req.required_skills
    assert "Django" in req.required_skills
    assert "PostgreSQL" in req.required_skills
    assert "Kubernetes" in req.preferred_skills
    assert "AWS" in req.preferred_skills
    assert req.min_yoe == 5
    assert req.required_degree == "bachelors"


def test_parse_jd_from_text_no_preferred_section():
    jd = "Looking for Python, Java and Docker expertise. 3 years of experience."
    req = parse_jd_from_text(jd)
    assert "Python" in req.required_skills
    assert req.preferred_skills == []
    assert req.min_yoe == 3


def test_parse_jd_from_text_no_yoe_or_degree():
    req = parse_jd_from_text("Looking for someone who knows React and GraphQL.")
    assert req.min_yoe is None
    assert req.required_degree is None
    assert set(req.required_skills) >= {"React", "GraphQL"}


def test_parse_jd_from_form_uses_explicit_fields():
    req = parse_jd_from_form(
        title="Data Engineer",
        required_skills=["Python", "Airflow"],
        preferred_skills=["Spark"],
        min_yoe=4,
        required_degree="bachelors",
        description="We need a pipeline wizard.",
    )
    assert req.title == "Data Engineer"
    assert req.required_skills == ["Python", "Airflow"]
    assert req.preferred_skills == ["Spark"]
    assert req.min_yoe == 4
    assert req.required_degree == "bachelors"


def test_parse_jd_from_form_normalizes_degree_from_free_text():
    req = parse_jd_from_form(required_degree="Master of Science")
    assert req.required_degree == "masters"


def test_parse_jd_from_form_falls_back_to_description_skills():
    req = parse_jd_from_form(description="Hiring a Python engineer with TensorFlow experience.")
    assert "Python" in req.required_skills
    assert "TensorFlow" in req.required_skills


def test_parse_jd_from_file_reads_sample_jd():
    sample = ROOT_DIR / "data" / "samples" / "sample_jd.txt"
    req = parse_jd_from_file(sample)
    assert isinstance(req, JobRequirements)
    assert req.title == "sample_jd"
    assert "Python" in req.required_skills
    assert req.min_yoe == 4
    # Sample mentions both Bachelor's and Master's as acceptable; highest wins.
    assert req.required_degree in {"bachelors", "masters", "doctorate"}


def test_parse_jd_from_file_rejects_unsupported_extension(tmp_path: Path):
    bad = tmp_path / "jd.rtf"
    bad.write_text("whatever")
    with pytest.raises(UnsupportedJDFormat):
        parse_jd_from_file(bad)


def test_job_requirements_all_skills_dedupes():
    req = JobRequirements(
        required_skills=["Python", "AWS"],
        preferred_skills=["python", "Kubernetes"],
    )
    # Case-insensitive dedup
    assert len(req.all_skills) == 3
