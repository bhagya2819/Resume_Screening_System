import json
from unittest.mock import patch

from src.extraction.entity_extractor import ResumeEntities
from src.matching.jd_parser import JobRequirements
from src.matching.ranker import rank_candidates
from src.utils import run_logger


def test_log_run_appends_jsonl_record(tmp_path, monkeypatch):
    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_logger, "_LOG_PATH", log_path)
    monkeypatch.setattr(run_logger, "LOGS_DIR", tmp_path)

    jd = JobRequirements(
        raw_text="python needed",
        title="Python Dev",
        required_skills=["Python"],
    )
    cand = ResumeEntities(
        source_filename="a.pdf",
        cleaned_text="Python engineer",
        skills=["Python"],
    )
    ranked = rank_candidates(jd, [cand], min_score=0.0)

    run_logger.log_run(
        jd,
        ranked,
        n_candidates_submitted=1,
        threshold=0.0,
        weights={"skills": 1, "semantic": 0, "experience": 0, "education": 0},
    )

    lines = log_path.read_text().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["jd_title"] == "Python Dev"
    assert record["jd_required_skills"] == ["Python"]
    assert record["n_candidates_submitted"] == 1
    assert record["ranked"][0]["filename"] == "a.pdf"


def test_log_run_appends_multiple_times(tmp_path, monkeypatch):
    log_path = tmp_path / "runs.jsonl"
    monkeypatch.setattr(run_logger, "_LOG_PATH", log_path)
    monkeypatch.setattr(run_logger, "LOGS_DIR", tmp_path)

    jd = JobRequirements(raw_text="x", required_skills=[])
    for _ in range(3):
        run_logger.log_run(
            jd,
            [],
            n_candidates_submitted=0,
            threshold=0.4,
            weights={},
        )
    assert len(log_path.read_text().splitlines()) == 3
