"""Append a JSONL record for each ranking run so we can review what was scored.

Writes to logs/runs.jsonl. Gitignored — meant for local debugging, not PII
storage. Nothing sensitive beyond what the user already uploaded in-session.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.config import LOGS_DIR
from src.matching.jd_parser import JobRequirements
from src.matching.ranker import RankedCandidate

_LOG_PATH = LOGS_DIR / "runs.jsonl"


def _serialize_candidate(r: RankedCandidate) -> dict:
    return {
        "rank": r.rank,
        "filename": r.filename,
        "overall": round(r.score.overall, 4),
        "skills": round(r.score.skills, 4),
        "semantic": round(r.score.semantic, 4),
        "experience": round(r.score.experience, 4),
        "education": round(r.score.education, 4),
        "matched_required": r.explanation.matched_required,
        "missing_required": r.explanation.missing_required,
    }


def log_run(
    jd: JobRequirements,
    ranked: Iterable[RankedCandidate],
    *,
    n_candidates_submitted: int,
    threshold: float,
    weights: dict,
) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "jd_title": jd.title,
        "jd_required_skills": jd.required_skills,
        "jd_preferred_skills": jd.preferred_skills,
        "jd_min_yoe": jd.min_yoe,
        "jd_required_degree": jd.required_degree,
        "n_candidates_submitted": n_candidates_submitted,
        "threshold": threshold,
        "weights": weights,
        "ranked": [_serialize_candidate(r) for r in ranked],
    }
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return _LOG_PATH
