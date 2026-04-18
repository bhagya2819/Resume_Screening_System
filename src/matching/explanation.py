"""Per-candidate match explanation: matched vs missing skills, YOE/edu deltas."""
from __future__ import annotations

from dataclasses import dataclass, field

from src.matching.scorer import education_match


@dataclass
class MatchExplanation:
    matched_required: list[str] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    matched_preferred: list[str] = field(default_factory=list)
    extra_skills: list[str] = field(default_factory=list)

    yoe_candidate: int | None = None
    yoe_required: int | None = None
    yoe_delta: int | None = None

    degree_candidate: str | None = None
    degree_required: str | None = None
    degree_meets_requirement: bool = True


_EXTRA_SKILL_CAP = 15


def explain(
    *,
    candidate_skills: list[str],
    required_skills: list[str],
    preferred_skills: list[str],
    candidate_yoe: int | None,
    required_yoe: int | None,
    candidate_degree: str | None,
    required_degree: str | None,
) -> MatchExplanation:
    cand_map = {s.lower(): s for s in candidate_skills}
    req_map = {s.lower(): s for s in required_skills}
    pref_map = {s.lower(): s for s in preferred_skills}

    matched_required = [req_map[k] for k in req_map if k in cand_map]
    missing_required = [req_map[k] for k in req_map if k not in cand_map]
    matched_preferred = [pref_map[k] for k in pref_map if k in cand_map]

    asked = set(req_map) | set(pref_map)
    extras = [cand_map[k] for k in cand_map if k not in asked][:_EXTRA_SKILL_CAP]

    yoe_delta: int | None = None
    if candidate_yoe is not None and required_yoe is not None:
        yoe_delta = candidate_yoe - required_yoe

    return MatchExplanation(
        matched_required=sorted(matched_required, key=str.lower),
        missing_required=sorted(missing_required, key=str.lower),
        matched_preferred=sorted(matched_preferred, key=str.lower),
        extra_skills=sorted(extras, key=str.lower),
        yoe_candidate=candidate_yoe,
        yoe_required=required_yoe,
        yoe_delta=yoe_delta,
        degree_candidate=candidate_degree,
        degree_required=required_degree,
        degree_meets_requirement=education_match(candidate_degree, required_degree) >= 1.0,
    )
