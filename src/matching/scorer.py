"""Composite match scoring: skills + semantic + experience + education."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import DEFAULTS, ScoringWeights
from src.extraction.education_extractor import EDUCATION_TIERS

_TIER_RANK = {tier: i for i, tier in enumerate(reversed(EDUCATION_TIERS))}
# Higher index = more advanced. doctorate=4, masters=3, bachelors=2, associate=1, certification=0.


@dataclass
class ScoreBreakdown:
    overall: float
    skills: float
    semantic: float
    experience: float
    education: float


def skills_overlap(
    candidate_skills: list[str],
    required: list[str],
    preferred: list[str] | None = None,
) -> float:
    """Return a [0, 1] score combining required (80%) and preferred (20%) coverage."""
    preferred = preferred or []
    cand_lower = {s.lower() for s in candidate_skills}

    if not required and not preferred:
        return 0.0

    req_score = 0.0
    if required:
        req_hits = sum(1 for s in required if s.lower() in cand_lower)
        req_score = req_hits / len(required)

    pref_score = 0.0
    if preferred:
        pref_hits = sum(1 for s in preferred if s.lower() in cand_lower)
        pref_score = pref_hits / len(preferred)

    if not required:
        return pref_score
    if not preferred:
        return req_score
    return 0.8 * req_score + 0.2 * pref_score


def yoe_match(candidate_yoe: int | None, min_yoe: int | None) -> float:
    """Fraction [0, 1] measuring how well the candidate meets the YOE requirement."""
    if not min_yoe:  # None or 0
        return 1.0
    if candidate_yoe is None:
        return 0.0
    if candidate_yoe >= min_yoe:
        return 1.0
    return candidate_yoe / min_yoe


def education_match(
    candidate_tier: str | None,
    required_tier: str | None,
) -> float:
    """Does the candidate meet the minimum degree tier?"""
    if required_tier is None:
        return 1.0

    req_rank = _TIER_RANK.get(required_tier, -1)
    if req_rank < 0:
        return 1.0  # Unknown requirement — don't penalize

    cand_rank = _TIER_RANK.get(candidate_tier or "", -1)
    if cand_rank < 0:
        return 0.0
    if cand_rank >= req_rank:
        return 1.0
    return 0.5  # Has some education but not the required tier


def compute_score(
    *,
    candidate_skills: list[str],
    required_skills: list[str],
    preferred_skills: list[str],
    candidate_yoe: int | None,
    min_yoe: int | None,
    candidate_degree_tier: str | None,
    required_degree_tier: str | None,
    semantic_similarity: float,
    weights: ScoringWeights | None = None,
) -> ScoreBreakdown:
    weights = weights or DEFAULTS.weights

    skills = skills_overlap(candidate_skills, required_skills, preferred_skills)
    semantic = max(0.0, min(1.0, semantic_similarity))
    experience = yoe_match(candidate_yoe, min_yoe)
    education = education_match(candidate_degree_tier, required_degree_tier)

    overall = (
        weights.skills * skills
        + weights.semantic * semantic
        + weights.experience * experience
        + weights.education * education
    )
    return ScoreBreakdown(
        overall=overall,
        skills=skills,
        semantic=semantic,
        experience=experience,
        education=education,
    )
