"""End-to-end ranking: TF-IDF → composite score → threshold filter → sort."""
from __future__ import annotations

from dataclasses import dataclass

from src.config import DEFAULTS, ScoringWeights
from src.extraction.entity_extractor import ResumeEntities
from src.matching.explanation import MatchExplanation, explain
from src.matching.jd_parser import JobRequirements
from src.matching.scorer import ScoreBreakdown, compute_score
from src.matching.tfidf_matcher import compute_tfidf_cosine


@dataclass
class RankedCandidate:
    rank: int
    filename: str
    entities: ResumeEntities
    score: ScoreBreakdown
    explanation: MatchExplanation


def _resume_text_for_tfidf(c: ResumeEntities) -> str:
    """Prefer the stored cleaned_text; fall back to joined sections."""
    if c.cleaned_text:
        return c.cleaned_text
    return "\n".join(c.sections.values())


def rank_candidates(
    jd: JobRequirements,
    candidates: list[ResumeEntities],
    *,
    weights: ScoringWeights | None = None,
    min_score: float | None = None,
    top_n: int | None = None,
) -> list[RankedCandidate]:
    """Score and rank candidates for a JD. Applies threshold filter and top-n cap."""
    if not candidates:
        return []

    weights = weights or DEFAULTS.weights
    min_score = DEFAULTS.min_score_threshold if min_score is None else min_score
    top_n = DEFAULTS.top_n if top_n is None else top_n

    jd_text_for_tfidf = jd.raw_text or " ".join(jd.all_skills)
    resume_texts = [_resume_text_for_tfidf(c) for c in candidates]
    similarities = compute_tfidf_cosine(jd_text_for_tfidf, resume_texts)

    ranked: list[RankedCandidate] = []
    for cand, sim in zip(candidates, similarities):
        score = compute_score(
            candidate_skills=cand.skills,
            required_skills=jd.required_skills,
            preferred_skills=jd.preferred_skills,
            candidate_yoe=cand.yoe,
            min_yoe=jd.min_yoe,
            candidate_degree_tier=cand.highest_degree_tier,
            required_degree_tier=jd.required_degree,
            semantic_similarity=sim,
            weights=weights,
        )
        expl = explain(
            candidate_skills=cand.skills,
            required_skills=jd.required_skills,
            preferred_skills=jd.preferred_skills,
            candidate_yoe=cand.yoe,
            required_yoe=jd.min_yoe,
            candidate_degree=cand.highest_degree_tier,
            required_degree=jd.required_degree,
        )
        ranked.append(
            RankedCandidate(
                rank=0,
                filename=cand.source_filename,
                entities=cand,
                score=score,
                explanation=expl,
            )
        )

    ranked = [r for r in ranked if r.score.overall >= min_score]
    ranked.sort(key=lambda r: r.score.overall, reverse=True)
    ranked = ranked[:top_n]
    for i, r in enumerate(ranked, start=1):
        r.rank = i
    return ranked
