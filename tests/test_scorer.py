from src.config import ScoringWeights
from src.matching.scorer import (
    compute_score,
    education_match,
    skills_overlap,
    yoe_match,
)


def test_skills_overlap_all_required_met():
    assert skills_overlap(["Python", "AWS"], ["Python", "AWS"]) == 1.0


def test_skills_overlap_half_met():
    assert skills_overlap(["Python"], ["Python", "AWS"]) == 0.5


def test_skills_overlap_case_insensitive():
    assert skills_overlap(["python"], ["Python"]) == 1.0


def test_skills_overlap_combined_required_and_preferred():
    # 2/2 required (weighted 0.8), 1/2 preferred (weighted 0.2) = 0.8 + 0.1 = 0.9
    score = skills_overlap(
        candidate_skills=["Python", "AWS", "Docker"],
        required=["Python", "AWS"],
        preferred=["Docker", "Kubernetes"],
    )
    assert abs(score - 0.9) < 1e-9


def test_skills_overlap_empty_required_and_preferred():
    assert skills_overlap(["Python"], [], []) == 0.0


def test_yoe_match_meets_requirement():
    assert yoe_match(5, 3) == 1.0
    assert yoe_match(3, 3) == 1.0


def test_yoe_match_partial_credit_below_requirement():
    assert yoe_match(2, 4) == 0.5


def test_yoe_match_no_requirement_is_full_score():
    assert yoe_match(None, None) == 1.0
    assert yoe_match(None, 0) == 1.0


def test_yoe_match_missing_candidate_yoe():
    assert yoe_match(None, 3) == 0.0


def test_education_match_meets():
    assert education_match("masters", "bachelors") == 1.0


def test_education_match_exact():
    assert education_match("bachelors", "bachelors") == 1.0


def test_education_match_partial_when_lower_tier():
    # Has an associate but needs a bachelors.
    assert education_match("associate", "bachelors") == 0.5


def test_education_match_zero_when_no_degree():
    assert education_match(None, "bachelors") == 0.0


def test_education_match_no_requirement_is_full_score():
    assert education_match(None, None) == 1.0
    assert education_match("associate", None) == 1.0


def test_compute_score_matches_weights():
    weights = ScoringWeights(skills=0.5, semantic=0.2, experience=0.2, education=0.1)
    breakdown = compute_score(
        candidate_skills=["Python"],
        required_skills=["Python"],
        preferred_skills=[],
        candidate_yoe=5,
        min_yoe=3,
        candidate_degree_tier="masters",
        required_degree_tier="bachelors",
        semantic_similarity=0.4,
        weights=weights,
    )
    expected = 0.5 * 1.0 + 0.2 * 0.4 + 0.2 * 1.0 + 0.1 * 1.0
    assert abs(breakdown.overall - expected) < 1e-9
    assert breakdown.skills == 1.0
    assert breakdown.semantic == 0.4
    assert breakdown.experience == 1.0
    assert breakdown.education == 1.0


def test_compute_score_clamps_semantic_similarity():
    # Odd inputs shouldn't crash.
    breakdown = compute_score(
        candidate_skills=[],
        required_skills=["Python"],
        preferred_skills=[],
        candidate_yoe=None,
        min_yoe=None,
        candidate_degree_tier=None,
        required_degree_tier=None,
        semantic_similarity=1.5,  # out of [0,1]
    )
    assert breakdown.semantic == 1.0
