from src.matching.explanation import explain


def test_explain_matched_and_missing():
    expl = explain(
        candidate_skills=["Python", "AWS", "Docker"],
        required_skills=["Python", "Kubernetes", "AWS"],
        preferred_skills=["Docker"],
        candidate_yoe=6,
        required_yoe=4,
        candidate_degree="masters",
        required_degree="bachelors",
    )
    assert set(expl.matched_required) == {"Python", "AWS"}
    assert expl.missing_required == ["Kubernetes"]
    assert expl.matched_preferred == ["Docker"]
    assert expl.extra_skills == []


def test_explain_lists_extra_skills_not_in_jd():
    expl = explain(
        candidate_skills=["Python", "Rust", "Go"],
        required_skills=["Python"],
        preferred_skills=[],
        candidate_yoe=None,
        required_yoe=None,
        candidate_degree=None,
        required_degree=None,
    )
    assert set(expl.extra_skills) == {"Rust", "Go"}


def test_explain_yoe_delta_positive():
    expl = explain(
        candidate_skills=[],
        required_skills=[],
        preferred_skills=[],
        candidate_yoe=7,
        required_yoe=4,
        candidate_degree=None,
        required_degree=None,
    )
    assert expl.yoe_delta == 3


def test_explain_yoe_delta_none_when_unknown():
    expl = explain(
        candidate_skills=[],
        required_skills=[],
        preferred_skills=[],
        candidate_yoe=None,
        required_yoe=5,
        candidate_degree=None,
        required_degree=None,
    )
    assert expl.yoe_delta is None


def test_explain_degree_does_not_meet_requirement():
    expl = explain(
        candidate_skills=[],
        required_skills=[],
        preferred_skills=[],
        candidate_yoe=None,
        required_yoe=None,
        candidate_degree="associate",
        required_degree="bachelors",
    )
    assert expl.degree_meets_requirement is False


def test_explain_degree_meets_requirement_when_equal():
    expl = explain(
        candidate_skills=[],
        required_skills=[],
        preferred_skills=[],
        candidate_yoe=None,
        required_yoe=None,
        candidate_degree="bachelors",
        required_degree="bachelors",
    )
    assert expl.degree_meets_requirement is True
