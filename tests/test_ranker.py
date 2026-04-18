from src.extraction.education_extractor import EducationMatch
from src.extraction.entity_extractor import ResumeEntities
from src.matching.jd_parser import JobRequirements
from src.matching.ranker import rank_candidates


def _candidate(
    filename: str,
    skills: list[str],
    *,
    yoe: int | None = None,
    degree: str | None = None,
    text: str | None = None,
) -> ResumeEntities:
    degrees = []
    if degree == "masters":
        degrees = [EducationMatch(name="M.S.", tier="masters")]
    elif degree == "bachelors":
        degrees = [EducationMatch(name="B.S.", tier="bachelors")]
    return ResumeEntities(
        source_filename=filename,
        cleaned_text=text or " ".join(skills),
        skills=skills,
        degrees=degrees,
        yoe=yoe,
    )


def test_rank_candidates_orders_by_overall_score():
    jd = JobRequirements(
        raw_text="Senior Python Data Scientist with Machine Learning and AWS.",
        required_skills=["Python", "Machine Learning", "AWS"],
        min_yoe=5,
        required_degree="bachelors",
    )
    candidates = [
        _candidate("strong.pdf", ["Python", "Machine Learning", "AWS"], yoe=6, degree="masters",
                   text="Python data scientist with machine learning and AWS at FAANG."),
        _candidate("weak.pdf", ["Java", "Spring"], yoe=2, degree=None,
                   text="Java backend developer with Spring Boot."),
        _candidate("medium.pdf", ["Python", "AWS"], yoe=3, degree="bachelors",
                   text="Python engineer with some AWS and SQL experience."),
    ]
    ranked = rank_candidates(jd, candidates, min_score=0.0)
    assert [r.filename for r in ranked[:3]] == ["strong.pdf", "medium.pdf", "weak.pdf"]
    assert ranked[0].rank == 1
    assert ranked[1].rank == 2
    assert ranked[2].rank == 3
    assert ranked[0].score.overall > ranked[1].score.overall > ranked[2].score.overall


def test_rank_candidates_applies_threshold():
    jd = JobRequirements(
        raw_text="Python and AWS",
        required_skills=["Python", "AWS", "Kubernetes", "Docker"],
        min_yoe=5,
    )
    candidates = [
        _candidate("strong.pdf", ["Python", "AWS", "Kubernetes", "Docker"], yoe=6),
        _candidate("empty.pdf", [], yoe=0, text="N/A"),
    ]
    ranked = rank_candidates(jd, candidates, min_score=0.5)
    filenames = [r.filename for r in ranked]
    assert "strong.pdf" in filenames
    assert "empty.pdf" not in filenames


def test_rank_candidates_respects_top_n():
    jd = JobRequirements(raw_text="python", required_skills=["Python"])
    candidates = [_candidate(f"r{i}.pdf", ["Python"], text="python dev") for i in range(5)]
    ranked = rank_candidates(jd, candidates, min_score=0.0, top_n=2)
    assert len(ranked) == 2


def test_rank_candidates_empty_list():
    jd = JobRequirements(raw_text="anything", required_skills=["Python"])
    assert rank_candidates(jd, []) == []


def test_rank_candidates_attaches_explanation():
    jd = JobRequirements(
        raw_text="Python and AWS",
        required_skills=["Python", "AWS"],
        preferred_skills=["Docker"],
    )
    cand = _candidate("x.pdf", ["Python", "Docker", "Go"], text="python aws docker go")
    ranked = rank_candidates(jd, [cand], min_score=0.0)
    assert len(ranked) == 1
    expl = ranked[0].explanation
    assert "Python" in expl.matched_required
    assert "AWS" in expl.missing_required
    assert "Docker" in expl.matched_preferred
    assert "Go" in expl.extra_skills
