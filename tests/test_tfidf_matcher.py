from src.matching.tfidf_matcher import compute_tfidf_cosine


def test_compute_tfidf_cosine_empty_resumes():
    assert compute_tfidf_cosine("any jd", []) == []


def test_compute_tfidf_cosine_identical_text_scores_highest():
    jd = "Looking for a Python data scientist with machine learning experience."
    resumes = [
        "Python data scientist with machine learning experience at FAANG.",
        "Graphic designer with Adobe Illustrator and Photoshop.",
        "Backend engineer with Java and Spring Boot.",
    ]
    sims = compute_tfidf_cosine(jd, resumes)
    assert len(sims) == 3
    assert sims[0] > sims[1]
    assert sims[0] > sims[2]


def test_compute_tfidf_cosine_values_in_unit_interval():
    jd = "Senior DevOps with Kubernetes."
    resumes = ["Kubernetes DevOps engineer.", "Marketing manager."]
    sims = compute_tfidf_cosine(jd, resumes)
    assert all(0.0 <= s <= 1.0 for s in sims)


def test_compute_tfidf_cosine_completely_unrelated_low_score():
    jd = "Software Engineer with Python and Django."
    resumes = ["Chef experienced in Italian cuisine and pastry."]
    sims = compute_tfidf_cosine(jd, resumes)
    assert sims[0] < 0.2
