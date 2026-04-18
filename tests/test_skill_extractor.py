from src.extraction.skill_extractor import extract_skills


def test_extract_skills_finds_multiple():
    text = "I know Python, JavaScript, and AWS. Also worked with Docker and Kubernetes."
    skills = extract_skills(text)
    for expected in ["Python", "JavaScript", "AWS", "Docker", "Kubernetes"]:
        assert expected in skills, f"missing {expected}"


def test_extract_skills_case_insensitive():
    text = "python and TENSORFLOW and ReactJS"
    skills = extract_skills(text)
    assert any(s.lower() == "python" for s in skills)
    assert any(s.lower() == "tensorflow" for s in skills)


def test_extract_skills_dedupes():
    text = "Python Python Python and more Python"
    assert extract_skills(text).count("Python") == 1


def test_extract_skills_empty_returns_empty():
    assert extract_skills("") == []
    assert extract_skills("the quick brown fox") == []


def test_extract_skills_multi_word_phrases():
    text = "Experience with Machine Learning and Natural Language Processing."
    skills = extract_skills(text)
    assert "Machine Learning" in skills
    assert "Natural Language Processing" in skills
