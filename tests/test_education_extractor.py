from src.extraction.education_extractor import (
    EducationMatch,
    extract_education,
    highest_tier,
)


def test_extract_education_finds_bachelors():
    matches = extract_education("B.Tech in Computer Science from IIT")
    assert any(m.tier == "bachelors" for m in matches)


def test_extract_education_finds_masters_and_phd():
    text = "PhD in Statistics, MBA from Wharton, B.Sc. in Math"
    matches = extract_education(text)
    tiers = {m.tier for m in matches}
    assert "doctorate" in tiers
    assert "masters" in tiers
    assert "bachelors" in tiers


def test_extract_education_empty():
    assert extract_education("") == []


def test_highest_tier_picks_doctorate_over_masters():
    matches = [
        EducationMatch(name="MBA", tier="masters"),
        EducationMatch(name="PhD", tier="doctorate"),
        EducationMatch(name="B.Tech", tier="bachelors"),
    ]
    assert highest_tier(matches) == "doctorate"


def test_highest_tier_none_for_empty():
    assert highest_tier([]) is None


def test_extract_education_dedupes():
    text = "MBA, MBA, MBA in Finance"
    matches = extract_education(text)
    mbas = [m for m in matches if m.name.lower() == "mba"]
    assert len(mbas) == 1
