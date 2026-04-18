from src.extraction.yoe_extractor import extract_yoe


def test_extract_yoe_plus_pattern():
    assert extract_yoe("5+ years of experience") == 5


def test_extract_yoe_of_experience():
    assert extract_yoe("3 years of experience in Python") == 3


def test_extract_yoe_range_returns_max():
    # Range mentions contribute both endpoints; we report the max across all matches.
    assert extract_yoe("5-7 years in data science") == 7


def test_extract_yoe_over_pattern():
    assert extract_yoe("over 10 years of experience") == 10


def test_extract_yoe_returns_max():
    text = "2 years at Acme. Then 4 years at Beta. Total 7 years of experience."
    assert extract_yoe(text) == 7


def test_extract_yoe_word_form():
    assert extract_yoe("five years of experience") == 5


def test_extract_yoe_none_when_no_match():
    assert extract_yoe("I love Python.") is None
    assert extract_yoe("") is None


def test_extract_yoe_caps_at_50():
    # "100 years" is almost certainly garbage and should be ignored.
    assert extract_yoe("I have 100 years of experience") is None
