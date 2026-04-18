from src.extraction.title_extractor import extract_titles


def test_extract_titles_basic():
    titles = extract_titles("I worked as a Software Engineer and Data Scientist.")
    assert "Software Engineer" in titles
    assert "Data Scientist" in titles


def test_extract_titles_prefers_longer_specific():
    # Both "Senior Data Scientist" and "Data Scientist" are in the taxonomy;
    # the longer one should win when both match the same span.
    titles = extract_titles("Currently a Senior Data Scientist at Foo Corp.")
    assert "Senior Data Scientist" in titles
    assert "Data Scientist" not in titles


def test_extract_titles_dedupes():
    titles = extract_titles("Software Engineer, Software Engineer, Software Engineer")
    assert titles.count("Software Engineer") == 1


def test_extract_titles_empty():
    assert extract_titles("") == []
    assert extract_titles("No job title here.") == []
