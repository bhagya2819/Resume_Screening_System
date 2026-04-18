from src.parsing.section_detector import detect_sections


def test_detect_sections_basic():
    text = """Jane Doe
jane@example.com

Summary
Experienced software engineer.

Experience
Acme Corp - Senior Engineer, 2020-present

Education
MS Computer Science, Stanford, 2018

Skills
Python, Java, AWS"""
    sections = detect_sections(text)
    assert "header" in sections
    assert "jane@example.com" in sections["header"]
    assert "Experienced software engineer" in sections["summary"]
    assert "Acme Corp" in sections["experience"]
    assert "Stanford" in sections["education"]
    assert "Python, Java, AWS" in sections["skills"]


def test_detect_sections_case_insensitive():
    text = "EXPERIENCE\nAcme\n\nEDUCATION\nMIT"
    sections = detect_sections(text)
    assert "Acme" in sections["experience"]
    assert "MIT" in sections["education"]


def test_detect_sections_variant_headers():
    text = "Work Experience\nJob 1\n\nTechnical Skills\nPython"
    sections = detect_sections(text)
    assert "Job 1" in sections["experience"]
    assert "Python" in sections["skills"]


def test_detect_sections_header_only_before_sections():
    text = "John Smith\njohn@x.com\n(555) 123-4567"
    sections = detect_sections(text)
    assert sections == {"header": "John Smith\njohn@x.com\n(555) 123-4567"}


def test_detect_sections_ignores_long_lines():
    # A line with 'Experience' in it but too long to be a header should not be treated as one.
    text = "I have five years of Experience with distributed systems."
    sections = detect_sections(text)
    assert "experience" not in sections
    assert "header" in sections
