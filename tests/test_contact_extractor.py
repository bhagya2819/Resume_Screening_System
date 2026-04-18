from src.extraction.contact_extractor import extract_email, extract_phone


def test_extract_email_standard():
    assert extract_email("Contact: jane.doe@example.com") == "jane.doe@example.com"


def test_extract_email_with_plus():
    assert extract_email("me+work@acme.co.uk here") == "me+work@acme.co.uk"


def test_extract_email_none_when_missing():
    assert extract_email("no email in this text") is None
    assert extract_email("") is None


def test_extract_phone_us_parens():
    assert extract_phone("Call (555) 123-4567 anytime") is not None


def test_extract_phone_dashed():
    assert extract_phone("555-123-4567") is not None


def test_extract_phone_international():
    assert extract_phone("+91 98765 43210") is not None


def test_extract_phone_none_for_short_number():
    assert extract_phone("ext 1234") is None


def test_extract_phone_none_when_missing():
    assert extract_phone("no phone in this text") is None
    assert extract_phone("") is None
