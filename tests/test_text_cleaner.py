from src.parsing.text_cleaner import clean_text


def test_clean_text_empty():
    assert clean_text("") == ""


def test_clean_text_collapses_whitespace():
    raw = "Hello    world  \n\n\n\n  Python  "
    cleaned = clean_text(raw)
    assert "Hello world" in cleaned
    assert "Python" in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_strips_bullets():
    raw = "• Led a team of 5\n● Built pipelines\n- Shipped features"
    cleaned = clean_text(raw)
    assert cleaned.startswith("Led a team of 5")
    assert "Built pipelines" in cleaned
    assert "Shipped features" in cleaned
    assert "•" not in cleaned
    assert "●" not in cleaned


def test_clean_text_unicode_normalization():
    # Fullwidth "Ｐｙｔｈｏｎ" should normalize to ASCII "Python"
    raw = "Ｐｙｔｈｏｎ Developer"
    assert clean_text(raw) == "Python Developer"


def test_clean_text_preserves_line_breaks():
    raw = "Line one\nLine two\nLine three"
    cleaned = clean_text(raw)
    assert cleaned.count("\n") == 2
