import pytest

from src.extraction.custom_ner_extractor import (
    CustomNERNotTrained,
    _parse_yoe_spans,
    _to_education_matches,
    get_custom_nlp,
)


def test_parse_yoe_spans_takes_max():
    assert _parse_yoe_spans(["3 years", "7 years", "5 years"]) == 7


def test_parse_yoe_spans_handles_plus_and_ranges():
    assert _parse_yoe_spans(["5+ years"]) == 5
    assert _parse_yoe_spans(["5-7 years"]) == 5  # regex picks first digit group


def test_parse_yoe_spans_empty_or_no_digits():
    assert _parse_yoe_spans([]) is None
    assert _parse_yoe_spans(["abc"]) is None


def test_parse_yoe_spans_rejects_impossible_values():
    assert _parse_yoe_spans(["99 years"]) is None  # >50 cap


def test_to_education_matches_resolves_tiers():
    matches = _to_education_matches(["M.S.", "B.Tech", "PhD"])
    tiers = {m.tier for m in matches}
    assert "masters" in tiers
    assert "bachelors" in tiers
    assert "doctorate" in tiers


def test_to_education_matches_unknown_defaults_to_certification():
    matches = _to_education_matches(["Some Obscure Cert"])
    assert matches[0].tier == "certification"


def test_to_education_matches_dedupes():
    matches = _to_education_matches(["MBA", "MBA", "MBA"])
    assert len(matches) == 1


def test_get_custom_nlp_raises_when_not_trained():
    """Before Phase 6 training runs, loading the model should fail cleanly."""
    get_custom_nlp.cache_clear()
    from src.config import CUSTOM_NER_MODEL_DIR
    if (CUSTOM_NER_MODEL_DIR / "model-best").exists():
        pytest.skip("Custom model already trained")
    with pytest.raises(CustomNERNotTrained):
        get_custom_nlp()
