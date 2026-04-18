"""Single shared spaCy pipeline instance used by every extractor."""
from __future__ import annotations

from functools import lru_cache

import spacy
from spacy.language import Language

from src.config import SPACY_BASE_MODEL


@lru_cache(maxsize=1)
def get_nlp() -> Language:
    """Return a cached spaCy pipeline. Loaded on first call."""
    return spacy.load(SPACY_BASE_MODEL)
