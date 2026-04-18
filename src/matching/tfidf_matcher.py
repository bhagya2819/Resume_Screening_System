"""TF-IDF vectorization + cosine similarity between a JD and resumes."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_cosine(jd_text: str, resume_texts: list[str]) -> list[float]:
    """Return cosine similarity of *jd_text* to each entry in *resume_texts*.

    Fits TF-IDF on [jd + all resumes] in one pass so the IDF weights reflect
    the full candidate pool. Returns values in [0, 1].
    """
    if not resume_texts:
        return []

    corpus = [jd_text] + list(resume_texts)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(corpus)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return [float(s) for s in sims]
