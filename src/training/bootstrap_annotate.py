"""Auto-label resume text with SKILL/DEGREE/JOB_TITLE/YOE spans via the
taxonomies from Phase 1. Writes spaCy DocBin files for training.

This is the "weak supervision" step: we use PhraseMatcher + regex to get
cheap, noisy labels, then (optionally) hand-correct a fraction of them in
manual_review.py before training.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from spacy.language import Language
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, DocBin, Span

from src.config import (
    DEGREES_TAXONOMY_PATH,
    JOB_TITLES_TAXONOMY_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SKILLS_TAXONOMY_PATH,
)
from src.extraction.nlp_loader import get_nlp
from src.parsing.resume_parser import parse_resume


YOE_PATTERN = re.compile(
    r"\b(\d{1,2}\+?(?:\s*-\s*\d{1,2})?)\s+(?:year|yr)s?\b",
    re.IGNORECASE,
)

LABELS = ("SKILL", "DEGREE", "JOB_TITLE", "YOE")


@dataclass
class AnnotationStats:
    total_docs: int = 0
    failed: int = 0
    by_label: dict[str, int] = field(default_factory=lambda: {lbl: 0 for lbl in LABELS})
    train_docs: int = 0
    dev_docs: int = 0

    def as_dict(self) -> dict:
        return {
            "total_docs": self.total_docs,
            "failed": self.failed,
            "by_label": dict(self.by_label),
            "train_docs": self.train_docs,
            "dev_docs": self.dev_docs,
        }


def _load_taxonomy_terms(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    seen: dict[str, str] = {}
    for items in data.values():
        for item in items:
            seen.setdefault(item.lower(), item)
    # Longest first so PhraseMatcher's first hit tends to be the more specific phrase.
    return sorted(seen.values(), key=lambda s: (-len(s), s.lower()))


def build_matchers(nlp: Language) -> dict[str, PhraseMatcher]:
    matchers: dict[str, PhraseMatcher] = {}
    for label, path in (
        ("SKILL", SKILLS_TAXONOMY_PATH),
        ("DEGREE", DEGREES_TAXONOMY_PATH),
        ("JOB_TITLE", JOB_TITLES_TAXONOMY_PATH),
    ):
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        matcher.add(label, [nlp.make_doc(t) for t in _load_taxonomy_terms(path)])
        matchers[label] = matcher
    return matchers


def _resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Greedy longest-match: sort by length desc, take if no token overlap with kept."""
    spans = sorted(spans, key=lambda s: (-(s.end - s.start), s.start))
    kept: list[Span] = []
    taken: set[int] = set()
    for span in spans:
        token_range = set(range(span.start, span.end))
        if token_range & taken:
            continue
        kept.append(span)
        taken |= token_range
    kept.sort(key=lambda s: s.start)
    return kept


def annotate_text(
    text: str,
    nlp: Language,
    matchers: dict[str, PhraseMatcher],
) -> Doc:
    """Return a Doc whose .ents are the auto-generated spans."""
    doc = nlp.make_doc(text)
    candidates: list[Span] = []

    for label, matcher in matchers.items():
        for _match_id, start, end in matcher(doc):
            candidates.append(Span(doc, start, end, label=label))

    for match in YOE_PATTERN.finditer(text):
        span = doc.char_span(match.start(), match.end(), label="YOE", alignment_mode="contract")
        if span is not None:
            candidates.append(span)

    doc.ents = _resolve_overlaps(candidates)
    return doc


def collect_resume_paths(
    roots: Iterable[Path],
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[Path]:
    """Gather all PDF/DOCX files under the given roots. If *limit* is set,
    sample evenly across category subfolders when possible.
    """
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(root.rglob("*.pdf"))
        files.extend(root.rglob("*.docx"))
    files.sort()
    if limit is not None and len(files) > limit:
        rng = random.Random(seed)
        files = rng.sample(files, limit)
    return files


def build_docbin(
    resume_paths: list[Path],
    output_dir: Path,
    *,
    train_ratio: float = 0.8,
    seed: int = 42,
    verbose: bool = False,
) -> AnnotationStats:
    nlp = get_nlp()
    matchers = build_matchers(nlp)
    stats = AnnotationStats()

    docs: list[Doc] = []
    for path in resume_paths:
        try:
            resume = parse_resume(path)
            if not resume.cleaned_text:
                continue
            doc = annotate_text(resume.cleaned_text, nlp, matchers)
            if not doc.ents:
                # Skip docs with zero auto-labels — they give no training signal
                # and add noise.
                continue
            docs.append(doc)
            stats.total_docs += 1
            for ent in doc.ents:
                stats.by_label[ent.label_] = stats.by_label.get(ent.label_, 0) + 1
            if verbose:
                print(f"  {path.name}: {len(doc.ents)} entities", file=sys.stderr)
        except Exception as exc:
            stats.failed += 1
            if verbose:
                print(f"  FAILED {path.name}: {exc}", file=sys.stderr)

    if not docs:
        raise RuntimeError("No annotated docs produced — check your inputs.")

    rng = random.Random(seed)
    rng.shuffle(docs)
    split = max(1, int(len(docs) * train_ratio))
    train_docs = docs[:split]
    dev_docs = docs[split:] or [docs[-1]]  # always keep at least one dev doc

    output_dir.mkdir(parents=True, exist_ok=True)
    DocBin(docs=train_docs).to_disk(output_dir / "train.spacy")
    DocBin(docs=dev_docs).to_disk(output_dir / "dev.spacy")

    stats.train_docs = len(train_docs)
    stats.dev_docs = len(dev_docs)
    return stats


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of resumes to annotate (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DATA_DIR / "ner",
        help="Where to write train.spacy and dev.spacy",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of docs for training vs dev (default 0.8)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    roots = [
        RAW_DATA_DIR / "snehaanbhawal_resumes" / "data" / "data",
        RAW_DATA_DIR / "snehaanbhawal_resumes" / "data",
    ]
    paths = collect_resume_paths(roots, limit=args.limit, seed=args.seed)
    if not paths:
        print(
            "No resumes found. Ensure data/raw/snehaanbhawal_resumes/ is populated.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Annotating {len(paths)} resumes → {args.output_dir} ...")
    stats = build_docbin(
        paths,
        args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed,
        verbose=args.verbose,
    )
    print(json.dumps(stats.as_dict(), indent=2))


if __name__ == "__main__":
    _cli()
