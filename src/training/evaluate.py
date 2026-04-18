"""Evaluate a trained NER model: per-label precision/recall/F1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example

from src.config import CUSTOM_NER_MODEL_DIR, PROCESSED_DATA_DIR


def evaluate_model(model_path: Path, dev_path: Path) -> dict:
    nlp = spacy.load(model_path)
    doc_bin = DocBin().from_disk(dev_path)
    gold_docs = list(doc_bin.get_docs(nlp.vocab))

    examples: list[Example] = []
    for gold in gold_docs:
        pred = nlp(gold.text)
        examples.append(Example(pred, gold))

    scorer = Scorer()
    scores = scorer.score(examples)

    per_label_raw = scores.get("ents_per_type", {}) or {}
    per_label = {
        label: {"precision": m["p"], "recall": m["r"], "f1": m["f"]}
        for label, m in per_label_raw.items()
    }

    return {
        "precision": scores.get("ents_p") or 0.0,
        "recall": scores.get("ents_r") or 0.0,
        "f1": scores.get("ents_f") or 0.0,
        "per_label": per_label,
        "n_dev_docs": len(gold_docs),
    }


def format_report(report: dict) -> str:
    lines = [
        f"Dev docs: {report['n_dev_docs']}",
        "",
        f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}",
        "-" * 48,
        f"{'OVERALL':<12} {report['precision']:>10.3f} {report['recall']:>10.3f} {report['f1']:>10.3f}",
    ]
    for label in sorted(report["per_label"]):
        m = report["per_label"][label]
        lines.append(
            f"{label:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f}"
        )
    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=CUSTOM_NER_MODEL_DIR / "model-best",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=PROCESSED_DATA_DIR / "ner" / "dev.spacy",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found at {args.model}. Train first.", file=sys.stderr)
        sys.exit(1)
    if not args.dev.exists():
        print(f"Dev set not found at {args.dev}.", file=sys.stderr)
        sys.exit(1)

    report = evaluate_model(args.model, args.dev)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))


if __name__ == "__main__":
    _cli()
