"""Terminal-based review of bootstrap-annotated documents.

Lets you page through docs in a DocBin, see what was auto-labeled, and drop
obviously-wrong spans. Not a replacement for Prodigy/Doccano, but enough to
clean up the worst false positives before training.

Keys: n=next  p=prev  d=delete an entity  s=save+quit  q=quit without saving
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from spacy.tokens import DocBin
from spacy.vocab import Vocab

from src.config import PROCESSED_DATA_DIR
from src.extraction.nlp_loader import get_nlp


_ANSI = {
    "SKILL": "\033[94m",       # blue
    "DEGREE": "\033[92m",      # green
    "JOB_TITLE": "\033[93m",   # yellow
    "YOE": "\033[95m",         # magenta
    "_RESET": "\033[0m",
}


def _highlight(doc, width: int = 2000) -> str:
    """Inline-highlight entities in the doc's text (truncated)."""
    text = doc.text
    segments: list[str] = []
    cursor = 0
    for ent in sorted(doc.ents, key=lambda e: e.start_char):
        if ent.start_char > cursor:
            segments.append(text[cursor : ent.start_char])
        color = _ANSI.get(ent.label_, "")
        reset = _ANSI["_RESET"]
        segments.append(f"{color}[{ent.text}|{ent.label_}]{reset}")
        cursor = ent.end_char
    segments.append(text[cursor:])
    out = "".join(segments)
    if len(text) > width:
        out = out[:width] + f"\n... (truncated, full length {len(text)})"
    return out


def _show(doc, idx: int, total: int) -> None:
    print("\n" + "=" * 78)
    print(f"Doc {idx + 1}/{total} — {len(doc.ents)} entities")
    print("=" * 78)
    # Legend
    legend = "  ".join(f"{_ANSI[l]}{l}{_ANSI['_RESET']}" for l in ("SKILL", "DEGREE", "JOB_TITLE", "YOE"))
    print(legend)
    print()
    print(_highlight(doc))
    print()
    print("Entities:")
    for i, ent in enumerate(doc.ents):
        print(f"  [{i}] {ent.label_:<10} {ent.text!r}")


def _delete_entity(doc, index: int):
    ents = list(doc.ents)
    if not 0 <= index < len(ents):
        print(f"Bad index {index}")
        return doc
    removed = ents.pop(index)
    doc.ents = ents
    print(f"Removed {removed.label_}:{removed.text!r}")
    return doc


def run(docbin_path: Path, output_path: Path | None = None) -> None:
    nlp = get_nlp()
    doc_bin = DocBin().from_disk(docbin_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    if not docs:
        print("No docs found.", file=sys.stderr)
        return

    idx = 0
    modified = False
    while 0 <= idx < len(docs):
        _show(docs[idx], idx, len(docs))
        cmd = input("[n]ext / [p]rev / [d] delete / [s]ave+quit / [q]uit: ").strip().lower()
        if cmd in ("n", ""):
            idx += 1
        elif cmd == "p":
            idx = max(0, idx - 1)
        elif cmd.startswith("d"):
            try:
                i = int(cmd[1:].strip())
            except ValueError:
                raw = input("Entity index to delete: ").strip()
                try:
                    i = int(raw)
                except ValueError:
                    continue
            docs[idx] = _delete_entity(docs[idx], i)
            modified = True
        elif cmd == "s":
            break
        elif cmd == "q":
            if modified:
                confirm = input("Discard changes? [y/N] ").strip().lower()
                if confirm != "y":
                    continue
            return
        else:
            print("Unknown command.")

    if modified and output_path is not None:
        DocBin(docs=docs).to_disk(output_path)
        print(f"Saved to {output_path}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROCESSED_DATA_DIR / "ner" / "train.spacy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the reviewed DocBin. Defaults to overwriting --input.",
    )
    args = parser.parse_args()
    out = args.output or args.input
    run(args.input, out)


if __name__ == "__main__":
    _cli()
