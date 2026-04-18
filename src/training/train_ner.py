"""Train a spaCy NER model from bootstrap-annotated DocBin files.

Usage:
    python -m src.training.train_ner \
        --train data/processed/ner/train.spacy \
        --dev data/processed/ner/dev.spacy

Uses a generated config (transfer-learning from en_core_web_md with a fresh
NER component) so the model inherits the base tokenizer + vectors.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.config import (
    CUSTOM_NER_MODEL_DIR,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
    SPACY_BASE_MODEL,
)


DEFAULT_CONFIG_PATH = ROOT_DIR / "src" / "training" / "ner_config.cfg"


def ensure_config(config_path: Path = DEFAULT_CONFIG_PATH, *, overwrite: bool = False) -> Path:
    """Generate a training config via `spacy init config` if absent.

    `spacy init config --optimize accuracy` defaults to `en_core_web_lg` vectors
    which we don't ship. Patch the generated file to use `en_core_web_md`.
    """
    if config_path.exists() and not overwrite:
        return config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "spacy",
        "init",
        "config",
        str(config_path),
        "--lang",
        "en",
        "--pipeline",
        "ner",
        "--optimize",
        "accuracy",
    ]
    if overwrite:
        cmd.append("--force")
    subprocess.run(cmd, check=True)

    text = config_path.read_text()
    text = text.replace('vectors = "en_core_web_lg"', f'vectors = "{SPACY_BASE_MODEL}"')
    config_path.write_text(text)
    return config_path


def run_training(
    train_path: Path,
    dev_path: Path,
    output_dir: Path = CUSTOM_NER_MODEL_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    extra_overrides: list[str] | None = None,
) -> None:
    """Invoke `spacy train` via subprocess. Raises CalledProcessError on failure."""
    ensure_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "spacy",
        "train",
        str(config_path),
        "--output",
        str(output_dir),
        "--paths.train",
        str(train_path),
        "--paths.dev",
        str(dev_path),
    ]
    if extra_overrides:
        cmd.extend(extra_overrides)

    print(f"$ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=PROCESSED_DATA_DIR / "ner" / "train.spacy",
    )
    parser.add_argument(
        "--dev",
        type=Path,
        default=PROCESSED_DATA_DIR / "ner" / "dev.spacy",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CUSTOM_NER_MODEL_DIR,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        help="Cap on training epochs (overrides config default)",
    )
    args = parser.parse_args()

    for path in (args.train, args.dev):
        if not path.exists():
            print(f"Missing {path}. Run bootstrap_annotate.py first.", file=sys.stderr)
            sys.exit(1)

    run_training(
        args.train,
        args.dev,
        args.output_dir,
        args.config,
        extra_overrides=[
            "--training.max_epochs",
            str(args.max_epochs),
        ],
    )
    print(f"Model written to {args.output_dir / 'model-best'}")


if __name__ == "__main__":
    _cli()
