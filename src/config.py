"""Central configuration: paths, model choices, scoring weights, and defaults."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"
MODELS_DIR = ROOT_DIR / "models"
EXPORTS_DIR = ROOT_DIR / "exports"
LOGS_DIR = ROOT_DIR / "logs"

SKILLS_TAXONOMY_PATH = ROOT_DIR / "src" / "resources" / "skills_taxonomy.json"
DEGREES_TAXONOMY_PATH = ROOT_DIR / "src" / "resources" / "degrees_taxonomy.json"
JOB_TITLES_TAXONOMY_PATH = ROOT_DIR / "src" / "resources" / "job_titles_taxonomy.json"

SPACY_BASE_MODEL = "en_core_web_md"
CUSTOM_NER_MODEL_DIR = MODELS_DIR / "custom_ner"
USE_CUSTOM_NER = False  # flipped on once Phase 6 completes


@dataclass
class ScoringWeights:
    """Weights for the composite match score. Must sum to 1.0."""
    skills: float = 0.45
    semantic: float = 0.30
    experience: float = 0.15
    education: float = 0.10

    def __post_init__(self) -> None:
        total = self.skills + self.semantic + self.experience + self.education
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Scoring weights must sum to 1.0, got {total}")


@dataclass
class ScreeningDefaults:
    min_score_threshold: float = 0.40    # 0.0–1.0; candidates below are filtered
    top_n: int = 25                      # cap on returned candidates
    weights: ScoringWeights = field(default_factory=ScoringWeights)


DEFAULTS = ScreeningDefaults()

SUPPORTED_RESUME_EXTENSIONS = {".pdf", ".docx"}
SUPPORTED_JD_EXTENSIONS = {".pdf", ".docx", ".txt"}
