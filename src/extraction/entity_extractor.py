"""Orchestrator: run every extractor over a ParsedResume.

Two extraction backends:
  1. Pretrained + PhraseMatcher (default) — the Phase 3 pipeline.
  2. Custom spaCy NER (opt-in) — the Phase 6 pipeline. Used when either
     config.USE_CUSTOM_NER is True or the caller passes use_custom_ner=True.
Contact info (email, phone) and name always use the regex / pretrained
PERSON NER path regardless of backend.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import USE_CUSTOM_NER
from src.extraction.contact_extractor import extract_email, extract_phone
from src.extraction.education_extractor import (
    EducationMatch,
    extract_education,
    highest_tier,
)
from src.extraction.name_extractor import extract_name
from src.extraction.skill_extractor import extract_skills
from src.extraction.title_extractor import extract_titles
from src.extraction.yoe_extractor import extract_yoe
from src.parsing.resume_parser import ParsedResume


@dataclass
class ResumeEntities:
    source_filename: str
    cleaned_text: str = ""
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    skills: list[str] = field(default_factory=list)
    degrees: list[EducationMatch] = field(default_factory=list)
    titles: list[str] = field(default_factory=list)
    yoe: int | None = None
    sections: dict[str, str] = field(default_factory=dict)

    @property
    def highest_degree_tier(self) -> str | None:
        return highest_tier(self.degrees)


def extract_entities(
    resume: ParsedResume,
    *,
    use_custom_ner: bool | None = None,
) -> ResumeEntities:
    text = resume.cleaned_text
    header = resume.section("header") or text[:600]

    use_custom = USE_CUSTOM_NER if use_custom_ner is None else use_custom_ner

    if use_custom:
        # Deferred import: loading spacy.load() on the custom model is heavy,
        # so only do it when actually requested.
        from src.extraction.custom_ner_extractor import extract_all_entities

        ner = extract_all_entities(text)
        skills = ner["skills"]
        degrees = ner["degrees"]
        titles = ner["titles"]
        yoe = ner["yoe"]
    else:
        skills = extract_skills(text)
        degrees = extract_education(text)
        titles = extract_titles(text)
        yoe = extract_yoe(text)

    return ResumeEntities(
        source_filename=resume.filename,
        cleaned_text=text,
        name=extract_name(header),
        email=extract_email(text),
        phone=extract_phone(text),
        skills=skills,
        degrees=degrees,
        titles=titles,
        yoe=yoe,
        sections=resume.sections,
    )
