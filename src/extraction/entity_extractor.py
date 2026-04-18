"""Orchestrator: run every extractor over a ParsedResume."""
from __future__ import annotations

from dataclasses import dataclass, field

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


def extract_entities(resume: ParsedResume) -> ResumeEntities:
    text = resume.cleaned_text
    header = resume.section("header") or text[:600]
    return ResumeEntities(
        source_filename=resume.filename,
        name=extract_name(header),
        email=extract_email(text),
        phone=extract_phone(text),
        skills=extract_skills(text),
        degrees=extract_education(text),
        titles=extract_titles(text),
        yoe=extract_yoe(text),
        sections=resume.sections,
    )
