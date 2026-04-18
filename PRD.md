# Resume Screening System — Product Requirements Document

**Owner:** dev.coreops26@gmail.com
**Last updated:** 2026-04-18
**Status:** Draft v1

---

## 1. Problem Statement

Recruiters and hiring managers receive hundreds of resumes per opening. Manual
screening is slow, inconsistent, and prone to missing qualified candidates.
We need an automated system that ranks resumes against a job description and
produces an interpretable shortlist in seconds.

## 2. Goals & Non-Goals

### Goals
- Automatically rank a batch of resumes against a single job description.
- Extract structured entities from resumes (skills, education, job titles, years of experience).
- Provide transparent, per-candidate match explanations (matched vs missing skills).
- Allow flexible JD input (paste, upload, or structured form).
- Run in a browser (Streamlit UI) with exportable results.
- Deployable to Streamlit Community Cloud for public demo.

### Non-Goals (v1)
- Multi-JD batch screening in a single run.
- Candidate ATS integration (Greenhouse, Lever, etc.).
- Interview scheduling / candidate communication.
- Bias auditing / fairness analysis (flagged as future work).
- Authentication or multi-tenant support.

## 3. Target User

- Primary: HR / recruiters running shortlists for a single role.
- Secondary: Developer using the tool as a portfolio / learning project.

## 4. Functional Requirements

### 4.1 Resume Ingestion
- Accept multi-file upload of **PDF** and **DOCX** resumes.
- Parse text while preserving section structure (education, experience, skills).
- Gracefully handle unparseable files with a clear error per file.

### 4.2 Job Description Input (all three supported)
1. **Paste JD text** into a textarea.
2. **Upload JD file** (PDF / DOCX / TXT).
3. **Structured form**: title, required skills, preferred skills, min years of experience, education.

### 4.3 Entity Extraction (NER)
- Custom spaCy NER model trained to recognize:
  - `SKILL` (technical + soft)
  - `DEGREE` (B.Tech, MBA, PhD, etc.)
  - `JOB_TITLE` (Software Engineer, Data Analyst, etc.)
  - `YEARS_OF_EXPERIENCE` (e.g. "3 years", "5+ yrs")
- MVP fallback: pretrained spaCy + PhraseMatcher until the custom model is trained.
- Training data sourced via **bootstrap + fine-tune**: PhraseMatcher auto-labels
  Kaggle resumes, ~200–300 samples manually corrected, then spaCy NER trained.

### 4.4 Scoring & Ranking
- Core algorithm: **TF-IDF vectorization + cosine similarity** between resume and JD.
- Composite score combines:
  - Skills overlap (explicit matched/missing skills from NER)
  - Text semantic similarity (TF-IDF cosine)
  - Experience match (candidate YOE vs required YOE)
  - Education match (degree vs required degree)
- Weighted score configurable; sensible defaults provided.
- Minimum-threshold filter auto-rejects resumes below a cutoff.

### 4.5 Match Explanation
- Per-candidate panel shows:
  - Final score (0–100)
  - Matched required skills (green)
  - Missing required skills (red)
  - Matched preferred skills
  - Detected YOE vs required YOE
  - Detected education vs required education

### 4.6 Output & Export
- Ranked results table on the UI (sortable).
- Export shortlist as **CSV** and **PDF**.

### 4.7 Deployment
- **Streamlit Community Cloud** public URL.
- Model files bundled or downloaded on first run (respecting free-tier disk limits).
- `requirements.txt` + `packages.txt` for system deps.

## 5. Non-Functional Requirements

- **Performance:** Rank 50 resumes against a JD in under 30 seconds on Streamlit Cloud.
- **Accuracy (initial target):** Skill extraction F1 ≥ 0.80 on hold-out set.
- **Observability:** Log each run (JD + N resumes + scores) to a local file for debugging.
- **Privacy:** Resumes are processed in memory; nothing persisted beyond the session unless exported.

## 6. Data

- **Primary dataset:** Kaggle "Resume Dataset" (~2,400 labeled resumes across categories).
- **Supplementary:** A handful of real resumes for realistic end-to-end testing.
- **Skills taxonomy:** Curated list of ~500 technical + soft skills (versioned in repo).

## 7. Tech Stack

| Layer           | Tool                                     |
|-----------------|------------------------------------------|
| Language        | Python 3.10+                             |
| PDF parsing     | `pdfplumber`                             |
| DOCX parsing    | `python-docx`                            |
| NLP             | `spaCy` (en_core_web_md), custom NER     |
| Vectorization   | `scikit-learn` TfidfVectorizer           |
| Similarity      | `scikit-learn` cosine_similarity         |
| UI              | `Streamlit`                              |
| Export (PDF)    | `reportlab` or `fpdf2`                   |
| Deployment      | Streamlit Community Cloud                |

## 8. Milestones & Phase Plan

Week 1 goal: MVP with **pretrained NER + PhraseMatcher** (Phases 1–5).
Post-MVP: train custom NER and deploy (Phases 6–7).

| Phase | Deliverable                                       | Est. Days |
|-------|---------------------------------------------------|-----------|
| 1     | Project scaffold, deps, skills taxonomy, dataset  | 0.5       |
| 2     | Resume parsing (PDF + DOCX) + cleaning            | 1         |
| 3     | Entity extraction (PhraseMatcher + regex + spaCy) | 1         |
| 4     | JD processing + TF-IDF matching engine + scoring  | 1.5       |
| 5     | Streamlit UI + CSV/PDF export                     | 1.5       |
| 6     | Custom spaCy NER training (bootstrap + fine-tune) | 2         |
| 7     | Streamlit Cloud deployment + polish               | 0.5       |

## 9. Success Criteria

- End-to-end: user can upload 10 resumes + JD and see a ranked list in < 10s locally.
- Top-3 ranked resumes manually judged "reasonable" for 5 test JDs.
- Skill NER F1 ≥ 0.80 on hold-out set after Phase 6.
- App deployed to a public Streamlit Cloud URL.

## 10. Open Questions / Risks

- **Streamlit Cloud resource limits** may constrain model size (en_core_web_md is ~40MB; custom NER adds on top). Mitigation: use `sm` variant if needed.
- **Kaggle dataset quality** — resumes are category-labeled, not entity-labeled. Bootstrap annotation accuracy depends heavily on taxonomy quality.
- **PDF parsing edge cases** — multi-column layouts, tables, and scanned PDFs. Scanned PDFs are out of scope for v1.
- **Bias** — TF-IDF + skills matching can encode historical bias present in the JD wording. Flag as future work for a fairness pass.
