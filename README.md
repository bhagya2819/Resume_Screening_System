# Resume Screening System

Automatically rank resumes against a job description using NLP.

- **Input:** Multiple resumes (PDF/DOCX) + a job description (paste, upload, or structured form)
- **NLP:** spaCy NER (custom-trained), PhraseMatcher for skills, regex for YOE
- **Matching:** TF-IDF + cosine similarity, composite score across skills / semantic / experience / education
- **Output:** Ranked candidate list with per-candidate match explanation, CSV/PDF export
- **UI:** Streamlit (deployable to Streamlit Community Cloud)

See [`PRD.md`](PRD.md) for the full product spec and phase plan.

## Project Structure

```
Resume_Screening_System/
├── PRD.md                     Product requirements document
├── requirements.txt           Python dependencies
├── data/
│   ├── raw/                   Kaggle datasets (not committed)
│   ├── processed/             Intermediate artifacts
│   └── samples/               Sample JD/resume for testing
├── src/
│   ├── config.py              Paths, weights, defaults
│   ├── resources/             Skill/degree/job-title taxonomies
│   ├── parsing/               PDF + DOCX parsers
│   ├── extraction/            Entity extractors (skills, YOE, degree, title)
│   ├── matching/              JD parser, TF-IDF, scorer, ranker
│   ├── training/              Custom spaCy NER training (Phase 6)
│   └── utils/
├── models/                    Trained custom NER (Phase 6 output)
├── ui/
│   └── app.py                 Streamlit app (Phase 5)
├── tests/
├── exports/                   CSV/PDF output
└── logs/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

## Datasets

Download manually from Kaggle and place under `data/raw/`:

- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) — PDFs for parsing + ranking tests
- [Resume Entities for NER](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner) — pre-annotated data for Phase 6

Expected layout:

```
data/raw/
├── snehaanbhawal_resumes/
└── dataturks_resume_ner/
```

## Running (once Phase 5 is done)

```bash
streamlit run ui/app.py
```

## Development Phases

| Phase | Deliverable                                       |
|-------|---------------------------------------------------|
| 1     | Scaffold + deps + skills taxonomy + dataset       |
| 2     | Resume parsing (PDF + DOCX) + cleaning            |
| 3     | Entity extraction (PhraseMatcher + regex + spaCy) |
| 4     | JD processing + TF-IDF matching + scoring         |
| 5     | Streamlit UI + CSV/PDF export                     |
| 6     | Custom spaCy NER training                         |
| 7     | Streamlit Cloud deployment                        |
