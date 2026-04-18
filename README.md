# Resume Screening System

Automatically rank resumes against a job description using NLP.

- **Input:** Multiple resumes (PDF / DOCX) + a job description (paste, upload, or structured form)
- **NLP:** spaCy NER (pretrained baseline + optional custom-trained model) · PhraseMatcher for skills / degrees / titles · regex for YOE / contact
- **Matching:** TF-IDF + cosine similarity · composite score across skills / semantic / experience / education
- **Output:** Ranked candidate list with per-candidate match explanation · CSV and PDF export
- **UI:** Streamlit · deployable to Streamlit Community Cloud

See [`PRD.md`](PRD.md) for the full product spec and phase plan.

## Project Structure

```
Resume_Screening_System/
├── PRD.md                      Product requirements document
├── requirements.txt            Pinned runtime deps (reproducible)
├── requirements-dev.txt        + pytest for local development
├── packages.txt                System packages for Streamlit Cloud (empty by default)
├── pytest.ini
├── .streamlit/config.toml      Theme + upload size + telemetry off
├── data/
│   ├── raw/                    Kaggle datasets (gitignored)
│   ├── processed/              Intermediate artifacts, DocBin files
│   └── samples/
├── src/
│   ├── config.py               Paths, weights, defaults, USE_CUSTOM_NER flag
│   ├── resources/              Skill / degree / job-title taxonomies
│   ├── parsing/                PDF / DOCX parsers, cleaner, section detector
│   ├── extraction/             Entity extractors (PhraseMatcher + custom NER backend)
│   ├── matching/               JD parser, TF-IDF, scorer, ranker, explanation
│   ├── training/               Bootstrap-annotate, train, evaluate, manual review
│   ├── utils/run_logger.py     Appends JSONL record per ranking run
│   └── export.py               CSV + PDF output
├── models/custom_ner/          Trained NER model (gitignored, Phase 6 output)
├── ui/app.py                   Streamlit app
├── tests/                      107+ pytest suite
├── exports/                    CSV / PDF output (gitignored)
└── logs/                       runs.jsonl (gitignored)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The spaCy `en_core_web_md` model is pinned as a wheel URL in `requirements.txt`, so it installs along with everything else.

## Datasets

Download manually from Kaggle and place under `data/raw/`:

- [Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) — PDFs for parsing + ranking tests
- [Resume Entities for NER](https://www.kaggle.com/datasets/dataturks/resume-entities-for-ner) — pre-annotated data (optional supplement for Phase 6)

Expected layout:

```
data/raw/
├── snehaanbhawal_resumes/
│   ├── data/data/<CATEGORY>/*.pdf
│   └── Resume/Resume.csv
└── dataturks_resume_ner/
    └── entities.json
```

## Running locally

```bash
streamlit run ui/app.py
```

Then open `http://localhost:8501` and follow the three tabs:

1. **Upload Resumes** — drag and drop PDF / DOCX files.
2. **Configure Job** — paste text, upload a JD file, or fill the structured form.
3. **Results** — click **Rank candidates**, then download CSV / PDF.

Sidebar sliders control scoring weights (auto-normalized), minimum threshold, and top-N cap.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Training the custom NER model (Phase 6, optional)

The default pipeline uses PhraseMatcher + regex for skills / degrees / titles. To opt in to the custom-trained NER:

```bash
# 1. Bootstrap-annotate a sample of resumes via the taxonomies.
python -m src.training.bootstrap_annotate --limit 500

# 2. Review / clean up noisy auto-labels in the terminal (optional).
python -m src.training.manual_review --input data/processed/ner/train.spacy

# 3. Train the NER model (~25 min on CPU for 500 resumes x 10 epochs).
python -m src.training.train_ner --max-epochs 10

# 4. Evaluate precision / recall / F1 per entity.
python -m src.training.evaluate
```

Then flip `USE_CUSTOM_NER = True` in `src/config.py` to route the extraction pipeline through the custom model.

## Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public).
2. Go to https://share.streamlit.io/ and connect your GitHub account.
3. **New app** → pick this repo → set **Main file path** to `ui/app.py`.
4. Streamlit Cloud reads `requirements.txt` and `packages.txt` automatically. First deploy takes ~3–5 min (numpy / spaCy are the slowest wheels).
5. The deployed app uses `USE_CUSTOM_NER = False` (the default). The 73 MB `models/custom_ner/` is gitignored and not deployed. If you want the custom NER on Cloud, either:
   - Re-train from within the app on first run (slow, uses Cloud CPU minutes), or
   - Host `model-best` on a CDN / S3 and download at startup, or
   - Stick with PhraseMatcher (runs well, no model download needed).

### Resource limits

Streamlit Community Cloud free tier gives ~1 GB RAM and ~1 GB disk. Current install is ~800 MB (spaCy + PyTorch-free md model), well within limits.

## Development Phases

| Phase | Deliverable                                       | Status     |
|-------|---------------------------------------------------|------------|
| 1     | Scaffold + deps + skills taxonomy + dataset       | ✓ Complete |
| 2     | Resume parsing (PDF + DOCX) + cleaning            | ✓ Complete |
| 3     | Entity extraction (PhraseMatcher + regex + spaCy) | ✓ Complete |
| 4     | JD processing + TF-IDF matching + scoring         | ✓ Complete |
| 5     | Streamlit UI + CSV/PDF export                     | ✓ Complete |
| 6     | Custom spaCy NER training                         | ✓ Complete |
| 7     | Streamlit Cloud deployment                        | ✓ Complete |
