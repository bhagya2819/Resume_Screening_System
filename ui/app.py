"""Streamlit app: upload resumes, configure a JD, rank candidates, export."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make src/ importable whether we're run via `streamlit run ui/app.py` or imported.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.config import DEFAULTS, ScoringWeights
from src.export import export_ranked_to_csv, export_ranked_to_pdf, ranked_to_dataframe
from src.extraction.entity_extractor import extract_entities
from src.matching.jd_parser import (
    parse_jd_from_file,
    parse_jd_from_form,
    parse_jd_from_text,
)
from src.matching.ranker import rank_candidates
from src.parsing.resume_parser import parse_resume
from src.utils.run_logger import log_run
from ui.styles import (
    inject_css,
    render_candidate_card,
    render_hero,
    section_header,
)


st.set_page_config(page_title="Resume Screening System", layout="wide")
inject_css()

# --- Session state ---
_defaults = {
    "candidates": [],
    "jd": None,
    "ranked": None,
    "weights": {
        "skills": DEFAULTS.weights.skills,
        "semantic": DEFAULTS.weights.semantic,
        "experience": DEFAULTS.weights.experience,
        "education": DEFAULTS.weights.education,
    },
    "threshold": DEFAULTS.min_score_threshold,
    "top_n": DEFAULTS.top_n,
}
for k, v in _defaults.items():
    st.session_state.setdefault(k, v)


render_hero(
    "Resume Screening System",
    "Rank resumes against a job description using spaCy NER + TF-IDF.",
)


# --- Sidebar: weights + threshold ---
with st.sidebar:
    st.header("Scoring weights")
    st.caption("Auto-normalized to sum to 1.")
    skills_w = st.slider("Skills", 0.0, 1.0, float(st.session_state.weights["skills"]), 0.05)
    semantic_w = st.slider("Semantic (TF-IDF)", 0.0, 1.0, float(st.session_state.weights["semantic"]), 0.05)
    experience_w = st.slider("Experience (YOE)", 0.0, 1.0, float(st.session_state.weights["experience"]), 0.05)
    education_w = st.slider("Education", 0.0, 1.0, float(st.session_state.weights["education"]), 0.05)
    total = skills_w + semantic_w + experience_w + education_w
    if total <= 0:
        st.error("At least one weight must be > 0.")
        total = 1.0
    st.session_state.weights = {
        "skills": skills_w / total,
        "semantic": semantic_w / total,
        "experience": experience_w / total,
        "education": education_w / total,
    }

    st.divider()
    st.header("Filters")
    st.session_state.threshold = st.slider(
        "Minimum score threshold", 0.0, 1.0, float(st.session_state.threshold), 0.05
    )
    st.session_state.top_n = st.slider(
        "Top N candidates", 1, 100, int(st.session_state.top_n), 1
    )


tab_upload, tab_jd, tab_results = st.tabs(
    ["1. Upload Resumes", "2. Configure Job", "3. Results"]
)


# ---- Tab 1: upload resumes ----
with tab_upload:
    section_header(
        "Upload resumes (PDF or DOCX)",
        "Drop multiple files — they're parsed in your session, never persisted.",
    )
    files = st.file_uploader(
        "Drag and drop or browse",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        key="resume_uploader",
    )
    if st.button("Parse resumes", type="primary", disabled=not files):
        parsed: list = []
        errors: list[tuple[str, str]] = []
        progress = st.progress(0.0, text="Parsing...")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            total_files = len(files or [])
            for i, f in enumerate(files or []):
                progress.progress(i / max(total_files, 1), text=f"Parsing {f.name}...")
                try:
                    path = tmp / f.name
                    path.write_bytes(f.getvalue())
                    resume = parse_resume(path)
                    parsed.append(extract_entities(resume))
                except Exception as exc:
                    errors.append((f.name, str(exc)))
        progress.empty()
        st.session_state.candidates = parsed
        st.session_state.ranked = None
        st.success(f"Parsed {len(parsed)} resume(s).")
        if errors:
            with st.expander(f"{len(errors)} file(s) failed to parse"):
                for name, err in errors:
                    st.error(f"{name}: {err}")

    if st.session_state.candidates:
        section_header(f"Parsed resumes ({len(st.session_state.candidates)})")
        summary = pd.DataFrame(
            [
                {
                    "File": c.source_filename,
                    "Name": c.name or "",
                    "Email": c.email or "",
                    "Phone": c.phone or "",
                    "YOE": c.yoe if c.yoe is not None else None,
                    "Skills found": len(c.skills),
                    "Degree": c.highest_degree_tier or "",
                    "Titles": ", ".join(c.titles[:3]),
                }
                for c in st.session_state.candidates
            ]
        )
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
            column_config={
                "File": st.column_config.TextColumn("File", width="medium"),
                "Name": st.column_config.TextColumn("Name", width="small"),
                "Email": st.column_config.TextColumn("Email"),
                "Phone": st.column_config.TextColumn("Phone"),
                "YOE": st.column_config.NumberColumn("YOE", format="%d yr"),
                "Skills found": st.column_config.NumberColumn(
                    "Skills found",
                    format="%d",
                    help="Distinct skills detected in this resume",
                ),
                "Degree": st.column_config.TextColumn("Degree"),
                "Titles": st.column_config.TextColumn("Titles"),
            },
        )


# ---- Tab 2: configure JD ----
with tab_jd:
    section_header(
        "Job description",
        "Choose any input mode — the ranking uses the same signals in all three.",
    )
    mode_paste, mode_file, mode_form = st.tabs(
        ["Paste text", "Upload file", "Structured form"]
    )

    with mode_paste:
        jd_text = st.text_area(
            "Paste the full job description", height=360, key="jd_paste_text"
        )
        if st.button(
            "Parse JD",
            type="primary",
            key="btn_parse_paste",
            disabled=not jd_text.strip(),
        ):
            st.session_state.jd = parse_jd_from_text(jd_text)
            st.session_state.ranked = None
            st.success("JD parsed.")

    with mode_file:
        jd_file = st.file_uploader(
            "Upload JD (PDF / DOCX / TXT)",
            type=["pdf", "docx", "txt"],
            key="jd_uploader",
        )
        if st.button(
            "Parse uploaded JD",
            type="primary",
            key="btn_parse_file",
            disabled=not jd_file,
        ):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    path = Path(tmpdir) / jd_file.name
                    path.write_bytes(jd_file.getvalue())
                    st.session_state.jd = parse_jd_from_file(path)
                st.session_state.ranked = None
                st.success("JD parsed from file.")
            except Exception as exc:
                st.error(f"Failed to parse JD: {exc}")

    with mode_form:
        col1, col2 = st.columns(2)
        with col1:
            title = st.text_input("Job title", key="f_title")
            required_skills_str = st.text_area(
                "Required skills (comma- or newline-separated)", key="f_req"
            )
            preferred_skills_str = st.text_area(
                "Preferred skills (comma- or newline-separated)", key="f_pref"
            )
        with col2:
            min_yoe = st.number_input(
                "Minimum years of experience", 0, 50, 0, key="f_yoe"
            )
            degree = st.selectbox(
                "Minimum required degree",
                ["", "certification", "associate", "bachelors", "masters", "doctorate"],
                key="f_degree",
            )
            description = st.text_area(
                "Job description (optional — improves semantic match)",
                height=200,
                key="f_desc",
            )

        if st.button("Build JD", type="primary", key="btn_build_form"):
            def _split(s: str) -> list[str]:
                parts = s.replace("\n", ",").split(",")
                return [p.strip() for p in parts if p.strip()]

            st.session_state.jd = parse_jd_from_form(
                title=title or None,
                required_skills=_split(required_skills_str),
                preferred_skills=_split(preferred_skills_str),
                min_yoe=int(min_yoe) if min_yoe else None,
                required_degree=degree or None,
                description=description,
            )
            st.session_state.ranked = None
            st.success("JD built.")

    if st.session_state.jd:
        jd = st.session_state.jd
        section_header(jd.title or "Parsed JD")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Required skills", len(jd.required_skills))
        c2.metric("Preferred skills", len(jd.preferred_skills))
        c3.metric("Min YOE", jd.min_yoe if jd.min_yoe is not None else "—")
        c4.metric("Min degree", jd.required_degree or "—")
        st.write("**Required:** " + (", ".join(jd.required_skills) or "—"))
        st.write("**Preferred:** " + (", ".join(jd.preferred_skills) or "—"))


# ---- Tab 3: results ----
with tab_results:
    section_header(
        "Ranked candidates",
        "Click Rank to score the uploaded resumes against the configured JD.",
    )

    missing = []
    if not st.session_state.candidates:
        missing.append("resumes (tab 1)")
    if st.session_state.jd is None:
        missing.append("JD (tab 2)")

    col_btn, col_msg = st.columns([1, 3])
    with col_btn:
        if st.button(
            "Rank candidates",
            type="primary",
            disabled=bool(missing),
        ):
            w = st.session_state.weights
            weights = ScoringWeights(
                skills=w["skills"],
                semantic=w["semantic"],
                experience=w["experience"],
                education=w["education"],
            )
            st.session_state.ranked = rank_candidates(
                st.session_state.jd,
                st.session_state.candidates,
                weights=weights,
                min_score=st.session_state.threshold,
                top_n=st.session_state.top_n,
            )
            try:
                log_run(
                    st.session_state.jd,
                    st.session_state.ranked,
                    n_candidates_submitted=len(st.session_state.candidates),
                    threshold=st.session_state.threshold,
                    weights=st.session_state.weights,
                )
            except Exception:
                pass  # Logging is best-effort; never fail a ranking run on a log error.
    with col_msg:
        if missing:
            st.info("Provide " + " and ".join(missing) + " first.")

    ranked = st.session_state.ranked
    if ranked is not None:
        if not ranked:
            st.warning("No candidates met the threshold. Lower the threshold or adjust weights.")
        else:
            overall_scores = [r.score.overall for r in ranked]
            top = max(overall_scores)
            avg = sum(overall_scores) / len(overall_scores)

            m1, m2, m3 = st.columns(3)
            m1.metric("Candidates shown", len(ranked))
            m2.metric("Top score", f"{top * 100:.1f}")
            m3.metric("Average score", f"{avg * 100:.1f}")

            df = ranked_to_dataframe(ranked)
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d", width="small"),
                    "Candidate": st.column_config.TextColumn("Candidate", width="medium"),
                    "Overall": st.column_config.ProgressColumn(
                        "Overall",
                        help="Weighted overall score",
                        format="%.1f",
                        min_value=0,
                        max_value=100,
                    ),
                    "Skills %": st.column_config.NumberColumn("Skills", format="%.0f%%"),
                    "Semantic %": st.column_config.NumberColumn("Semantic", format="%.0f%%"),
                    "Experience %": st.column_config.NumberColumn("Experience", format="%.0f%%"),
                    "Education %": st.column_config.NumberColumn("Education", format="%.0f%%"),
                    "YOE": st.column_config.NumberColumn("YOE", format="%d yr"),
                    "Degree": st.column_config.TextColumn("Degree"),
                    "Email": st.column_config.TextColumn("Email"),
                    "Phone": st.column_config.TextColumn("Phone"),
                    # Hide verbose columns in the on-screen table; the candidate
                    # cards below show this info, and CSV/PDF export still includes it.
                    "Filename": None,
                    "Matched Required": None,
                    "Missing Required": None,
                    "Matched Preferred": None,
                },
            )

            section_header(
                "Per-candidate details",
                "Matched / missing / preferred skills for each candidate.",
            )
            for r in ranked:
                render_candidate_card(r)

            section_header("Export")
            jd_title = st.session_state.jd.title if st.session_state.jd else None
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "Download CSV",
                    data=export_ranked_to_csv(ranked),
                    file_name="ranked_candidates.csv",
                    mime="text/csv",
                )
            with c2:
                st.download_button(
                    "Download PDF",
                    data=export_ranked_to_pdf(ranked, jd_title=jd_title),
                    file_name="ranked_candidates.pdf",
                    mime="application/pdf",
                )
