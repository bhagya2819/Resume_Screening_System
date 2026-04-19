"""Centralized CSS + HTML helpers for the Streamlit UI.

Injected once per page load via inject_css(). All helpers that emit HTML
escape user-controlled strings before interpolating them — skill names and
resume content routinely contain characters like <, >, &, ', ".
"""
from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


CSS = """
/* -------- Page chrome -------- */
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }

/* -------- Hero banner -------- */
.hero {
  margin: 0 0 1.4rem 0;
  padding: 1.6rem 1.8rem;
  border-radius: 14px;
  background: linear-gradient(135deg, #2E4057 0%, #4A6FA5 60%, #6C8EBF 100%);
  color: #FFFFFF;
  box-shadow: 0 6px 18px rgba(46, 64, 87, 0.18);
}
.hero h1 { margin: 0 0 0.35rem 0; font-size: 1.9rem; font-weight: 700; color: #FFFFFF; }
.hero p  { margin: 0; font-size: 1rem; opacity: 0.92; color: #E8EEF7; }

/* -------- Sidebar tweaks -------- */
section[data-testid="stSidebar"] { background-color: #F7F9FC; }
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
section[data-testid="stSidebar"] h2 {
  font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.04em;
  color: #2E4057; margin-top: 0.8rem; margin-bottom: 0.3rem;
}
section[data-testid="stSidebar"] hr {
  margin: 0.9rem 0; border: none; border-top: 1px solid #D7DEE8;
}

/* -------- File uploader drop zone -------- */
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed #4A6FA5;
  background: #F5F8FC;
  border-radius: 12px;
  padding: 1.2rem;
  transition: background 0.15s ease-in;
}
[data-testid="stFileUploaderDropzone"]:hover { background: #ECF2FA; }

/* -------- Section header (colored-header drop-in) -------- */
.section-header {
  border-top: 3px solid #2E4057;
  padding: 0.8rem 0 0.2rem 0;
  margin: 1rem 0 0.3rem 0;
}
.section-header h3 {
  margin: 0;
  font-size: 1.2rem;
  font-weight: 700;
  color: #2E4057;
}
.section-header p {
  margin: 0.15rem 0 0 0;
  font-size: 0.85rem;
  color: #6B7280;
}

/* -------- JD summary card accent -------- */
.jd-card {
  border-left: 4px solid #2E4057;
  background: #FFFFFF;
  padding: 0.9rem 1.1rem;
  border-radius: 8px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.06);
  margin: 0.6rem 0 1rem 0;
}

/* -------- Metric cards (applied to st.metric globally) -------- */
[data-testid="stMetric"] {
  background: #FFFFFF;
  border: 1px solid #E4E8EF;
  border-left: 4px solid #2E4057;
  border-radius: 8px;
  padding: 0.8rem 1rem;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
[data-testid="stMetricLabel"] p {
  color: #6B7280;
  font-size: 0.82rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
  color: #1F2937;
  font-weight: 700;
}

/* -------- Candidate card -------- */
.candidate-card {
  border: 1px solid #E4E8EF;
  background: #FFFFFF;
  border-radius: 12px;
  padding: 1.1rem 1.3rem;
  margin: 0.8rem 0;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.candidate-card .cc-head {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 0.7rem;
}
.candidate-card .cc-title { font-size: 1.1rem; font-weight: 600; color: #1F2937; }
.candidate-card .cc-rank {
  display: inline-block; background: #2E4057; color: #FFFFFF;
  font-weight: 700; font-size: 0.8rem;
  padding: 0.15rem 0.55rem; border-radius: 999px; margin-right: 0.55rem;
}
.candidate-card .cc-sub  { font-size: 0.85rem; color: #6B7280; margin-top: 0.15rem; }

.score-badge {
  display: inline-block; padding: 0.35rem 0.85rem; border-radius: 8px;
  background: #EEF5FF; color: #1F4C8A; font-weight: 700; font-size: 1rem;
  border: 1px solid #CFE0F5;
}
.score-badge.high { background: #E7F6EC; color: #186A3B; border-color: #BFE5CA; }
.score-badge.mid  { background: #FFF4D6; color: #7A5900; border-color: #F1DDA0; }
.score-badge.low  { background: #FDECEC; color: #8A1F1F; border-color: #F2C5C5; }

.cc-subscores { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 0.3rem 0 0.8rem 0; }
.cc-subscores .chip {
  background: #F2F4F7; color: #374151; border: 1px solid #E4E8EF;
  padding: 0.22rem 0.6rem; border-radius: 999px; font-size: 0.78rem;
}

/* -------- Pills -------- */
.pill {
  display: inline-block; padding: 0.2rem 0.55rem; margin: 0.15rem 0.25rem 0.15rem 0;
  border-radius: 999px; font-size: 0.78rem; font-weight: 500;
  border: 1px solid transparent; line-height: 1.2;
}
.pill-matched   { background: #E7F6EC; color: #186A3B; border-color: #BFE5CA; }
.pill-missing   { background: #FDECEC; color: #8A1F1F; border-color: #F2C5C5;
                  text-decoration: line-through; }
.pill-preferred { background: #EEF5FF; color: #1F4C8A; border-color: #CFE0F5; }
.pill-extra     { background: #F2F4F7; color: #374151; border-color: #E4E8EF; }

.pill-row-label {
  font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: #6B7280; margin: 0.55rem 0 0.25rem 0; font-weight: 600;
}
"""


def inject_css() -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def _esc(value: str | None) -> str:
    if value is None or value == "":
        return "—"
    return html.escape(str(value), quote=True)


def render_hero(title: str, tagline: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{_esc(title)}</h1><p>{_esc(tagline)}</p></div>',
        unsafe_allow_html=True,
    )


def section_header(label: str, description: str = "") -> None:
    desc_html = f"<p>{_esc(description)}</p>" if description else ""
    st.markdown(
        f'<div class="section-header"><h3>{_esc(label)}</h3>{desc_html}</div>',
        unsafe_allow_html=True,
    )


def _pill(text: str, kind: str) -> str:
    return f'<span class="pill {kind}">{html.escape(text, quote=True)}</span>'


def _pills_or_dash(items: Iterable[str], kind: str) -> str:
    items = list(items)
    if not items:
        return '<span class="cc-sub">—</span>'
    return "".join(_pill(s, kind) for s in items)


def _score_tier(overall_0_to_1: float) -> str:
    pct = overall_0_to_1 * 100
    if pct >= 75:
        return "high"
    if pct >= 50:
        return "mid"
    return "low"


def render_candidate_card(r) -> None:
    """Render one RankedCandidate as a styled HTML card."""
    e = r.entities
    exp = r.explanation
    tier = _score_tier(r.score.overall)
    name = e.name or r.filename

    yoe_txt = str(e.yoe) if e.yoe is not None else "—"
    yoe_req_txt = str(exp.yoe_required) if exp.yoe_required is not None else "—"
    deg_txt = e.highest_degree_tier or "—"
    deg_req_txt = exp.degree_required or "—"

    html_str = (
        '<div class="candidate-card">'
        '<div class="cc-head">'
        '<div>'
        f'<span class="cc-rank">#{r.rank}</span>'
        f'<span class="cc-title">{_esc(name)}</span>'
        f'<div class="cc-sub">{_esc(e.email)} &middot; {_esc(e.phone)} &middot; {_esc(r.filename)}</div>'
        '</div>'
        f'<div class="score-badge {tier}">{r.score.overall * 100:.1f}</div>'
        '</div>'
        '<div class="cc-subscores">'
        f'<span class="chip">Skills {r.score.skills * 100:.0f}%</span>'
        f'<span class="chip">Semantic {r.score.semantic * 100:.0f}%</span>'
        f'<span class="chip">Experience {r.score.experience * 100:.0f}%</span>'
        f'<span class="chip">Education {r.score.education * 100:.0f}%</span>'
        f'<span class="chip">YOE {_esc(yoe_txt)} / req {_esc(yoe_req_txt)}</span>'
        f'<span class="chip">Degree {_esc(deg_txt)} / req {_esc(deg_req_txt)}</span>'
        '</div>'
        '<div class="pill-row-label">Matched required</div>'
        f'<div>{_pills_or_dash(exp.matched_required, "pill-matched")}</div>'
        '<div class="pill-row-label">Missing required</div>'
        f'<div>{_pills_or_dash(exp.missing_required, "pill-missing")}</div>'
        '<div class="pill-row-label">Matched preferred</div>'
        f'<div>{_pills_or_dash(exp.matched_preferred, "pill-preferred")}</div>'
        '</div>'
    )
    st.markdown(html_str, unsafe_allow_html=True)
