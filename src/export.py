"""CSV and PDF export for ranked candidates."""
from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from src.matching.ranker import RankedCandidate


def ranked_to_dataframe(ranked: list[RankedCandidate]) -> pd.DataFrame:
    """Tabular view of ranked candidates suitable for on-screen and CSV export."""
    rows = []
    for r in ranked:
        rows.append(
            {
                "Rank": r.rank,
                "Candidate": r.entities.name or r.filename,
                "Filename": r.filename,
                "Email": r.entities.email or "",
                "Phone": r.entities.phone or "",
                "Overall": round(r.score.overall * 100, 1),
                "Skills %": round(r.score.skills * 100, 1),
                "Semantic %": round(r.score.semantic * 100, 1),
                "Experience %": round(r.score.experience * 100, 1),
                "Education %": round(r.score.education * 100, 1),
                "YOE": r.entities.yoe if r.entities.yoe is not None else "",
                "Degree": r.entities.highest_degree_tier or "",
                "Matched Required": ", ".join(r.explanation.matched_required),
                "Missing Required": ", ".join(r.explanation.missing_required),
                "Matched Preferred": ", ".join(r.explanation.matched_preferred),
            }
        )
    return pd.DataFrame(rows)


def export_ranked_to_csv(ranked: list[RankedCandidate]) -> bytes:
    return ranked_to_dataframe(ranked).to_csv(index=False).encode("utf-8")


def export_ranked_to_pdf(
    ranked: list[RankedCandidate],
    *,
    jd_title: str | None = None,
) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        leftMargin=0.5 * inch,
        rightMargin=0.5 * inch,
    )
    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=9, leading=11)

    story: list = []
    title = "Resume Screening Results"
    if jd_title:
        title = f"{title} — {jd_title}"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), small))
    story.append(Spacer(1, 0.2 * inch))

    if not ranked:
        story.append(Paragraph("No candidates met the threshold.", styles["Normal"]))
        doc.build(story)
        return buf.getvalue()

    # Summary table
    summary: list[list[str]] = [
        ["#", "Candidate", "Overall", "Skills", "Semantic", "YOE", "Degree"]
    ]
    for r in ranked:
        candidate = r.entities.name or r.filename
        summary.append(
            [
                str(r.rank),
                candidate[:34],
                f"{r.score.overall * 100:.1f}",
                f"{r.score.skills * 100:.0f}%",
                f"{r.score.semantic * 100:.0f}%",
                str(r.entities.yoe or "—"),
                r.entities.highest_degree_tier or "—",
            ]
        )
    table = Table(summary, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2E4057")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F4F7")]),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.3 * inch))

    # Per-candidate details
    story.append(Paragraph("Candidate Details", styles["Heading2"]))
    for r in ranked:
        cand = r.entities.name or r.filename
        story.append(
            Paragraph(
                f"<b>#{r.rank} — {cand}</b> &nbsp; "
                f"<font color='#555'>Score: {r.score.overall * 100:.1f}</font>",
                styles["Heading3"],
            )
        )
        if r.entities.email or r.entities.phone:
            contact = " · ".join(filter(None, [r.entities.email, r.entities.phone]))
            story.append(Paragraph(contact, small))

        matched = ", ".join(r.explanation.matched_required) or "—"
        missing = ", ".join(r.explanation.missing_required) or "—"
        preferred = ", ".join(r.explanation.matched_preferred) or "—"
        story.append(Paragraph(f"<b>Matched required:</b> {matched}", small))
        story.append(Paragraph(f"<b>Missing required:</b> {missing}", small))
        story.append(Paragraph(f"<b>Matched preferred:</b> {preferred}", small))

        yoe_line = f"<b>YOE:</b> {r.entities.yoe or '—'} (required: {r.explanation.yoe_required or '—'})"
        deg_line = (
            f"<b>Degree:</b> {r.entities.highest_degree_tier or '—'} "
            f"(required: {r.explanation.degree_required or '—'})"
        )
        story.append(Paragraph(yoe_line, small))
        story.append(Paragraph(deg_line, small))
        story.append(Spacer(1, 0.15 * inch))

    doc.build(story)
    return buf.getvalue()
