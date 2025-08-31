# report_wind.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import io
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak
)
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

TOOL_NAME = "Windpowerlib Dashboard"
AUTHOR_NAME = "Vaijayanth Sheri"

def _fig_png(fig) -> bytes:
    bio = io.BytesIO()
    fig.savefig(bio, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    bio.seek(0)
    return bio.getvalue()

def _escape(s: str) -> str:
    return str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ---------- Plots ----------
def plot_monthly_energy(power_W: pd.Series) -> bytes | None:
    if power_W is None or power_W.empty:
        return None
    monthly_kwh = (power_W.resample("M").sum() / 1e3)
    months = monthly_kwh.index.strftime("%b")
    fig = plt.figure(figsize=(6.4, 3.0))
    ax = fig.add_subplot(111)
    ax.bar(range(len(monthly_kwh)), monthly_kwh.values)
    ax.set_xticks(range(len(monthly_kwh)))
    ax.set_xticklabels(months)
    ax.set_ylabel("kWh")
    ax.set_title("Monthly Energy")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    return _fig_png(fig)

def plot_first_week(power_W: pd.Series) -> bytes | None:
    if power_W is None or power_W.empty:
        return None
    i0 = power_W.index.min()
    s = power_W.loc[i0:i0 + pd.Timedelta(days=7)] / 1e3
    fig = plt.figure(figsize=(6.4, 3.0))
    ax = fig.add_subplot(111)
    ax.plot(s.index, s.values)
    ax.set_ylabel("Power [kW]")
    ax.set_title("Sample Time Series (First Week)")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    return _fig_png(fig)

def plot_wind_rose(direction_deg: pd.Series, hub_speed: pd.Series, bins: int = 16) -> bytes | None:
    if direction_deg is None or hub_speed is None or hub_speed.empty:
        return None
    df = pd.concat({"dir": direction_deg, "spd": hub_speed}, axis=1).dropna()
    if df.empty:
        return None
    edges = np.linspace(0, 360, bins + 1)
    cat = pd.cut(df["dir"] % 360, bins=edges, right=False, include_lowest=True)
    sp = df["spd"].groupby(cat).mean()
    centers = (edges[:-1] + edges[1:]) / 2.0
    values = np.zeros_like(centers)
    for i, c in enumerate(sp.index.categories):
        if c in sp.index:
            values[i] = sp.loc[c]
    fig = plt.figure(figsize=(6.0, 6.0))
    ax = fig.add_subplot(111, projection="polar")
    theta = np.deg2rad(centers)
    ax.bar(theta, values, width=np.deg2rad(360 / bins), edgecolor="k", linewidth=0.5)
    ax.set_title("Wind Rose (mean hub wind per sector)")
    return _fig_png(fig)

# ---------- KPIs ----------
def compute_kpis(power_W: pd.Series, rated_power_W: Optional[float]) -> Dict[str, str]:
    """
    Annual energy and capacity factor (no availability).
    """
    if power_W is None or power_W.empty:
        return {"annual_kwh": "-", "capacity_factor": "-"}
    annual_kwh = float(power_W.sum() / 1e3)
    cf = "-"
    if rated_power_W and rated_power_W > 0:
        hours = len(power_W)
        cf_val = annual_kwh * 1e3 / (rated_power_W * hours)  # kWh -> Wh
        cf = f"{cf_val:.2%}"
    return {"annual_kwh": f"{annual_kwh:,.0f}", "capacity_factor": cf}

# ---------- Footer ----------
def _footer(canv: canvas.Canvas, doc):
    canv.saveState()
    footer_text = f"{TOOL_NAME}/{AUTHOR_NAME} — Page {doc.page}"
    canv.setFont("Helvetica", 8)
    canv.setFillColor(colors.grey)
    canv.drawString(36, 20, footer_text)
    canv.restoreState()

# ---------- Main writer ----------
def write_pdf(
    outpath: Path,
    *,
    site_info: Dict,          # {"lat":..., "lon":..., "addr":..., "period":...}
    weather_source: Dict,     # {"name":..., "details": {...}}
    turbine_cfg: Dict,        # {"name":..., "hub_height":..., "rated_power_W":...}
    model_cfg: Dict,          # ModelChain options selected
    losses_cfg: Dict,         # kept for appendix only
    power_W: pd.Series,
    hub_speed: Optional[pd.Series] = None,
    hub_density: Optional[pd.Series] = None,
    hub_temperature_K: Optional[pd.Series] = None,
    wind_dir_deg: Optional[pd.Series] = None,
) -> Path:
    # KPIs (no availability)
    rated = turbine_cfg.get("rated_power_W")
    kpis = compute_kpis(power_W, rated_power_W=rated)

    # Figures (no power-duration curve)
    monthly_png = plot_monthly_energy(power_W)
    week_png = plot_first_week(power_W)
    rose_png = plot_wind_rose(wind_dir_deg, hub_speed) if wind_dir_deg is not None else None

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Heading1C", parent=styles["Heading1"], alignment=1))
    styles.add(ParagraphStyle(name="Heading2S", parent=styles["Heading2"], spaceBefore=12, spaceAfter=6))
    styles.add(ParagraphStyle(name="Cell", fontName="Helvetica", fontSize=9, leading=11, wordWrap="CJK"))
    styles.add(ParagraphStyle(name="Mono", fontName="Courier", fontSize=8, leading=10))

    doc = SimpleDocTemplate(str(outpath), pagesize=A4,
                            rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    story = []

    # Title
    story.append(Spacer(1, 100))
    story.append(Paragraph(f"<b>{_escape(TOOL_NAME)}</b>", styles["Heading1C"]))
    story.append(Spacer(1, 18))
    story.append(Paragraph("Wind Simulation Report", styles["Heading2S"]))
    story.append(Spacer(1, 24))
    story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}", styles["Normal"]))
    if "addr" in site_info:
        story.append(Paragraph(f"Site: {_escape(site_info['addr'])}", styles["Normal"]))
    story.append(PageBreak())

    # Summary KPIs
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Spacer(1, 6))
    kpi_rows = [
        ["Annual Energy (kWh)", kpis.get("annual_kwh", "-")],
        ["Capacity Factor",     kpis.get("capacity_factor", "-")],
    ]
    kpi_tbl = Table([["Metric", "Value"]] + kpi_rows, colWidths=[220, 180], hAlign="LEFT")
    kpi_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
    ]))
    story.append(kpi_tbl)
    story.append(Spacer(1, 18))

    # Inputs
    story.append(Paragraph("Inputs", styles["Heading1"]))
    def dict_to_rows(d: Dict) -> list[list]:
        return [[Paragraph(_escape(str(k)), styles["Cell"]), Paragraph(_escape(str(d[k])), styles["Cell"])]
                for k in d.keys()]

    site_rows = dict_to_rows({
        "Latitude": f"{site_info.get('lat')}",
        "Longitude": f"{site_info.get('lon')}",
        "Period": f"{site_info.get('period', '')}",
    })
    ws_details = weather_source.get("details") if isinstance(weather_source, dict) else {}
    weather_rows = dict_to_rows({
        "Source": weather_source.get("name", "Weather"),
        "Notes": json.dumps(ws_details, ensure_ascii=False)[:300] + ("..." if len(json.dumps(ws_details)) > 300 else "")
    })
    turb_rows = dict_to_rows({k: str(v) for k, v in turbine_cfg.items()})
    model_rows = dict_to_rows({k: str(v) for k, v in model_cfg.items()})
    loss_rows  = dict_to_rows({k: str(v) for k, v in losses_cfg.items()})

    def block_table(title: str, rows: list[list]):
        story.append(Paragraph(title, styles["Heading2S"]))
        tbl = Table([["Parameter", "Value"]] + rows, colWidths=[220, 523-220], hAlign="LEFT")
        tbl.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
            ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
            ("VALIGN",(0,0),(-1,-1), "TOP"),
            ("FONTNAME",(0,0),(-1,0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.whitesmoke, colors.white]),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 12))

    block_table("Site", site_rows)
    block_table("Weather source", weather_rows)
    block_table("Turbine", turb_rows)
    block_table("ModelChain options", model_rows)
    block_table("Losses & Notes", loss_rows)
    story.append(PageBreak())

    # Charts (no PDC)
    story.append(Paragraph("Charts", styles["Heading1"]))
    if monthly_png:
        story.append(RLImage(io.BytesIO(monthly_png), width=480, height=260))
        story.append(Spacer(1, 12))
    if week_png:
        story.append(RLImage(io.BytesIO(week_png), width=480, height=260))
        story.append(Spacer(1, 12))
    if rose_png:
        story.append(RLImage(io.BytesIO(rose_png), width=360, height=360))
        story.append(Spacer(1, 12))
    story.append(PageBreak())

    # Appendix: JSON configs
    story.append(Paragraph("Appendix: Configuration JSON", styles["Heading1"]))
    cfg_blob = {
        "site": site_info,
        "weather_source": weather_source,
        "turbine_cfg": turbine_cfg,
        "model_cfg": model_cfg,
        "losses_cfg": losses_cfg,
    }
    cfg_json = json.dumps(cfg_blob, indent=2, default=str)
    story.append(Paragraph(_escape(cfg_json).replace("\n", "<br/>"), styles["Mono"]))

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return outpath

def build_report_bytes(**kwargs) -> bytes:
    path = Path("wind_report.pdf")
    write_pdf(path, **kwargs)
    with open(path, "rb") as f:
        return f.read()
