# export.py
from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


@dataclass
class ExportBundle:
    """
    A simple container for export artifacts so the Streamlit layer can download them.
    """
    power_csv: bytes
    weather_csv: bytes
    metadata_json: bytes
    pdf_bytes: bytes


def flatten_weather_columns(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MultiIndex columns (var, height) -> 'var_height' flat columns for CSV export.
    """
    flat = weather_df.copy()
    flat.columns = [f"{v}_{h}" for (v, h) in flat.columns]
    return flat


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=True).encode("utf-8")


def build_metadata_json(inputs: Dict) -> bytes:
    import json
    return json.dumps(inputs, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def make_pdf_summary(
    kpis: Dict[str, str],
    site: Dict[str, str],
    turbine: Dict[str, str],
    notes: Optional[str] = None,
) -> bytes:
    """
    Minimal, robust PDF generator using reportlab (no heavy template deps).
    """
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    width, height = A4

    def text(x, y, s, size=11):
        c.setFont("Helvetica", size)
        c.drawString(x, y, s)

    y = height - 2 * cm
    text(2 * cm, y, "Windpowerlib Dashboard — Summary", 14)
    y -= 1.2 * cm

    text(2 * cm, y, "Site"); y -= 0.6 * cm
    for k, v in site.items():
        text(2.5 * cm, y, f"{k}: {v}"); y -= 0.5 * cm

    y -= 0.4 * cm
    text(2 * cm, y, "Turbine"); y -= 0.6 * cm
    for k, v in turbine.items():
        text(2.5 * cm, y, f"{k}: {v}"); y -= 0.5 * cm

    y -= 0.4 * cm
    text(2 * cm, y, "KPIs"); y -= 0.6 * cm
    for k, v in kpis.items():
        text(2.5 * cm, y, f"{k}: {v}"); y -= 0.5 * cm

    if notes:
        y -= 0.6 * cm
        text(2 * cm, y, "Notes"); y -= 0.6 * cm
        for line in notes.splitlines():
            text(2.5 * cm, y, line); y -= 0.45 * cm

    c.showPage()
    c.save()
    return buf.getvalue()


def build_export_bundle(
    power_W: pd.Series,
    weather_df: pd.DataFrame,
    metadata: Dict,
    kpis: Dict[str, str],
    site: Dict[str, str],
    turbine_meta: Dict[str, str],
    notes: Optional[str] = None,
) -> ExportBundle:
    power_csv = to_csv_bytes(power_W.to_frame("power_W"))
    weather_csv = to_csv_bytes(flatten_weather_columns(weather_df))
    metadata_json = build_metadata_json(metadata)
    pdf_bytes = make_pdf_summary(kpis, site, turbine_meta, notes=notes)
    return ExportBundle(
        power_csv=power_csv,
        weather_csv=weather_csv,
        metadata_json=metadata_json,
        pdf_bytes=pdf_bytes,
    )
