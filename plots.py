# plots.py
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_power_timeseries(power_W: pd.Series) -> go.Figure:
    fig = px.line(power_W, labels={"value": "Power [W]", "index": "Time"}, title="Power Output")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_speed_vs_power_hist(power_W: pd.Series, hub_speed: pd.Series) -> go.Figure:
    df = pd.DataFrame({"Power [kW]": power_W / 1e3, "Wind @ hub [m/s]": hub_speed})
    fig = px.density_heatmap(df, x="Wind @ hub [m/s]", y="Power [kW]", nbinsx=24, nbinsy=24,
                             title="Distribution: Hub Wind vs Power")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_diurnal_profile(power_W: pd.Series) -> go.Figure:
    g = power_W.groupby(power_W.index.hour).mean() / 1000.0  # kW
    fig = px.bar(g, labels={"value": "Avg Power [kW]", "index": "Hour"}, title="Diurnal Profile")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def plot_monthly_climatology(series: pd.Series, name: str, unit: str) -> go.Figure:
    m = series.groupby(series.index.month).mean()
    fig = px.bar(m, labels={"value": f"Avg {name} [{unit}]", "index": "Month"},
                 title=f"Monthly Climatology — {name}")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    return fig


def wind_rose(direction_deg: pd.Series, hub_speed: pd.Series, bins: int = 16) -> go.Figure:
    """
    Simple wind rose using polar bar chart.
    """
    # Sector edges 0..360
    edges = np.linspace(0, 360, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    cat = pd.cut(direction_deg % 360, bins=edges, right=False, include_lowest=True)
    sp = hub_speed.groupby(cat).mean()
    # Convert to arrays aligned with centers
    values = np.zeros_like(centers)
    for i, c in enumerate(sp.index.categories):
        values[i] = sp.loc.get(c, 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Barpolar(theta=centers, r=values, name="Mean speed", marker_line_width=1)
    )
    fig.update_layout(
        title="Wind Rose (mean hub wind per sector)",
        polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return fig
