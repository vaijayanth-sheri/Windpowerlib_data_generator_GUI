# qc_bias.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class QCReport:
    n_rows: int
    coverage_pct: float
    gaps_pct_by_month: pd.Series
    ranges_ok: bool
    issues: list


def range_checks(df: pd.DataFrame) -> Tuple[bool, list]:
    """
    Basic sanity checks for normalized weather columns (MultiIndex).
    """
    issues = []

    def check(series: Optional[pd.Series], lo: float, hi: float, name: str):
        if series is None:
            return
        bad = (~series.isna()) & ((series < lo) | (series > hi))
        if bad.any():
            n = int(bad.sum())
            issues.append(f"{name}: {n} values outside [{lo},{hi}]")

    col = df.columns
    ws10 = df[col.get_loc(("wind_speed", 10))] if ("wind_speed", 10) in col else None
    ws50 = df[col.get_loc(("wind_speed", 50))] if ("wind_speed", 50) in col else None
    ws100 = df[col.get_loc(("wind_speed", 100))] if ("wind_speed", 100) in col else None
    t2 = df[col.get_loc(("temperature", 2))] if ("temperature", 2) in col else None
    t10 = df[col.get_loc(("temperature", 10))] if ("temperature", 10) in col else None
    p0 = df[col.get_loc(("pressure", 0))] if ("pressure", 0) in col else None

    check(ws10, 0.0, 60.0, "wind_speed@10m")
    check(ws50, 0.0, 60.0, "wind_speed@50m")
    check(ws100, 0.0, 60.0, "wind_speed@100m")
    check(t2, 190.0, 330.0, "temperature@2m[K]")
    check(t10, 190.0, 330.0, "temperature@10m[K]")
    check(p0, 50_000.0, 110_000.0, "pressure@0[Pa]")

    return (len(issues) == 0), issues


def gap_summary(df: pd.DataFrame) -> pd.Series:
    """
    Share of missing rows (any required variable missing) per calendar month.
    """
    required = [c for c in df.columns if c[0] in ("wind_speed", "temperature", "pressure")]
    sub = df[required]
    row_missing = sub.isna().any(axis=1)
    by_month = row_missing.groupby([sub.index.year, sub.index.month]).mean().rename("gap_pct")
    by_month.index = pd.MultiIndex.from_tuples(by_month.index, names=["year", "month"])
    return by_month * 100.0


def qc_report(df: pd.DataFrame) -> QCReport:
    n_rows = len(df)
    gaps_pct_by_month = gap_summary(df)
    coverage_pct = 100.0 - gaps_pct_by_month.mean() if not gaps_pct_by_month.empty else 100.0
    ranges_ok, issues = range_checks(df)
    return QCReport(
        n_rows=n_rows,
        coverage_pct=float(coverage_pct) if coverage_pct == coverage_pct else 0.0,
        gaps_pct_by_month=gaps_pct_by_month,
        ranges_ok=ranges_ok,
        issues=issues,
    )


# ---------- Bias correction ----------
def monthly_mean_variance_bias_correct(
    driver: pd.Series,
    reference: pd.Series,
) -> pd.Series:
    """
    Simple per-month mean/variance scaling of the driver (e.g., wind speed).
    Assumes both series aligned to the same timestamps; drops NA pairs.
    """
    df = pd.concat({"drv": driver, "ref": reference}, axis=1).dropna()
    if df.empty:
        return driver

    def adjust(group):
        mu_d, sigma_d = group["drv"].mean(), group["drv"].std(ddof=0)
        mu_r, sigma_r = group["ref"].mean(), group["ref"].std(ddof=0)
        if np.isfinite(mu_d) and np.isfinite(sigma_d) and sigma_d > 1e-9:
            scaled = (group["drv"] - mu_d) * (sigma_r / max(sigma_d, 1e-9)) + mu_r
        else:
            scaled = group["drv"] * 0.0 + mu_r
        return scaled

    adj = df.groupby([df.index.month]).apply(adjust)
    # Place corrected values back on the original driver index where available
    out = driver.copy()
    out.loc[adj.index.get_level_values(1)] = adj.values
    return out


def quantile_mapping(
    driver: pd.Series, reference: pd.Series, quantiles: int = 20
) -> pd.Series:
    """
    Quantile mapping bias correction.
    """
    df = pd.concat({"drv": driver, "ref": reference}, axis=1).dropna()
    if df.empty:
        return driver

    qs = np.linspace(0.0, 1.0, quantiles + 1)
    drv_q = df["drv"].quantile(qs)
    ref_q = df["ref"].quantile(qs)

    def map_value(x):
        # Find the quantile bin
        idx = np.searchsorted(drv_q.values, x, side="right") - 1
        idx = np.clip(idx, 0, len(ref_q.values) - 2)
        # Linear interpolation within the bin
        x0, x1 = drv_q.values[idx], drv_q.values[idx + 1]
        y0, y1 = ref_q.values[idx], ref_q.values[idx + 1]
        if x1 - x0 < 1e-9:
            return y0
        w = (x - x0) / (x1 - x0)
        return y0 * (1 - w) + y1 * w

    return driver.apply(map_value)
