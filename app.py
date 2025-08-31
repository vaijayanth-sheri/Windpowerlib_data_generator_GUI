# app.py
# Streamlit dashboard focused on windpowerlib
# Requires: streamlit, pandas, numpy, requests, geopy, windpowerlib, pvlib
from __future__ import annotations

import io
import json
import calendar
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# Local adapters
from datasources_open_meteo import fetch_open_meteo       # Open-Meteo archive API (10m/100m wind, temp, pressure). :contentReference[oaicite:4]{index=4}
from datasources_power import fetch_power_hourly          # NASA POWER hourly (RE community). :contentReference[oaicite:5]{index=5}
from datasources_pvgis import fetch_pvgis_tmy             # pvlib-backed PVGIS TMY (data, meta). :contentReference[oaicite:6]{index=6}
from user_upload import parse_epw                         # pvlib-backed EPW parser
from report_wind import build_report_bytes                # Report generator

from windpowerlib import ModelChain, WindTurbine
from windpowerlib import data as wt  # OEDB turbine list

from normalize import (
    assemble_weather_df,
    ensure_two_temps_if_needed,
    to_timezone_and_hourly,
)
from physics import uv_to_speed_dir  # (kept if you later add u/v to speed/dir)
def _get_version(pkg: str) -> str:
    try:
        from importlib.metadata import version, PackageNotFoundError
        return version(pkg)
    except Exception:
        try:
            # fallback for older Pythons
            import pkg_resources
            return pkg_resources.get_distribution(pkg).version
        except Exception:
            return "n/a"
# --- Geocoding helpers ---
@st.cache_data(ttl=24*3600)
def geocode_once(query: str):
    geolocator = Nominatim(user_agent="windpowerlib-dashboard")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0, swallow_exceptions=True)
    loc = geocode(query, addressdetails=True, language="en")
    if not loc:
        return None
    return {"lat": float(loc.latitude), "lon": float(loc.longitude), "display_name": loc.address}

st.set_page_config(page_title="Windpowerlib Dashboard", layout="wide")

# ---- Sidebar: Global inputs ----
st.sidebar.header("Run configuration")
current_year = datetime.now(timezone.utc).year
year = st.sidebar.number_input("Simulation year (UTC)", min_value=1979, max_value=current_year,
                               value=current_year - 1, step=1)

# Full-year UTC range
start = datetime(int(year), 1, 1, 0, 0, tzinfo=timezone.utc)
end   = datetime(int(year), 12, 31, 23, 0, tzinfo=timezone.utc)
is_leap = calendar.isleap(int(year))
hours_in_year = 8784 if is_leap else 8760
st.sidebar.caption(f"Range: {start.isoformat()} → {end.isoformat()}  ({hours_in_year} hours{' — leap year' if is_leap else ''})")

# ---- Location selection ----
st.sidebar.subheader("Location")
q = st.sidebar.text_input("Search place (city / postcode / address)", value="", placeholder="e.g., Berlin or 10117 or 'Munich, Germany'")
search_col1, search_col2 = st.sidebar.columns([1,1])
do_search = search_col1.button("Search")
clear_loc = search_col2.button("Clear")
if clear_loc:
    st.session_state.pop("geo_result", None)
if do_search and q.strip():
    res = geocode_once(q.strip())
    if res is None:
        st.sidebar.error("No result found. Try a more specific query.")
    else:
        st.session_state["geo_result"] = res
if "geo_result" in st.session_state:
    st.sidebar.success(st.session_state["geo_result"]["display_name"])

col_sb1, col_sb2 = st.sidebar.columns(2)
lat = col_sb1.number_input("Latitude", value=float(st.session_state.get("geo_result", {}).get("lat", 52.52)), format="%.6f")
lon = col_sb2.number_input("Longitude", value=float(st.session_state.get("geo_result", {}).get("lon", 13.405)), format="%.6f")

# Surface roughness fallback
z0_default = st.sidebar.number_input("Surface roughness z0 (m) if not provided by data", value=0.05, min_value=0.0, step=0.01)

# ModelChain choices
st.sidebar.subheader("ModelChain options")
wind_speed_model = st.sidebar.selectbox("Wind speed model", ["logarithmic", "hellman", "interpolation_extrapolation"], index=0)
density_model = st.sidebar.selectbox("Air density model", ["barometric", "ideal_gas", "interpolation_extrapolation"], index=1)
temperature_model = st.sidebar.selectbox("Temperature model", ["linear_gradient", "interpolation_extrapolation"], index=0)
density_correction = st.sidebar.checkbox("Density correction", value=True)
obstacle_height = st.sidebar.number_input("Obstacle height (m)", value=0.0, min_value=0.0, step=1.0)
hellman_exp = st.sidebar.number_input("Hellman exponent α (if used)", value=0.14, min_value=0.0, step=0.01)

st.title("Windpowerlib Dashboard")
st.caption("Data sources follow provider units, then normalized to windpowerlib MultiIndex weather format (variables × heights): wind_speed [m/s], temperature [K], pressure [Pa], roughness_length [m].")

# ---- Tabs ----
tab_data, tab_turbine, tab_run, tab_results, tab_about = st.tabs(
    ["1) Weather data", "2) Turbine", "3) Run", "4) Results", "5) About"]
)

# ---- Tab 1: Weather data ----
with tab_data:
    st.subheader("Choose source")
    src = st.selectbox(
        "Weather source",
        ["Open-Meteo (archive)", "NASA POWER (hourly)", "PVGIS TMY (typical year)", "User upload (EPW)"],
        index=0,
    )

    weather_wpl = None

    if src == "Open-Meteo (archive)":
        st.markdown("Open-Meteo historical API: wind at 10 m & 100 m, T2M, surface pressure (UTC).")
        if st.button("Fetch Open-Meteo"):
            df = fetch_open_meteo(lat, lon, start, end)
            weather_wpl = assemble_weather_df(
                idx=df.index,
                wind10=df.get("ws10"),
                wind100=df.get("ws100"),
                temp2=df.get("t2m_k"),
                press=df.get("sp_pa"),
                z0_series=df.get("z0"),
            )
            weather_wpl = ensure_two_temps_if_needed(weather_wpl)
            weather_wpl = to_timezone_and_hourly(weather_wpl, tz="UTC")
            if ("roughness_length", 0) not in weather_wpl.columns:
                weather_wpl[("roughness_length", 0)] = z0_default
            st.session_state["weather_wpl"] = weather_wpl
            st.session_state["weather_source_meta"] = {"name": "Open-Meteo (archive)", "details": {"lat": lat, "lon": lon, "rows": int(len(weather_wpl))}}
            st.session_state["period_label"] = f"{start.date()} to {end.date()}"
            st.success(f"Fetched {len(weather_wpl)} hours from Open-Meteo.")  # Open-Meteo archive docs. :contentReference[oaicite:7]{index=7}

    elif src == "NASA POWER (hourly)":
        st.markdown("NASA POWER hourly (RE community): WS10M/WS50M, T2M [°C], PS [kPa], Z0M [m] (UTC).")
        if st.button("Fetch NASA POWER"):
            df = fetch_power_hourly(lat, lon, start, end, community="re")
            weather_wpl = assemble_weather_df(
                idx=df.index,
                wind10=df.get("WS10M"),
                wind50=df.get("WS50M"),
                temp2=df.get("T2M_K"),
                press=df.get("PS_PA"),
                z0_series=df.get("Z0M"),
            )
            weather_wpl = ensure_two_temps_if_needed(weather_wpl)
            weather_wpl = to_timezone_and_hourly(weather_wpl, tz="UTC")
            if ("roughness_length", 0) not in weather_wpl.columns:
                weather_wpl[("roughness_length", 0)] = z0_default
            st.session_state["weather_wpl"] = weather_wpl
            st.session_state["weather_source_meta"] = {"name": "NASA POWER (hourly, RE)", "details": {"lat": lat, "lon": lon, "rows": int(len(weather_wpl))}}
            st.session_state["period_label"] = f"{start.date()} to {end.date()}"
            st.success(f"Fetched {len(weather_wpl)} hours from NASA POWER (RE).")  # POWER hourly API. :contentReference[oaicite:8]{index=8}

    elif src == "PVGIS TMY (typical year)":
        st.markdown("PVGIS TMY: 8760-hour typical year near the site (via pvlib).")
        if st.button("Fetch PVGIS TMY"):
            df, meta = fetch_pvgis_tmy(lat, lon)  # returns (data, meta) on pvlib >= 0.13. :contentReference[oaicite:9]{index=9}
            # Accept pvlib-mapped 'wind_speed' or original 'WS10m'
            ws = df.get("WS10m") if "WS10m" in df.columns else df.get("wind_speed")
            tK = df.get("T2m_K")
            sp = df.get("SP_Pa")
            if ((ws is None or ws.dropna().empty) and (tK is None or tK.dropna().empty) and (sp is None or sp.dropna().empty)):
                st.error("PVGIS returned no usable wind/temperature/pressure fields. Try another site or source.")
            else:
                weather_wpl = assemble_weather_df(idx=df.index, wind10=ws, temp2=tK, press=sp)
                weather_wpl = ensure_two_temps_if_needed(weather_wpl)
                weather_wpl = to_timezone_and_hourly(weather_wpl, tz="UTC")
                if ("roughness_length", 0) not in weather_wpl.columns:
                    weather_wpl[("roughness_length", 0)] = z0_default
                st.session_state["weather_wpl"] = weather_wpl
                st.session_state["weather_source_meta"] = {"name": "PVGIS TMY (via pvlib)", "details": {"rows": int(len(weather_wpl)), "meta_excerpt": str(meta)[:500]}}
                st.session_state["period_label"] = "Typical Meteorological Year (8760 h)"
                st.success(f"Loaded {len(weather_wpl)} TMY hours from PVGIS.")

    elif src == "User upload (EPW)":
        st.markdown("Upload an **EnergyPlus EPW** file. We'll parse wind speed (10 m), air temperature, and station pressure.")
        upl = st.file_uploader("Upload EPW", type=["epw"])
        if upl is not None and st.button("Parse & normalize"):
            df = parse_epw(upl)  # pvlib-backed EPW reader
            ws = df.get("WS10m"); tK = df.get("T2m_K"); sp = df.get("SP_Pa")
            if ((ws is None or ws.dropna().empty) and (tK is None or tK.dropna().empty) and (sp is None or sp.dropna().empty)):
                st.error("No usable columns detected in EPW (wind_speed / temp_air / atmospheric_pressure). Try another EPW file.")
            else:
                weather_wpl = assemble_weather_df(idx=df.index, wind10=ws, temp2=tK, press=sp)
                weather_wpl = ensure_two_temps_if_needed(weather_wpl)
                weather_wpl = to_timezone_and_hourly(weather_wpl, tz="UTC")
                if ("roughness_length", 0) not in weather_wpl.columns:
                    weather_wpl[("roughness_length", 0)] = z0_default
                st.session_state["weather_wpl"] = weather_wpl
                st.session_state["weather_source_meta"] = {"name": "User EPW", "details": {"rows": int(len(weather_wpl))}}
                st.session_state["period_label"] = f"{weather_wpl.index.min().date()} to {weather_wpl.index.max().date()}"
                st.success(f"Parsed & normalized {len(weather_wpl)} hours from EPW.")

# ---- Tab 2: Turbine ----
with tab_turbine:
    st.subheader("Turbine selection")
    mode = st.radio("Choose", ["Library turbine (OEDB)", "Custom turbine (power curve)"], horizontal=True)

    selected_turbine = None
    if mode == "Library turbine (OEDB)":
        df_types = wt.get_turbine_types(print_out=False)  # manufacturer, turbine_type, flags
        mf = st.selectbox("Manufacturer", sorted(df_types["manufacturer"].unique()))
        subset = df_types[df_types["manufacturer"] == mf]
        tt = st.selectbox("Turbine type", sorted(subset["turbine_type"].unique()))
        hub_height = st.number_input("Hub height (m)", value=100, step=1)
        if st.button("Select turbine"):
            selected_turbine = WindTurbine(turbine_type=tt, hub_height=hub_height)
            st.session_state["turbine"] = selected_turbine
            st.session_state["turbine_cfg"] = {
                "name": tt, "hub_height": hub_height,
                "rated_power_W": getattr(selected_turbine, "nominal_power", None)
            }
            st.success(f"Selected {mf} {tt} @ {hub_height} m")
    else:
        st.markdown("Upload a power curve table with columns: `wind_speed` [m/s], `value` [W].")
        pc_file = st.file_uploader("Power curve CSV", type=["csv"])
        nominal_power = st.number_input("Nominal power [W]", value=3_000_000, step=1000)
        hub_height = st.number_input("Hub height (m)", value=100, step=1)
        if pc_file is not None and st.button("Use custom turbine"):
            pc = pd.read_csv(pc_file)
            if not {"wind_speed", "value"}.issubset(pc.columns):
                st.error("CSV must contain 'wind_speed' and 'value' columns.")
            else:
                tdict = {"nominal_power": float(nominal_power), "hub_height": float(hub_height), "power_curve": pc[["wind_speed", "value"]]}
                selected_turbine = WindTurbine(**tdict)
                st.session_state["turbine"] = selected_turbine
                st.session_state["turbine_cfg"] = {
                    "name": "Custom power curve", "hub_height": hub_height, "rated_power_W": float(nominal_power)
                }
                st.success("Custom turbine loaded.")

    if "turbine" in st.session_state:
        t: WindTurbine = st.session_state["turbine"]
        with st.expander("Power curve preview"):
            try:
                if t.power_curve is not None:
                    st.line_chart(t.power_curve.set_index("wind_speed")["value"])
                elif t.power_coefficient_curve is not None:
                    st.line_chart(t.power_coefficient_curve.set_index("wind_speed")["value"])
            except Exception:
                pass

# ---- Tab 3: Run ----
with tab_run:
    st.subheader("Run windpowerlib ModelChain")
    run_ok = st.button("Compute power time series")
    if run_ok:
        if "weather_wpl" not in st.session_state:
            st.error("No weather data loaded. Use tab 1 first.")
        elif "turbine" not in st.session_state:
            st.error("No turbine selected. Use tab 2.")
        else:
            weather = st.session_state["weather_wpl"]
            turbine = st.session_state["turbine"]

            if ("wind_speed", 10) not in weather.columns and ("wind_speed", 100) not in weather.columns:
                st.error("Need wind speed at 10 m and/or 100 m.")
            else:
                mc_kwargs = dict(
                    wind_speed_model=wind_speed_model,
                    density_model=density_model,
                    temperature_model=temperature_model,
                    power_output_model="power_curve" if turbine.power_curve is not None else "power_coefficient_curve",
                    density_correction=density_correction,
                    obstacle_height=obstacle_height,
                    hellman_exp=hellman_exp if wind_speed_model == "hellman" else None,
                )
                mc = ModelChain(turbine, **{k: v for k, v in mc_kwargs.items() if v is not None}).run_model(weather)

                # Persist results & configs
                st.session_state["power_output_W"] = mc.power_output
                # Optional: store hub variables when available
                st.session_state["wind_speed_hub"] = getattr(mc, "wind_speed_hub", None)
                st.session_state["density_hub"] = getattr(mc, "density_hub", None)
                st.session_state["temperature_hub"] = getattr(mc, "temperature_hub", None)
                st.session_state["wind_dir_deg"] = getattr(mc, "wind_direction", None)
                st.session_state["mc_kwargs"] = mc_kwargs
                st.session_state["model_cfg"] = mc_kwargs
                # Example losses/availability placeholder (extend as you add UI)
                st.session_state["losses_cfg"] = {"availability": 1.0}
                st.session_state["results_ready"] = True

                st.success("Model run complete.")
                with st.expander("ModelChain parameters used"):
                    st.json({k: v for k, v in mc_kwargs.items() if v is not None})

# ---- Tab 4: Results ----
with tab_results:
    st.subheader("Results")
    if "power_output_W" in st.session_state:
        po = st.session_state["power_output_W"]
        st.line_chart(po)

        # KPIs
        avg_kw = po.mean() / 1000.0
        cap_fac = 100.0 * po.mean() / float(getattr(st.session_state["turbine"], "nominal_power", np.nan))
        st.metric("Average power [kW]", f"{avg_kw:.1f}")
        st.metric("Capacity factor [%]", f"{cap_fac:.1f}")

        with st.expander("Daily energy [MWh]"):
            energy_MWh = po.resample("1D").sum() / 1e6 * (1.0 / 3600.0)  # W*s -> MWh
            st.dataframe(energy_MWh.rename("energy_MWh"))

        # Downloads
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv_po = po.to_frame(name="power_W").to_csv(index=True).encode()
            st.download_button("Download power time series (CSV)", data=csv_po, file_name="power_timeseries.csv")
        with col_dl2:
            if "weather_wpl" in st.session_state:
                flat_cols = ["{}_{}".format(v, h) for v, h in st.session_state["weather_wpl"].columns]
                w_flat = st.session_state["weather_wpl"].copy()
                w_flat.columns = flat_cols
                csv_w = w_flat.to_csv(index=True).encode()
                st.download_button("Download normalized weather (CSV)", data=csv_w, file_name="weather_normalized.csv")

        # -------- Report generation --------
        st.subheader("Report")
        if st.session_state.get("results_ready"):
            site_info = {
                "lat": f"{lat:.5f}",
                "lon": f"{lon:.5f}",
                "addr": st.session_state.get("geo_result", {}).get("display_name", ""),
                "period": st.session_state.get("period_label", f"{start.date()} to {end.date()}"),
            }
            weather_source = st.session_state.get("weather_source_meta", {"name": "Weather", "details": {}})
            turbine_cfg = st.session_state.get("turbine_cfg", {})
            model_cfg = st.session_state.get("model_cfg", {})
            losses_cfg = st.session_state.get("losses_cfg", {})

            if st.button("Generate PDF report"):
                pdf_bytes = build_report_bytes(
                    site_info=site_info,
                    weather_source=weather_source,
                    turbine_cfg=turbine_cfg,
                    model_cfg=model_cfg,
                    losses_cfg=losses_cfg,
                    power_W=st.session_state["power_output_W"],
                    hub_speed=st.session_state.get("wind_speed_hub"),
                    hub_density=st.session_state.get("density_hub"),
                    hub_temperature_K=st.session_state.get("temperature_hub"),
                    wind_dir_deg=st.session_state.get("wind_dir_deg"),
                )
                st.download_button(
                    "Download PDF report",
                    data=pdf_bytes,
                    file_name="wind_report.pdf",
                    mime="application/pdf",
                )
    else:
        st.info("Run the model first on tab 3.")

# ---- Tab 5: About ----
with tab_about:
    st.subheader("About this tool")

    st.markdown(
        """
**Windpowerlib Dashboard** is a streamlined web app for simulating wind turbine energy production using
the open-source **windpowerlib** and a small set of robust weather inputs.

### What it does
- Fetches and normalizes weather from:
  - **Open-Meteo (archive)** — 10 m & 100 m wind, 2 m temperature, surface pressure
  - **NASA POWER (hourly, RE)** — WS10M/WS50M, T2M, PS, Z0M
  - **PVGIS TMY** (via pvlib) — typical meteorological year
  - **EPW upload** — EnergyPlus files, parsed via pvlib
- Converts data to windpowerlib’s expected MultiIndex format (variables × heights).
- Lets you pick a library turbine (OEDB) or upload a custom power curve.
- Runs **ModelChain** to compute hourly power.
- Exports:
  - CSVs for normalized weather and power time series
  - A polished **PDF report** with inputs, KPIs and charts
        """
    )

    # Versions
    st.markdown("### Libraries & Versions")
    colA, colB = st.columns(2)
    with colA:
        st.write(
            {
                "Python": f"{__import__('sys').version.split()[0]}",
                "streamlit": _get_version("streamlit"),
                "pandas": _get_version("pandas"),
                "numpy": _get_version("numpy"),
                "requests": _get_version("requests"),
                "geopy": _get_version("geopy"),
            }
        )
    with colB:
        st.write(
            {
                "windpowerlib": _get_version("windpowerlib"),
                "pvlib": _get_version("pvlib"),
                "reportlab": _get_version("reportlab"),
                "matplotlib": _get_version("matplotlib"),
            }
        )

    st.markdown(
        """
### How to use
1. **Weather data**: choose a source (or upload EPW) and press the fetch/parse button.  
2. **Turbine**: pick an OEDB turbine and hub height (or upload a custom power curve).  
3. **Run**: press *Compute power time series* to simulate.  
4. **Results**: preview plots/tables, download CSVs, and generate a **PDF report**.

**Tip:** For a whole-year simulation, the app fixes the period to the selected calendar year (UTC).  
If a source doesn’t provide roughness, we inject your sidebar **z0** default.

### Developer
Built by **Vaijayanth Sheri**, focused on pragmatic, reproducible energy analytics.
        """
    )
