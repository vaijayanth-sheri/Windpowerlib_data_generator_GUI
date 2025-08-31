# datasources_open_meteo.py
from __future__ import annotations
import requests
import pandas as pd

# Open-Meteo historical weather API:
# wind_speed_10m / 100m [request in m/s], wind_direction, temperature_2m [°C], surface_pressure [hPa]
# Docs: https://open-meteo.com/en/docs/historical-weather-api

def fetch_open_meteo(lat, lon, start_dt, end_dt) -> pd.DataFrame:
    base = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": start_dt.strftime("%Y-%m-%d"),
        "end_date": end_dt.strftime("%Y-%m-%d"),
        "hourly": ",".join(
            [
                "wind_speed_10m",
                "wind_speed_100m",
                "wind_direction_10m",
                "wind_direction_100m",
                "temperature_2m",
                "surface_pressure",
            ]
        ),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "timeformat": "iso8601",
    }
    r = requests.get(base, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    hourly = js.get("hourly", {})
    time = pd.to_datetime(hourly.get("time", []), utc=True)

    def sget(name):
        arr = hourly.get(name)
        return pd.Series(arr, index=time) if arr is not None else None

    df = pd.DataFrame(index=time)
    ws10 = sget("wind_speed_10m")
    ws100 = sget("wind_speed_100m")
    wd10 = sget("wind_direction_10m")
    wd100 = sget("wind_direction_100m")
    t2m_c = sget("temperature_2m")
    sp_hpa = sget("surface_pressure")

    if ws10 is not None:
        df["ws10"] = ws10
    if ws100 is not None:
        df["ws100"] = ws100
    if wd10 is not None:
        df["wd10"] = wd10
    if wd100 is not None:
        df["wd100"] = wd100
    if t2m_c is not None:
        df["t2m_k"] = t2m_c + 273.15  # °C -> K
    if sp_hpa is not None:
        df["sp_pa"] = sp_hpa * 100.0  # hPa -> Pa

    # Note: Open-Meteo does not provide z0 (roughness). Caller may set fallback.
    return df.sort_index()
