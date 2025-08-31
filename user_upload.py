# user_upload.py
from __future__ import annotations
import io
import pandas as pd
from pvlib.iotools import read_epw  # robust EPW parser

def parse_epw(file_like) -> pd.DataFrame:
    """
    Read EPW using pvlib, then map to our fields:
      temp_air [°C] -> T2m_K
      atmospheric_pressure [Pa] -> SP_Pa
      wind_speed [m/s] -> WS10m
    """
    # Get bytes from Streamlit UploadedFile and decode to text
    content = file_like.getvalue() if hasattr(file_like, "getvalue") else file_like.read()
    text = content.decode("utf-8", errors="ignore") if isinstance(content, (bytes, bytearray)) else str(content)

    data, _meta = read_epw(io.StringIO(text))  # pvlib expects a text buffer
    if data is None or data.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz="UTC"))

    idx = pd.to_datetime(data.index, utc=True)
    out = pd.DataFrame(index=idx)

    # pvlib EPW canonical column names
    if "temp_air" in data.columns:
        out["T2m_K"] = pd.to_numeric(data["temp_air"], errors="coerce") + 273.15
    if "atmospheric_pressure" in data.columns:
        out["SP_Pa"] = pd.to_numeric(data["atmospheric_pressure"], errors="coerce")
    if "wind_speed" in data.columns:
        out["WS10m"] = pd.to_numeric(data["wind_speed"], errors="coerce")  # 10 m in EPW

    return out.sort_index().dropna(how="all")
