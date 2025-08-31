# model_windpowerlib.py
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
from windpowerlib import ModelChain, WindTurbine


def run_modelchain(
    turbine: WindTurbine,
    weather_wpl: pd.DataFrame,
    *,
    wind_speed_model: str = "logarithmic",
    density_model: str = "ideal_gas",
    temperature_model: str = "linear_gradient",
    power_output_model: str = "power_curve",
    density_correction: bool = True,
    obstacle_height: float = 0.0,
    hellman_exp: Optional[float] = None,
) -> Dict[str, pd.Series]:
    """
    Execute windpowerlib ModelChain against a normalized weather DataFrame.

    Parameters are aligned with the current stable ModelChain API.
    """
    mc_kwargs = dict(
        wind_speed_model=wind_speed_model,
        density_model=density_model,
        temperature_model=temperature_model,
        power_output_model=power_output_model,
        density_correction=density_correction,
        obstacle_height=obstacle_height,
    )
    # hellman exponent is only meaningful if wind_speed_model == "hellman"
    if wind_speed_model == "hellman" and hellman_exp is not None:
        mc_kwargs["hellman_exp"] = float(hellman_exp)

    mc = ModelChain(turbine, **mc_kwargs)
    mc.run_model(weather_wpl)

    # Expose key series; ModelChain attaches results to turbine / mc
    out = {
        "power_W": mc.power_output,             # W per turbine
        "wind_speed_hub": mc.wind_speed_hub,    # m/s at hub
        "density_hub": mc.density_hub,          # kg/m^3 if computed
        "temperature_hub": mc.temperature_hub,  # K if computed
    }
    return out
