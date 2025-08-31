# physics.py
from __future__ import annotations
import numpy as np
import pandas as pd

R_AIR = 287.058  # J/(kg*K)

def air_density_ideal_gas(T_K, p_Pa):
    """
    ρ = p / (R T)
    """
    return p_Pa / (R_AIR * T_K)

def uv_to_speed_dir(u, v):
    """
    Convert U,V (eastward, northward) to wind speed [m/s] and direction [deg, meteorological "from"].
    Direction = arctan2(-u, -v) in degrees, range [0, 360).
    Accepts arrays/Series.
    """
    u_arr = np.asarray(u)
    v_arr = np.asarray(v)
    spd = np.sqrt(u_arr**2 + v_arr**2)
    # meteorological direction (blowing from)
    direc = (np.degrees(np.arctan2(-u_arr, -v_arr)) + 360.0) % 360.0
    if isinstance(u, pd.Series):
        spd = pd.Series(spd, index=u.index)
        direc = pd.Series(direc, index=u.index)
    return spd, direc

def shear_loglaw(v_ref, z_ref, z_target, z0, d0=0.0):
    """
    Log-law wind profile: v(z) = v_ref * ln((z-d0)/z0) / ln((z_ref-d0)/z0)
    """
    return v_ref * np.log((z_target - d0) / z0) / np.log((z_ref - d0) / z0)

def shear_hellman(v_ref, z_ref, z_target, alpha=0.14):
    """
    Power-law wind shear: v(z) = v_ref * (z/z_ref)^alpha
    """
    return v_ref * (z_target / z_ref) ** alpha
