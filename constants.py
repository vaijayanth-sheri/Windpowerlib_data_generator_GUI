# constants.py
from __future__ import annotations

# Sensible defaults and limits used across the app

# Wind speed plausible range [m/s]
WIND_MIN = 0.0
WIND_MAX = 60.0

# Temperature plausible range [K]
TEMP_MIN = 190.0
TEMP_MAX = 330.0

# Pressure plausible range [Pa]
PRES_MIN = 50_000.0
PRES_MAX = 110_000.0

# Defaults
DEFAULT_Z0 = 0.05            # m  (short grassland-ish)
DEFAULT_HELLMAN_ALPHA = 0.14
DEFAULT_AVAILABILITY = 0.97  # fraction
DEFAULT_WAKE_LOSS = 0.10     # fraction

# Plot settings
ROSE_BINS = 16
