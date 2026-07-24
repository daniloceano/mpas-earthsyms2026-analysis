"""Thermodynamic derived quantities.

MPAS-Atmosphere history output carries potential temperature (``theta``) and
pressure at layer centers, not actual temperature directly.
"""

from __future__ import annotations

import numpy as np

P0_PA = 100000.0  # reference pressure, Pa
RD_CP = 0.286  # Rd/cp for dry air (~2/7), Poisson's equation exponent


def temperature_from_theta_pressure(theta, pressure):
    """Actual temperature (K) from potential temperature (K) and pressure (Pa)."""
    return theta * (pressure / P0_PA) ** RD_CP
