"""Model-vs-observation verification statistics.

Shared by the P0 scatter and Taylor-diagram figures so every figure that
reports skill metrics uses the same formulas.
"""

from __future__ import annotations

import numpy as np


def mse_decomposition(obs: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    """Takacs (1985) MSE decomposition into dissipative (amplitude) and
    dispersive (phase) error, as used in the RBMet reference for this project
    (scielo.br/j/rbmet, section 2.2.3 "Erros, metricas, escores e indices
    estatisticos").

    ``mean((model - obs)**2)`` decomposes *exactly* as
    ``bias**2 + MSE_diss + MSE_disp``, where::

        MSE_diss = (std(model) - std(obs))**2               # amplitude error
        MSE_disp = 2 * std(model) * std(obs) * (1 - corr)    # phase error

    (bias == mean(model - obs); corr is the Pearson correlation between
    *obs* and *model*). Returns ``(mse_diss, mse_disp)``.
    """
    std_obs = float(np.std(obs))
    std_model = float(np.std(model))
    corr = float(np.corrcoef(obs, model)[0, 1])
    mse_diss = (std_model - std_obs) ** 2
    mse_disp = 2.0 * std_model * std_obs * (1.0 - corr)
    return mse_diss, mse_disp


def taylor_stats(obs: np.ndarray, model: np.ndarray) -> tuple[float, float]:
    """Normalized standard deviation (model/obs) and Pearson correlation —
    the two coordinates plotted on a normalized Taylor diagram."""
    std_obs = float(np.std(obs))
    std_model = float(np.std(model))
    corr = float(np.corrcoef(obs, model)[0, 1])
    return std_model / std_obs, corr
