"""Noise model functions for device simulation.

All functions accept conductance arrays and return modified arrays.
They are designed to be composable and stateless — randomness is
controlled via a ``numpy.random.Generator`` passed explicitly or
obtained from the global config.

Scientific assumptions
----------------------
* **Gaussian read noise**: Models thermal + 1/f noise. σ is specified
  as a fraction of ``g_range`` (relative sigma).  This is a common
  simplification — real noise depends on bias voltage and temperature.
* **Uniform noise**: Simple bounded noise for sensitivity analysis.
* **Programming error**: Models write inaccuracy as Gaussian jitter
  applied after quantization.

All results are clamped to ``[g_min, g_max]`` to respect physical bounds.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.config import get_rng


def gaussian_noise(
    g: np.ndarray,
    sigma: float,
    g_min: float,
    g_max: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add zero-mean Gaussian read noise.

    Parameters
    ----------
    g : ndarray
        Input conductances.
    sigma : float
        Relative standard deviation (fraction of ``g_max - g_min``).
    g_min, g_max : float
        Conductance bounds for clamping.
    rng : Generator, optional
        NumPy random generator.  Falls back to global seed if ``None``.

    Returns
    -------
    ndarray
        Noisy conductances, clamped to ``[g_min, g_max]``.
    """
    if sigma <= 0.0:
        return g.copy()
    if rng is None:
        rng = get_rng()
    abs_sigma = sigma * (g_max - g_min)
    noise = rng.normal(loc=0.0, scale=abs_sigma, size=g.shape)
    return np.clip(g + noise, g_min, g_max)


def uniform_noise(
    g: np.ndarray,
    half_range: float,
    g_min: float,
    g_max: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add uniformly distributed noise in ``[-half_range, +half_range]``.

    Parameters
    ----------
    g : ndarray
        Input conductances.
    half_range : float
        Half-width of the uniform noise, as fraction of ``g_max - g_min``.
    g_min, g_max : float
        Conductance bounds.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    ndarray
        Noisy conductances, clamped.
    """
    if half_range <= 0.0:
        return g.copy()
    if rng is None:
        rng = get_rng()
    abs_half = half_range * (g_max - g_min)
    noise = rng.uniform(-abs_half, abs_half, size=g.shape)
    return np.clip(g + noise, g_min, g_max)


def programming_error(
    g: np.ndarray,
    sigma: float,
    g_min: float,
    g_max: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simulate write/programming inaccuracy.

    Applied *after* quantization to model the fact that programming a
    memristive device to a target conductance is imprecise.

    Parameters
    ----------
    g : ndarray
        Target (quantized) conductances.
    sigma : float
        Relative programming error std (fraction of ``g_max - g_min``).
    g_min, g_max : float
        Conductance bounds.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    ndarray
        Programmed conductances with error, clamped.
    """
    return gaussian_noise(g, sigma, g_min, g_max, rng)


def stuck_at_faults(
    g: np.ndarray,
    fault_rate: float,
    g_min: float,
    g_max: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Randomly set a fraction of devices to stuck-at-min or stuck-at-max.

    Parameters
    ----------
    g : ndarray
        Input conductances.
    fault_rate : float
        Fraction of devices affected (0.0 to 1.0).
    g_min, g_max : float
        Values used for stuck-at-min / stuck-at-max.
    rng : Generator, optional
        NumPy random generator.

    Returns
    -------
    ndarray
        Conductances with faults injected.
    """
    if fault_rate <= 0.0:
        return g.copy()
    if rng is None:
        rng = get_rng()
    result = g.copy()
    mask = rng.random(size=g.shape) < fault_rate
    # 50/50 stuck-at-min vs stuck-at-max
    high_mask = rng.random(size=g.shape) < 0.5
    result[mask & high_mask] = g_max
    result[mask & ~high_mask] = g_min
    return result
