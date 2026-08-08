"""Ideal (perfect) analog device — no quantization, no noise."""

from __future__ import annotations

import numpy as np

from .base import Device


class IdealDevice(Device):
    """A mathematically perfect analog memory device.

    * Infinite conductance resolution (continuous, no quantization).
    * Zero read noise, zero variation, zero programming error.

    Useful as a baseline for comparing against realistic devices.

    Parameters
    ----------
    g_min : float
        Minimum conductance (S).  Default 0.
    g_max : float
        Maximum conductance (S).  Default 1.0 (normalised).
    """

    def __init__(
        self,
        g_min: float = 0.0,
        g_max: float = 1.0,
        num_states: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(g_min=g_min, g_max=g_max, num_states=0)

    def quantize(self, g: np.ndarray) -> np.ndarray:
        """No quantization — just clamp to ``[g_min, g_max]``."""
        return np.clip(g, self.g_min, self.g_max)

    def add_noise(self, g: np.ndarray) -> np.ndarray:
        """No noise — return a copy."""
        return g.copy()

    def add_variation(self, g: np.ndarray) -> np.ndarray:
        """No variation — return a copy."""
        return g.copy()
