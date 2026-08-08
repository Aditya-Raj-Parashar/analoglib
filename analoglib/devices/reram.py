"""ReRAM (Resistive RAM) device model.

Scientific basis
----------------
* Conductance is modelled as a set of discrete states uniformly
  distributed between ``g_min`` and ``g_max``.  This is the standard
  uniform-level quantization model used in NeuroSim, CrossSim, etc.
* Read noise is modelled as additive Gaussian with σ proportional to
  the conductance window.  In real devices σ may also depend on the
  programmed state — this is a simplification.
* Device-to-device (D2D) variation shifts each device's conductance by
  a random offset drawn at *program time* (modelled here as a single
  call to ``add_variation``).
* Stuck-at faults simulate devices permanently stuck at ``g_min`` or
  ``g_max``.
* Programming error models write imprecision as Gaussian jitter.

All approximations are clearly documented.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..core.config import get_rng
from .base import Device
from . import noise as noise_fn


class ReRAM(Device):
    """Resistive RAM device with configurable non-idealities.

    Parameters
    ----------
    g_min : float
        Minimum conductance (S).
    g_max : float
        Maximum conductance (S).
    num_states : int
        Number of discrete programmable levels (e.g. 256 for 8-bit).
    read_noise_sigma : float
        Relative σ of Gaussian read noise (fraction of window).
    programming_error_sigma : float
        Relative σ of programming error.
    d2d_variation_sigma : float
        Relative σ of device-to-device variation.
    stuck_at_fault_rate : float
        Fraction of devices with stuck-at faults (0–1).
    """

    def __init__(
        self,
        g_min: float = 1e-6,
        g_max: float = 100e-6,
        num_states: int = 256,
        read_noise_sigma: float = 0.0,
        programming_error_sigma: float = 0.0,
        d2d_variation_sigma: float = 0.0,
        stuck_at_fault_rate: float = 0.0,
    ) -> None:
        super().__init__(g_min=g_min, g_max=g_max, num_states=num_states)
        self.read_noise_sigma = read_noise_sigma
        self.programming_error_sigma = programming_error_sigma
        self.d2d_variation_sigma = d2d_variation_sigma
        self.stuck_at_fault_rate = stuck_at_fault_rate

    # ------------------------------------------------------------------
    # Device interface
    # ------------------------------------------------------------------

    def quantize(self, g: np.ndarray) -> np.ndarray:
        """Snap conductances to nearest discrete level.

        1. Clamp to ``[g_min, g_max]``
        2. Round to nearest of ``num_states`` uniformly spaced levels
        3. Apply programming error (if configured)
        """
        g_clamped = np.clip(g, self.g_min, self.g_max)

        if self.num_states <= 1:
            return g_clamped

        levels = self.conductance_levels()
        step = levels[1] - levels[0]

        # Quantize: snap to nearest level
        idx = np.round((g_clamped - self.g_min) / step).astype(int)
        idx = np.clip(idx, 0, self.num_states - 1)
        quantized = levels[idx]

        # Optional programming error
        if self.programming_error_sigma > 0:
            quantized = noise_fn.programming_error(
                quantized,
                self.programming_error_sigma,
                self.g_min,
                self.g_max,
            )

        return quantized

    def add_noise(self, g: np.ndarray) -> np.ndarray:
        """Inject Gaussian read noise."""
        if self.read_noise_sigma <= 0:
            return g.copy()
        return noise_fn.gaussian_noise(
            g, self.read_noise_sigma, self.g_min, self.g_max
        )

    def add_variation(self, g: np.ndarray) -> np.ndarray:
        """Apply device-to-device variation + stuck-at faults."""
        result = g.copy()

        if self.d2d_variation_sigma > 0:
            result = noise_fn.gaussian_noise(
                result, self.d2d_variation_sigma, self.g_min, self.g_max
            )

        if self.stuck_at_fault_rate > 0:
            result = noise_fn.stuck_at_faults(
                result, self.stuck_at_fault_rate, self.g_min, self.g_max
            )

        return result

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "read_noise_sigma": self.read_noise_sigma,
            "programming_error_sigma": self.programming_error_sigma,
            "d2d_variation_sigma": self.d2d_variation_sigma,
            "stuck_at_fault_rate": self.stuck_at_fault_rate,
        })
        return d

    def __repr__(self) -> str:
        return (
            f"ReRAM(g_min={self.g_min:.2e}, g_max={self.g_max:.2e}, "
            f"states={self.num_states}, "
            f"noise_σ={self.read_noise_sigma}, "
            f"prog_err_σ={self.programming_error_sigma}, "
            f"d2d_σ={self.d2d_variation_sigma}, "
            f"stuck_at={self.stuck_at_fault_rate})"
        )
