"""Differential mapping: W → (G⁺, G⁻).

The most common mapping for signed weights in crossbar arrays.

Mathematical basis
------------------
Given a weight ``w`` in ``[-w_max, +w_max]``:

1. Normalise to ``[0, 1]``:  ``w_norm = (w + w_max) / (2 * w_max)``
2. Map to conductance:
   * ``G⁺ = g_min + w_norm * (g_max - g_min)``
   * ``G⁻ = g_min + (1 - w_norm) * (g_max - g_min)``
3. Effective weight ∝ ``G⁺ - G⁻``

Inverse:
   ``w_norm = (G⁺ - g_min) / (g_max - g_min)``
   ``w = w_norm * 2 * w_max - w_max``

Properties:
- Every weight is represented by *two* physical devices.
- When ``w = 0``: ``G⁺ = G⁻ = g_mid``.
- Symmetric: ``w`` and ``-w`` swap ``G⁺`` and ``G⁻``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..devices.base import Device
from .base import MappingStrategy


class DifferentialMapping(MappingStrategy):
    """Differential (G⁺ / G⁻) weight-to-conductance mapping.

    Parameters
    ----------
    w_max : float or None
        Maximum absolute weight value for normalisation.  If ``None``,
        ``w_max`` is inferred from ``max(|W|)`` during mapping (data-dependent).
    """

    def __init__(self, w_max: float | None = None) -> None:
        self.w_max = w_max

    def weights_to_conductance(
        self,
        W: np.ndarray,
        device: Device,
    ) -> Tuple[np.ndarray, np.ndarray]:
        w_max = self.w_max if self.w_max is not None else np.max(np.abs(W))
        if w_max == 0:
            w_max = 1.0  # avoid division by zero for all-zero weights

        # Normalise to [0, 1]
        w_norm = (W + w_max) / (2.0 * w_max)
        w_norm = np.clip(w_norm, 0.0, 1.0)

        # Map to conductances
        g_pos = device.g_min + w_norm * device.g_range
        g_neg = device.g_min + (1.0 - w_norm) * device.g_range

        return g_pos, g_neg

    def conductance_to_weights(
        self,
        *G: np.ndarray,
        device: Device,
    ) -> np.ndarray:
        if len(G) != 2:
            raise ValueError(
                f"DifferentialMapping expects 2 conductance matrices (G+, G-), "
                f"got {len(G)}"
            )
        g_pos, g_neg = G

        w_max = self.w_max
        if w_max is None:
            # Cannot perfectly recover w_max from conductances alone;
            # use the full conductance range as the mapping domain.
            w_max = 1.0

        w_norm = (g_pos - device.g_min) / device.g_range
        W = w_norm * 2.0 * w_max - w_max
        return W

    def to_dict(self):
        d = super().to_dict()
        d["w_max"] = self.w_max
        return d

    def __repr__(self) -> str:
        return f"DifferentialMapping(w_max={self.w_max})"
