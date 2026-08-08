"""Offset mapping: W → single G with a conductance offset.

Mathematical basis
------------------
Uses a single device per weight with an offset at mid-conductance:

1. ``g_mid = (g_min + g_max) / 2``
2. ``G = g_mid + w * scale``    where ``scale = g_range / (2 * w_max)``
3. Effective weight: ``w = (G - g_mid) / scale``

Properties:
- Half the devices compared to differential mapping.
- Lower dynamic range for signed weights.
- Offset current must be subtracted (can introduce systematic error).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from ..devices.base import Device
from .base import MappingStrategy


class OffsetMapping(MappingStrategy):
    """Single-device offset weight-to-conductance mapping.

    Parameters
    ----------
    w_max : float or None
        Maximum absolute weight for normalisation.
    """

    def __init__(self, w_max: float | None = None) -> None:
        self.w_max = w_max

    def weights_to_conductance(
        self,
        W: np.ndarray,
        device: Device,
    ) -> Tuple[np.ndarray]:
        w_max = self.w_max if self.w_max is not None else np.max(np.abs(W))
        if w_max == 0:
            w_max = 1.0

        g_mid = (device.g_min + device.g_max) / 2.0
        scale = device.g_range / (2.0 * w_max)

        G = g_mid + W * scale
        G = np.clip(G, device.g_min, device.g_max)
        return (G,)

    def conductance_to_weights(
        self,
        *G: np.ndarray,
        device: Device,
    ) -> np.ndarray:
        if len(G) != 1:
            raise ValueError(
                f"OffsetMapping expects 1 conductance matrix, got {len(G)}"
            )
        g = G[0]

        w_max = self.w_max if self.w_max is not None else 1.0
        g_mid = (device.g_min + device.g_max) / 2.0
        scale = device.g_range / (2.0 * w_max)

        return (g - g_mid) / scale

    def to_dict(self):
        d = super().to_dict()
        d["w_max"] = self.w_max
        return d

    def __repr__(self) -> str:
        return f"OffsetMapping(w_max={self.w_max})"
