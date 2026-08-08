"""Analog-to-Digital Converter (ADC) model.

Converts continuous analog current/voltage signals to discrete digital
values.  Models the quantization error introduced by finite ADC resolution.

Scientific assumptions
----------------------
* **Uniform quantization** — levels are equally spaced between
  ``v_min`` and ``v_max``.  Non-uniform (e.g. logarithmic) ADC can be
  added as a subclass.
* Clipping: values outside ``[v_min, v_max]`` are hard-clipped.
* No ADC nonlinearity or offset error in the base model (can be added).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class ADC:
    """Uniform quantization ADC.

    Parameters
    ----------
    bits : int
        ADC resolution in bits.  ``n`` bits → ``2ⁿ`` quantization levels.
    v_min : float
        Minimum representable value (e.g. 0 V for unipolar).
    v_max : float
        Maximum representable value.
    """

    def __init__(
        self,
        bits: int = 8,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ) -> None:
        if bits < 1:
            raise ValueError(f"ADC bits must be ≥ 1, got {bits}")
        if v_max <= v_min:
            raise ValueError(f"v_max ({v_max}) must be > v_min ({v_min})")

        self.bits = bits
        self.v_min = v_min
        self.v_max = v_max
        self.num_levels = 2 ** bits

    @property
    def resolution(self) -> float:
        """Voltage step per LSB."""
        return (self.v_max - self.v_min) / (self.num_levels - 1)

    def convert(self, x: np.ndarray) -> np.ndarray:
        """Quantize continuous values to ADC levels.

        Parameters
        ----------
        x : ndarray
            Analog values.

        Returns
        -------
        ndarray
            Quantized values (same shape, still float for downstream math).
        """
        clipped = np.clip(x, self.v_min, self.v_max)
        step = self.resolution
        quantized = np.round((clipped - self.v_min) / step) * step + self.v_min
        return quantized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }

    def __repr__(self) -> str:
        return f"ADC(bits={self.bits}, range=[{self.v_min}, {self.v_max}])"
