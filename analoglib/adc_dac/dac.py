"""Digital-to-Analog Converter (DAC) model.

Converts digital input values to quantized analog voltage levels for
driving crossbar row inputs.

Scientific assumptions
----------------------
* **Uniform quantization** — voltage levels are equally spaced.
* In real hardware, DAC output settles with finite slew rate — this
  model assumes instantaneous settling (static DAC).
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class DAC:
    """Uniform quantization DAC.

    Parameters
    ----------
    bits : int
        DAC resolution in bits.
    v_min : float
        Minimum output voltage.
    v_max : float
        Maximum output voltage.
    """

    def __init__(
        self,
        bits: int = 8,
        v_min: float = 0.0,
        v_max: float = 1.0,
    ) -> None:
        if bits < 1:
            raise ValueError(f"DAC bits must be ≥ 1, got {bits}")
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
        """Quantize input to DAC voltage levels.

        Parameters
        ----------
        x : ndarray
            Digital input values (will be clipped to ``[v_min, v_max]``).

        Returns
        -------
        ndarray
            Quantized voltage values.
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
        return f"DAC(bits={self.bits}, range=[{self.v_min}, {self.v_max}])"
