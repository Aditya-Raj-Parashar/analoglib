"""Drift effect — temporal conductance relaxation (retention loss).

Physical model
--------------
ReRAM devices exhibit power-law conductance relaxation over time:

    G(t) = G_0 × (t / t_0)^(-nu)

where:
    G_0  : conductance at reference time t_0
    t_0  : reference time (1 second by default)
    nu   : drift exponent (typically 0.02–0.1 for ReRAM, 0.1–0.14 for PCM)
    t    : elapsed time since programming (seconds)

At t = t_0, G(t_0) = G_0 (no drift).
For t > t_0, G decreases monotonically.

Reference: Ambrogio et al., "Neuromorphic Learning and Recognition with
One-Transistor-One-Resistor Synapses", IEEE T-ED 2016.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Effect, EffectContext


class Drift(Effect):
    """Power-law temporal conductance drift.

    Parameters
    ----------
    nu : float
        Drift exponent. Higher = faster drift. Typical: 0.02–0.12.
    t_0 : float
        Reference time in seconds (default 1.0).
    """

    def __init__(self, nu: float = 0.05, t_0: float = 1.0) -> None:
        if nu < 0:
            raise ValueError(f"nu must be >= 0, got {nu}")
        if t_0 <= 0:
            raise ValueError(f"t_0 must be > 0, got {t_0}")
        self.nu  = nu
        self.t_0 = t_0

    def apply(self, g: np.ndarray, context: EffectContext) -> np.ndarray:
        """Scale conductances with power-law drift factor.

        Parameters
        ----------
        g : ndarray
            Nominal conductance matrix.
        context : EffectContext
            Must contain t_seconds (elapsed time since programming).

        Returns
        -------
        ndarray
            Drifted conductance matrix.
        """
        t = context.t_seconds
        if t <= 0 or self.nu == 0:
            return g.copy()

        # drift_factor = (t / t_0)^(-nu)
        # Clip t to >= t_0 so factor <= 1 (conductance can only decrease)
        t_ratio = max(t, self.t_0) / self.t_0
        drift_factor = t_ratio ** (-self.nu)
        return g * drift_factor

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Drift", "nu": self.nu, "t_0": self.t_0}

    @classmethod
    def _from_dict_impl(cls, d: Dict[str, Any]) -> "Drift":
        return cls(nu=d["nu"], t_0=d.get("t_0", 1.0))

    def __repr__(self) -> str:
        return f"Drift(nu={self.nu}, t_0={self.t_0} s)"
