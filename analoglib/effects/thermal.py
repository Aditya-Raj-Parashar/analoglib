"""Thermal effect — temperature-dependent conductance shift.

Physical model
--------------
ReRAM conductance follows Arrhenius-like temperature dependence:

    G(T) = G_0 × exp(-E_a / (k_B × T))

where:
    G_0  : pre-exponential conductance factor
    E_a  : activation energy (eV), typically 0.1–0.5 eV for ReRAM
    k_B  : Boltzmann constant (8.617e-5 eV/K)
    T    : temperature in Kelvin

We implement this as a multiplicative correction factor relative to the
nominal temperature T_ref (default 300 K):

    correction(T) = exp(-E_a/k_B × (1/T - 1/T_ref))

At T = T_ref, correction = 1.0 (no change).
At T > T_ref, G increases (thermal activation).
At T < T_ref, G decreases.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Effect, EffectContext

# Boltzmann constant in eV/K
_k_B = 8.617333262e-5


class Thermal(Effect):
    """Temperature-dependent conductance scaling.

    Parameters
    ----------
    E_a : float
        Activation energy in eV (default 0.1 eV).
    T_ref : float
        Reference temperature in Kelvin at which G is nominal (default 300 K).
    """

    def __init__(self, E_a: float = 0.1, T_ref: float = 300.0) -> None:
        if E_a < 0:
            raise ValueError(f"E_a must be >= 0, got {E_a}")
        if T_ref <= 0:
            raise ValueError(f"T_ref must be > 0, got {T_ref}")
        self.E_a   = E_a
        self.T_ref = T_ref

    def apply(self, g: np.ndarray, context: EffectContext) -> np.ndarray:
        """Scale all conductances by the Arrhenius temperature correction.

        Parameters
        ----------
        g : ndarray
            Nominal conductance matrix.
        context : EffectContext
            Must contain T_kelvin.

        Returns
        -------
        ndarray
            Temperature-corrected conductance matrix.
        """
        T = context.T_kelvin
        if T == self.T_ref:
            return g.copy()

        inv_T_diff = (1.0 / T) - (1.0 / self.T_ref)
        correction = np.exp(-self.E_a / _k_B * inv_T_diff)
        return g * correction

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "Thermal", "E_a": self.E_a, "T_ref": self.T_ref}

    @classmethod
    def _from_dict_impl(cls, d: Dict[str, Any]) -> "Thermal":
        return cls(E_a=d["E_a"], T_ref=d.get("T_ref", 300.0))

    def __repr__(self) -> str:
        return f"Thermal(E_a={self.E_a} eV, T_ref={self.T_ref} K)"
