"""Base classes for hardware effects.

All physical effects inherit from ``Effect`` and implement ``apply()``,
which receives a conductance matrix and an EffectContext and returns a
modified conductance matrix.

Design notes
------------
* Effects are stateless — all parameters are in __init__.
* Effects are composable — apply them as a list in Crossbar.
* EffectContext carries geometry and electrical state for each pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EffectContext:
    """Runtime context provided to each Effect during a VMM pass.

    Attributes
    ----------
    V_row : ndarray
        Applied row voltage vector (shape: rows).
    G : ndarray
        Conductance matrix at the time of the call (shape: rows x cols).
    T_kelvin : float
        Ambient temperature in Kelvin (default 300 K).
    t_seconds : float
        Time elapsed since programming (for drift models).
    mode : str
        Simulation mode string ("ideal", "device", "hardware").
    meta : dict
        Arbitrary extra context.
    """
    V_row: np.ndarray
    G: np.ndarray
    T_kelvin: float = 300.0
    t_seconds: float = 0.0
    mode: str = "hardware"
    meta: Dict[str, Any] = field(default_factory=dict)


class Effect(ABC):
    """Abstract base for all physical hardware effects.

    Subclasses are automatically registered by name via __init_subclass__.

    Parameters
    ----------
    name : str
        Human-readable identifier (set by subclass).
    """

    _registry: Dict[str, type] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        Effect._registry[cls.__name__] = cls

    @classmethod
    def registry(cls) -> Dict[str, type]:
        """Return dict of all registered Effect subclasses."""
        return dict(cls._registry)

    @classmethod
    def get(cls, name: str) -> type:
        """Look up an Effect class by name."""
        if name not in cls._registry:
            raise KeyError(f"Unknown effect {name!r}. Available: {list(cls._registry)}")
        return cls._registry[name]

    @abstractmethod
    def apply(self, g: np.ndarray, context: EffectContext) -> np.ndarray:
        """Apply this effect to a conductance matrix.

        Parameters
        ----------
        g : ndarray  shape (rows, cols)
            Input conductance matrix.
        context : EffectContext
            Runtime information (voltages, temperature, etc.).

        Returns
        -------
        ndarray  shape (rows, cols)
            Modified conductance matrix.
        """

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize this effect's configuration."""

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Effect":
        """Reconstruct an Effect from a serialized config dict."""
        name = d["type"]
        klass = cls.get(name)
        return klass._from_dict_impl(d)

    @classmethod
    def _from_dict_impl(cls, d: Dict[str, Any]) -> "Effect":
        raise NotImplementedError(f"{cls.__name__} must implement _from_dict_impl")
