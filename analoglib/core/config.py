"""Global configuration and reproducibility utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class AnalogConfig:
    """Library-wide configuration.

    Attributes
    ----------
    seed : int or None
        Master random seed.  ``None`` means non-deterministic.
    default_dtype : numpy dtype
        Default floating-point precision for conductance/weight arrays.
    float_tolerance : float
        Absolute tolerance used in numerical comparisons (e.g. roundtrip tests).
    """
    seed: Optional[int] = None
    default_dtype: np.dtype = field(default_factory=lambda: np.float64)
    float_tolerance: float = 1e-12

    def apply_seed(self) -> np.random.Generator:
        """Return a NumPy ``Generator`` seeded with :pyattr:`seed`.

        If ``seed`` is ``None``, an unpredictable generator is returned.
        """
        return np.random.default_rng(self.seed)


# Singleton — importable everywhere via ``from analoglib.core.config import CFG``
CFG = AnalogConfig()


def set_seed(seed: int) -> None:
    """Set the global random seed for reproducibility."""
    CFG.seed = seed


def get_rng() -> np.random.Generator:
    """Get a NumPy random generator honoring the global seed."""
    return CFG.apply_seed()
