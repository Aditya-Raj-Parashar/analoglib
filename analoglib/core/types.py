"""Enumerations and type aliases used throughout analoglib."""

from __future__ import annotations

from enum import Enum, auto
from typing import Union

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class SimulationMode(Enum):
    """Fidelity levels for analog simulation."""
    IDEAL = auto()      # Perfect math — no noise, no quantization
    DEVICE = auto()     # Device-aware — quantization + noise + variation
    HARDWARE = auto()   # Hardware-aware — + ADC/DAC, voltage limits, IR drop
    SPICE = auto()      # Circuit-level — export / interface with SPICE


class MappingMode(Enum):
    """Weight-to-conductance mapping strategies."""
    DIFFERENTIAL = auto()   # G+ / G- pair per weight
    OFFSET = auto()         # Single device with a conductance offset
    CUSTOM = auto()         # User-supplied mapping


class NoiseType(Enum):
    """Supported noise injection models."""
    NONE = auto()
    GAUSSIAN = auto()
    UNIFORM = auto()
    THERMAL = auto()
    SHOT = auto()


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Accept anything array-like (will be cast to np.ndarray internally)
ArrayLike = Union[np.ndarray, list, tuple]

# Conductance value in Siemens (float)
Conductance = float

# Voltage in Volts (float)
Voltage = float
