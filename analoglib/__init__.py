"""AnalogLib — Open-Source Analog Computing Library.

A Python library for analog neural-network simulation on resistive
crossbar architectures (ReRAM, memristive devices).

Quick start::

    import analoglib as al

    device = al.devices.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)
    xbar = al.Crossbar(128, 64, device=device, differential=True)
    xbar.load_weights(W)
    out = xbar.vmm(V)

    al.save("model.analog", [xbar])
    loaded = al.load("model.analog")
"""

from __future__ import annotations

# Version
from .version import __version__

# Core types
from .core.types import SimulationMode, MappingMode, NoiseType
from .core.config import set_seed, get_rng, CFG
from .core.backend import to_numpy

# Devices
from . import devices
from .devices.base import Device
from .devices.ideal import IdealDevice
from .devices.reram import ReRAM

# Mapping
from . import mapping
from .mapping.base import MappingStrategy
from .mapping.differential import DifferentialMapping
from .mapping.offset import OffsetMapping

# Crossbar
from .crossbar.crossbar import Crossbar

# ADC / DAC
from .adc_dac.adc import ADC
from .adc_dac.dac import DAC

# Simulation
from .simulation.engine import SimulationEngine

# Serialization
from .serialization.analog_format import save, load


__all__ = [
    # Version
    "__version__",
    # Core
    "SimulationMode", "MappingMode", "NoiseType",
    "set_seed", "get_rng", "CFG", "to_numpy",
    # Devices
    "devices", "Device", "IdealDevice", "ReRAM",
    # Mapping
    "mapping", "MappingStrategy", "DifferentialMapping", "OffsetMapping",
    # Crossbar
    "Crossbar",
    # ADC / DAC
    "ADC", "DAC",
    # Simulation
    "SimulationEngine",
    # Serialization
    "save", "load",
]
