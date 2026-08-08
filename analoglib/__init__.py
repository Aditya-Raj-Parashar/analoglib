"""AnalogLib — Open-Source Analog Computing Library.

A Python library for analog neural-network simulation on resistive
crossbar architectures (ReRAM, memristive devices).

Quick start::

    import analoglib as al

    device = al.devices.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)
    xbar = al.Crossbar(128, 64, device=device)
    xbar.load_weights(W)
    out = xbar.vmm(V)

    # High-level workflow via AIR
    model = al.AnalogModel.from_numpy([W1, W2])
    model.compile(device=device, adc_bits=8, dac_bits=8)
    result = model.simulate(x, mode="hardware")
    result.report()
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

# Crossbar (single + tiled)
from .crossbar.crossbar import Crossbar
from .crossbar.tiled import TiledCrossbar

# ADC / DAC
from .adc_dac.adc import ADC
from .adc_dac.dac import DAC

# Simulation
from .simulation.engine import SimulationEngine

# Serialization
from .serialization.analog_format import save, load
from . import serialization

# AIR — Analog Intermediate Representation
from . import air
from .air.schema import AIRGraph, AIRLayer, LayerType, PeripheralConfig, EffectConfig
from .air.lower import lower
from .air.model import AnalogModel, SimulationResult

# Hardware Effects
from . import effects
from .effects.ir_drop import IRDrop
from .effects.thermal import Thermal
from .effects.drift import Drift

# Analytics
from . import analysis
from .analysis.profiler import AnalogProfiler, AnalogReport

# Exporters
from . import exporters
from .exporters.spice import SpiceExporter

# Visualization
from . import visualization

# Neural converters
from . import neural


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
    "Crossbar", "TiledCrossbar",
    # ADC / DAC
    "ADC", "DAC",
    # Simulation
    "SimulationEngine",
    # Serialization
    "save", "load", "serialization",
    # AIR
    "air",
    "AIRGraph", "AIRLayer", "LayerType", "PeripheralConfig", "EffectConfig",
    "lower",
    "AnalogModel", "SimulationResult",
    # Effects
    "effects", "IRDrop", "Thermal", "Drift",
    # Analytics
    "analysis", "AnalogProfiler", "AnalogReport",
    # Exporters
    "exporters", "SpiceExporter",
    # Visualization
    "visualization",
    # Neural
    "neural",
]
