"""analoglib.air — Analog Intermediate Representation.

The AIR module is the central contract between:
  - Model converters (PyTorch / TF / ONNX)
  - Hardware backends (Crossbar / TiledCrossbar)
  - Exporters (SPICE / Verilog-A)
  - Analyzers (power, area, latency)

Public API::

    from analoglib.air import AIRGraph, AIRLayer, LayerType, lower
    from analoglib.air import AnalogModel
"""

from .schema import AIRGraph, AIRLayer, LayerType, PeripheralConfig, EffectConfig
from .lower import lower
from .model import AnalogModel

__all__ = [
    "AIRGraph",
    "AIRLayer",
    "LayerType",
    "PeripheralConfig",
    "EffectConfig",
    "lower",
    "AnalogModel",
]
