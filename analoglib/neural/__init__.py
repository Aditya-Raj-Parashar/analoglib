"""analoglib.neural — neural network model converters.

Converts trained framework models (NumPy / PyTorch / ONNX) into AIRGraph
for simulation with AnalogLib.

Usage::

    from analoglib.neural import from_numpy, from_torch
"""

from .numpy_converter import from_numpy
from .torch_converter import from_torch

__all__ = ["from_numpy", "from_torch"]
