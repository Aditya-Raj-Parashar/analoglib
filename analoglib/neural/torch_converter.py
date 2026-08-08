"""PyTorch model converter — walks nn.Module and emits AIRGraph.

Supports
--------
* ``nn.Linear``              → CrossbarLayer
* ``nn.ReLU / Sigmoid / Tanh / Softmax`` → ActivationLayer
* ``nn.Sequential``         → walks children recursively
* ``nn.Conv2d``             → im2col lowered to CrossbarLayer

Usage::

    import torch.nn as nn
    import analoglib as al

    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )
    air = al.neural.from_torch(model, input_shape=(128,))
    analog = al.AnalogModel(air).compile(device=al.ReRAM(...))
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np

from ..air.schema import AIRGraph, AIRLayer, LayerType, ActivationFn


_ACTIVATION_MAP = {
    "ReLU":    ActivationFn.RELU,
    "Sigmoid": ActivationFn.SIGMOID,
    "Tanh":    ActivationFn.TANH,
    "Softmax": ActivationFn.SOFTMAX,
}


def from_torch(
    module: Any,
    *,
    name: str = "torch_model",
    input_shape: Optional[Tuple[int, ...]] = None,
) -> AIRGraph:
    """Convert a PyTorch ``nn.Module`` to an AIRGraph.

    Parameters
    ----------
    module : nn.Module
        PyTorch model. Supported layers: ``Linear``, ``ReLU``,
        ``Sigmoid``, ``Tanh``, ``Sequential``, ``Conv2d`` (im2col).
    name : str
        Name for the resulting AIRGraph.
    input_shape : tuple, optional
        Input shape excluding batch dimension (for Conv2d im2col sizing).

    Returns
    -------
    AIRGraph
    """
    try:
        import torch.nn as nn
    except ImportError:
        raise ImportError(
            "PyTorch is required for from_torch(). "
            "Install it with: pip install torch"
        )

    g = AIRGraph(name=name, description=f"Converted from PyTorch {type(module).__name__}")
    _walk_module(module, g, nn, counter={"crossbar": 0, "act": 0})
    return g


def _walk_module(module, g: AIRGraph, nn, counter: dict) -> None:
    """Recursively walk module children and emit AIRLayer nodes."""
    try:
        import torch.nn as _nn
    except ImportError:
        _nn = None

    cls_name = type(module).__name__

    if cls_name == "Linear":
        W = module.weight.detach().cpu().numpy().T  # (in, out)
        i = counter["crossbar"]
        g.add_layer(AIRLayer(
            layer_type=LayerType.CROSSBAR,
            name=f"crossbar_{i}",
            matrix_shape=(W.shape[0], W.shape[1]),
            weights=W.copy(),
            meta={"source": "nn.Linear", "bias": module.bias is not None},
        ))
        counter["crossbar"] += 1

    elif cls_name in _ACTIVATION_MAP:
        fn = _ACTIVATION_MAP[cls_name]
        i = counter["act"]
        g.add_layer(AIRLayer(
            layer_type=LayerType.ACTIVATION,
            name=f"act_{i}",
            activation_fn=fn,
        ))
        counter["act"] += 1

    elif cls_name == "Conv2d":
        # im2col: weight shape (out_ch, in_ch, kH, kW)
        # → reshape to (in_ch * kH * kW, out_ch)
        W_raw = module.weight.detach().cpu().numpy()
        out_ch, in_ch, kH, kW = W_raw.shape
        W = W_raw.reshape(out_ch, in_ch * kH * kW).T  # (in_ch*kH*kW, out_ch)
        i = counter["crossbar"]
        g.add_layer(AIRLayer(
            layer_type=LayerType.CROSSBAR,
            name=f"crossbar_{i}_conv_im2col",
            matrix_shape=(W.shape[0], W.shape[1]),
            weights=W.copy(),
            meta={
                "source": "nn.Conv2d (im2col)",
                "kernel": (kH, kW),
                "in_channels": in_ch,
                "out_channels": out_ch,
            },
        ))
        counter["crossbar"] += 1

    elif cls_name == "Sequential" or hasattr(module, "children"):
        for child in module.children():
            _walk_module(child, g, nn, counter)

    else:
        # Unknown layer — skip with a warning stored as inline layer
        g.add_layer(AIRLayer(
            layer_type=LayerType.INLINE,
            name=f"unknown_{cls_name}_{id(module)}",
            meta={"warning": f"Unsupported layer type: {cls_name}"},
        ))
