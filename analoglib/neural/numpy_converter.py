"""NumPy model converter — simplest converter for weight arrays.

Converts a plain list of NumPy/PyTorch/TF weight arrays into an AIRGraph.
This is the foundation that the other converters (PyTorch, ONNX) build on.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from ..air.schema import AIRGraph, AIRLayer, LayerType, ActivationFn
from ..core.backend import to_numpy


def from_numpy(
    weights: List[Any],
    *,
    name: str = "model",
    activations: Optional[List[Optional[str]]] = None,
) -> AIRGraph:
    """Build an AIRGraph from a list of weight matrices.

    Parameters
    ----------
    weights : list of array-like
        Each entry is a 2-D weight matrix for one dense layer.
    name : str
        Model name in the AIRGraph.
    activations : list of str or None, optional
        Activation fn names (relu, sigmoid, tanh, softmax, none).

    Returns
    -------
    AIRGraph
    """
    if activations is not None and len(activations) != len(weights):
        raise ValueError(
            f"activations length ({len(activations)}) must match weights length ({len(weights)})"
        )

    g = AIRGraph(name=name)
    for i, W in enumerate(weights):
        W = to_numpy(W)
        if W.ndim != 2:
            raise ValueError(f"Weight matrix {i} must be 2-D, got shape {W.shape}")

        g.add_layer(AIRLayer(
            layer_type=LayerType.CROSSBAR,
            name=f"crossbar_{i}",
            matrix_shape=(W.shape[0], W.shape[1]),
            weights=W.copy(),
        ))

        if activations is not None and activations[i] is not None:
            fn_str = activations[i].lower()
            try:
                fn = ActivationFn(fn_str)
            except ValueError:
                raise ValueError(
                    f"Unknown activation {fn_str!r}. "
                    f"Must be one of: {[e.value for e in ActivationFn]}"
                )
            if fn != ActivationFn.NONE:
                g.add_layer(AIRLayer(
                    layer_type=LayerType.ACTIVATION,
                    name=f"act_{i}",
                    activation_fn=fn,
                ))

    return g
