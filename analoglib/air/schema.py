"""AIR Schema — core data structures for the Analog Intermediate Representation.

AIRGraph is a framework-neutral model graph. Every converter (PyTorch, TF, ONNX)
produces an AIRGraph. Every backend (Crossbar, TiledCrossbar, SPICE) consumes one.

Design principles
-----------------
* Pure Python dataclasses — no NumPy dependency in the schema itself.
* Fully serializable to/from plain dict (JSON/YAML-compatible).
* Explicit unit documentation in all numeric fields.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LayerType(str, Enum):
    """Type of a node in an AIR graph."""
    CROSSBAR   = "crossbar"    # dense weight matrix / analog VMM
    ACTIVATION = "activation"  # non-linearity (ReLU, Sigmoid, …)
    INLINE     = "inline"      # arbitrary numpy function (for testing)


class ActivationFn(str, Enum):
    """Supported activation functions."""
    RELU    = "relu"
    SIGMOID = "sigmoid"
    TANH    = "tanh"
    SOFTMAX = "softmax"
    NONE    = "none"


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class PeripheralConfig:
    """ADC/DAC specification for a crossbar layer.

    Parameters
    ----------
    dac_bits : int
        DAC resolution (input quantization). 0 means no DAC.
    adc_bits : int
        ADC resolution (output quantization). 0 means no ADC.
    dac_v_min, dac_v_max : float
        DAC voltage range (Volts).
    adc_v_min, adc_v_max : float
        ADC input range (Amperes — matches crossbar output units).
    """
    dac_bits: int = 0
    adc_bits: int = 0
    dac_v_min: float = 0.0
    dac_v_max: float = 1.0
    adc_v_min: float = -500e-6  # A (default: ±500 µA)
    adc_v_max: float = 500e-6

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dac_bits": self.dac_bits,
            "adc_bits": self.adc_bits,
            "dac_v_min": self.dac_v_min,
            "dac_v_max": self.dac_v_max,
            "adc_v_min": self.adc_v_min,
            "adc_v_max": self.adc_v_max,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PeripheralConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EffectConfig:
    """Configuration for physical hardware effects (Phase 4).

    Currently stored as an opaque dict so the schema remains stable
    when new effects are added. Each key is an effect name, each value
    is its parameter dict.

    Example::

        EffectConfig(effects={"ir_drop": {"r_wire": 1.0}, "drift": {"nu": 0.02}})
    """
    effects: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.effects) == 0

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.effects)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EffectConfig":
        return cls(effects=copy.deepcopy(d))


# ---------------------------------------------------------------------------
# AIR Layer
# ---------------------------------------------------------------------------

@dataclass
class AIRLayer:
    """A single node in an AIR graph.

    Parameters
    ----------
    layer_type : LayerType
        Type of this node.
    name : str
        Human-readable identifier (must be unique within a graph).
    matrix_shape : tuple (rows, cols), optional
        Weight matrix shape for CROSSBAR layers. None for activation/inline.
    weights : ndarray, optional
        Weight values. May be None for un-compiled graphs or placeholder layers.
    device_config : dict, optional
        Serialized device description (from ``Device.to_dict()``).
    mapping_config : dict, optional
        Serialized mapping description (from ``MappingStrategy.to_dict()``).
    tile_shape : tuple (tile_rows, tile_cols), optional
        If set, the lowering pass creates a ``TiledCrossbar``.
        If None, a single ``Crossbar`` is created.
    peripherals : PeripheralConfig, optional
        ADC/DAC specification. None means no peripheral quantization.
    effects : EffectConfig, optional
        Physical effects applied during simulation.
    activation_fn : ActivationFn
        For ACTIVATION layers, the function to apply.
    meta : dict
        Arbitrary user metadata (preserved through save/load).
    """
    layer_type: LayerType
    name: str

    # CROSSBAR fields
    matrix_shape: Optional[Tuple[int, int]] = None
    weights: Optional[np.ndarray] = None
    device_config: Optional[Dict[str, Any]] = None
    mapping_config: Optional[Dict[str, Any]] = None
    tile_shape: Optional[Tuple[int, int]] = None
    peripherals: Optional[PeripheralConfig] = None
    effects: Optional[EffectConfig] = None

    # ACTIVATION fields
    activation_fn: ActivationFn = ActivationFn.NONE

    # Freeform metadata
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        if self.layer_type == LayerType.CROSSBAR:
            if self.matrix_shape is None:
                raise ValueError(f"AIRLayer '{self.name}': CROSSBAR layer requires matrix_shape")
            if len(self.matrix_shape) != 2 or any(d <= 0 for d in self.matrix_shape):
                raise ValueError(
                    f"AIRLayer '{self.name}': matrix_shape must be (rows>0, cols>0), "
                    f"got {self.matrix_shape}"
                )
            if self.weights is not None:
                if self.weights.shape != tuple(self.matrix_shape):
                    raise ValueError(
                        f"AIRLayer '{self.name}': weights shape {self.weights.shape} "
                        f"does not match matrix_shape {self.matrix_shape}"
                    )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "layer_type": self.layer_type.value,
            "name": self.name,
            "activation_fn": self.activation_fn.value,
            "meta": self.meta,
        }
        if self.matrix_shape is not None:
            d["matrix_shape"] = list(self.matrix_shape)
        if self.weights is not None:
            d["weights"] = self.weights.tolist()
        if self.device_config is not None:
            d["device_config"] = self.device_config
        if self.mapping_config is not None:
            d["mapping_config"] = self.mapping_config
        if self.tile_shape is not None:
            d["tile_shape"] = list(self.tile_shape)
        if self.peripherals is not None:
            d["peripherals"] = self.peripherals.to_dict()
        if self.effects is not None:
            d["effects"] = self.effects.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AIRLayer":
        kwargs: Dict[str, Any] = {
            "layer_type": LayerType(d["layer_type"]),
            "name": d["name"],
            "activation_fn": ActivationFn(d.get("activation_fn", ActivationFn.NONE.value)),
            "meta": d.get("meta", {}),
        }
        if "matrix_shape" in d:
            kwargs["matrix_shape"] = tuple(d["matrix_shape"])
        if "weights" in d and d["weights"] is not None:
            kwargs["weights"] = np.array(d["weights"], dtype=np.float64)
        if "device_config" in d:
            kwargs["device_config"] = d["device_config"]
        if "mapping_config" in d:
            kwargs["mapping_config"] = d["mapping_config"]
        if "tile_shape" in d:
            kwargs["tile_shape"] = tuple(d["tile_shape"])
        if "peripherals" in d:
            kwargs["peripherals"] = PeripheralConfig.from_dict(d["peripherals"])
        if "effects" in d:
            kwargs["effects"] = EffectConfig.from_dict(d["effects"])
        return cls(**kwargs)

    def __repr__(self) -> str:
        if self.layer_type == LayerType.CROSSBAR:
            tiled = f", tile={self.tile_shape}" if self.tile_shape else ""
            wt = "loaded" if self.weights is not None else "empty"
            return f"AIRLayer(crossbar, name={self.name!r}, shape={self.matrix_shape}{tiled}, {wt})"
        return f"AIRLayer({self.layer_type.value}, name={self.name!r}, fn={self.activation_fn.value})"


# ---------------------------------------------------------------------------
# AIR Graph
# ---------------------------------------------------------------------------

class AIRGraph:
    """An ordered computation graph in the Analog Intermediate Representation.

    An AIRGraph is produced by model converters and consumed by:
    * The lowering pass (→ SimulationEngine / Crossbar objects)
    * Exporters (→ SPICE / Verilog-A)
    * The CLI (→ .analog file)

    Parameters
    ----------
    name : str
        Model name.
    description : str
        Optional description.
    meta : dict
        Arbitrary user metadata.
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.meta: Dict[str, Any] = meta or {}
        self._layers: List[AIRLayer] = []
        self._names: set = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_layer(self, layer: AIRLayer) -> "AIRGraph":
        """Append a layer to the graph. Returns self for chaining."""
        if layer.name in self._names:
            raise ValueError(f"AIRGraph already contains a layer named {layer.name!r}")
        self._layers.append(layer)
        self._names.add(layer.name)
        return self

    # ------------------------------------------------------------------
    # Read accessors
    # ------------------------------------------------------------------

    @property
    def layers(self) -> List[AIRLayer]:
        """Ordered list of all layers (read-only view)."""
        return list(self._layers)

    @property
    def crossbar_layers(self) -> List[AIRLayer]:
        """Crossbar-type layers only."""
        return [l for l in self._layers if l.layer_type == LayerType.CROSSBAR]

    def __len__(self) -> int:
        return len(self._layers)

    def __iter__(self):
        return iter(self._layers)

    def __repr__(self) -> str:
        n_xbar = len(self.crossbar_layers)
        n_act  = sum(1 for l in self._layers if l.layer_type == LayerType.ACTIVATION)
        return (
            f"AIRGraph(name={self.name!r}, "
            f"layers={len(self._layers)} "
            f"[{n_xbar} crossbar, {n_act} activation])"
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError if the graph is inconsistent."""
        if not self._layers:
            raise ValueError("AIRGraph is empty")
        for i, layer in enumerate(self._layers):
            if layer.layer_type == LayerType.CROSSBAR:
                if layer.weights is None:
                    raise ValueError(
                        f"Layer {i} ({layer.name!r}) has no weights loaded. "
                        f"Set layer.weights before lowering."
                    )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "meta": self.meta,
            "layers": [l.to_dict() for l in self._layers],
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AIRGraph":
        g = cls(
            name=d.get("name", ""),
            description=d.get("description", ""),
            meta=d.get("meta", {}),
        )
        for ld in d.get("layers", []):
            g.add_layer(AIRLayer.from_dict(ld))
        return g
