"""AnalogModel — high-level user-facing model API.

``AnalogModel`` wraps an ``AIRGraph`` and exposes a clean, researcher-friendly
interface:

    model = AnalogModel.from_numpy(weights_list)
    model.compile(device=ReRAM(...), mapping="differential", tile=None)
    result = model.simulate(x, mode="hardware")
    result.report()

The class acts as the *compiler front-end*: it holds the AIRGraph,
lowers it to a SimulationEngine, and returns a SimulationResult.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .schema import (
    AIRGraph, AIRLayer, LayerType, ActivationFn,
    PeripheralConfig, EffectConfig,
)
from .lower import lower
from ..core.backend import to_numpy
from ..core.types import SimulationMode
from ..devices.base import Device
from ..devices.ideal import IdealDevice
from ..devices.reram import ReRAM
from ..mapping.base import MappingStrategy
from ..mapping.differential import DifferentialMapping
from ..simulation.engine import SimulationEngine


# ---------------------------------------------------------------------------
# SimulationResult
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Container for the output of ``AnalogModel.simulate()``.

    Attributes
    ----------
    output : ndarray
        Final network output.
    mode : str
        Simulation mode used.
    elapsed_s : float
        Wall-clock seconds for the simulation pass.
    engine : SimulationEngine
        The engine that produced the result.
    meta : dict
        Arbitrary extra info from the model.
    """
    output: np.ndarray
    mode: str
    elapsed_s: float
    engine: SimulationEngine
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------

    def report(self) -> None:
        """Print a structured hardware summary."""
        n_xbar = len(self.engine.crossbars)
        total_params = sum(
            xb.rows * xb.cols for xb in self.engine.crossbars
        )
        print("=" * 50)
        print("  AnalogLib Simulation Report")
        print("=" * 50)
        print(f"  Mode:         {self.mode.upper()}")
        print(f"  Crossbars:    {n_xbar}")
        print(f"  Parameters:   {total_params:,}")
        for i, xb in enumerate(self.engine.crossbars):
            print(f"  Layer {i}:      {xb.rows}x{xb.cols}  device={xb.device.__class__.__name__}")
        if self.engine.adc:
            print(f"  ADC:          {self.engine.adc.bits}-bit  [{self.engine.adc.v_min:.2e}, {self.engine.adc.v_max:.2e}]")
        if self.engine.dac:
            print(f"  DAC:          {self.engine.dac.bits}-bit  [{self.engine.dac.v_min:.2e}, {self.engine.dac.v_max:.2e}]")
        print(f"  Elapsed:      {self.elapsed_s * 1000:.3f} ms")
        print(f"  Output shape: {self.output.shape}")
        print(f"  Output:       {self.output}")
        print("=" * 50)


# ---------------------------------------------------------------------------
# AnalogModel
# ---------------------------------------------------------------------------

class AnalogModel:
    """High-level analog neural network model.

    Wraps an AIRGraph and provides compile/simulate/report workflow.

    Parameters
    ----------
    air_graph : AIRGraph
        The underlying analog intermediate representation.

    Examples
    --------
    Build from NumPy weight matrices::

        model = AnalogModel.from_numpy(
            [W1, W2],
            name="my_mlp",
        )
        model.compile(device=al.ReRAM(...), mapping="differential")
        result = model.simulate(x, mode="hardware")
        result.report()

    Build manually via AIRGraph::

        g = AIRGraph(name="demo")
        g.add_layer(AIRLayer(LayerType.CROSSBAR, name="fc0",
                             matrix_shape=(128, 64), weights=W))
        model = AnalogModel(g)
    """

    def __init__(self, air_graph: AIRGraph) -> None:
        self._graph = air_graph
        self._engine: Optional[SimulationEngine] = None
        self._compiled = False

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_numpy(
        cls,
        weights: List[np.ndarray],
        *,
        name: str = "model",
        description: str = "",
        activations: Optional[List[Optional[str]]] = None,
    ) -> "AnalogModel":
        """Build an AnalogModel from a list of NumPy weight matrices.

        Parameters
        ----------
        weights : list of ndarray
            Each array is a 2-D weight matrix ``(rows, cols)`` for one layer.
        name : str
            Model name stored in the AIRGraph.
        description : str
            Optional description.
        activations : list of str or None, optional
            Activation function name after each cross-bar layer.
            ``None`` or ``"none"`` means no activation.
            Must be same length as ``weights`` if provided.

        Returns
        -------
        AnalogModel
        """
        if activations is not None and len(activations) != len(weights):
            raise ValueError(
                f"activations list length ({len(activations)}) must match "
                f"weights list length ({len(weights)})"
            )

        g = AIRGraph(name=name, description=description)
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

            # Activation after this crossbar
            if activations is not None and activations[i] is not None:
                fn_str = activations[i].lower()
                fn = ActivationFn(fn_str)
                if fn != ActivationFn.NONE:
                    g.add_layer(AIRLayer(
                        layer_type=LayerType.ACTIVATION,
                        name=f"act_{i}",
                        activation_fn=fn,
                    ))

        return cls(g)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(
        self,
        *,
        device: Optional[Device] = None,
        mapping: Union[str, MappingStrategy, None] = None,
        tile: Optional[tuple] = None,
        adc_bits: int = 0,
        dac_bits: int = 0,
        adc_range: tuple = (-500e-6, 500e-6),
        dac_range: tuple = (0.0, 1.0),
        quantize: bool = True,
    ) -> "AnalogModel":
        """Compile AIR graph into a hardware simulation engine.

        Parameters
        ----------
        device : Device, optional
            Device model for all crossbar layers.  Defaults to IdealDevice.
        mapping : str or MappingStrategy, optional
            Mapping strategy.  "differential" (default) or "offset".
        tile : (tile_rows, tile_cols), optional
            Tile shape for TiledCrossbar (Phase 3).  None = full crossbar.
        adc_bits : int
            ADC resolution for all layers.  0 = no ADC.
        dac_bits : int
            DAC resolution for all layers.  0 = no DAC.
        adc_range : (min_A, max_A)
            ADC input current range in Amperes.
        dac_range : (min_V, max_V)
            DAC voltage range in Volts.
        quantize : bool
            Quantize conductances to device levels.

        Returns
        -------
        self (for method chaining)
        """
        device = device or IdealDevice()
        if isinstance(mapping, str):
            if mapping == "offset":
                from ..mapping.offset import OffsetMapping
                mapping = OffsetMapping()
            else:
                mapping = DifferentialMapping()
        mapping = mapping or DifferentialMapping()

        peripherals = None
        if adc_bits > 0 or dac_bits > 0:
            peripherals = PeripheralConfig(
                dac_bits=dac_bits,
                adc_bits=adc_bits,
                dac_v_min=dac_range[0],
                dac_v_max=dac_range[1],
                adc_v_min=adc_range[0],
                adc_v_max=adc_range[1],
            )

        # Stamp device/mapping/peripherals/tile onto all crossbar layers
        for layer in self._graph.layers:
            if layer.layer_type == LayerType.CROSSBAR:
                layer.device_config  = device.to_dict()
                layer.mapping_config = mapping.to_dict()
                layer.tile_shape     = tile
                layer.peripherals    = peripherals

        self._engine = lower(self._graph, quantize=quantize)
        self._compiled = True
        return self

    # ------------------------------------------------------------------
    # Simulate
    # ------------------------------------------------------------------

    def simulate(
        self,
        x: Any,
        mode: Union[str, SimulationMode] = "ideal",
    ) -> SimulationResult:
        """Run a forward pass through the compiled model.

        Parameters
        ----------
        x : array-like
            Input vector or batch.
        mode : str or SimulationMode
            "ideal", "device", or "hardware".

        Returns
        -------
        SimulationResult
        """
        if not self._compiled:
            raise RuntimeError(
                "Model not compiled. Call model.compile(...) before simulate()."
            )
        x = to_numpy(x)

        t0 = time.perf_counter()
        output = self._engine.run(x, mode=mode)
        elapsed = time.perf_counter() - t0

        mode_str = mode if isinstance(mode, str) else mode.name.lower()
        return SimulationResult(
            output=output,
            mode=mode_str,
            elapsed_s=elapsed,
            engine=self._engine,
            meta={"model_name": self._graph.name},
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def graph(self) -> AIRGraph:
        """The underlying AIRGraph."""
        return self._graph

    @property
    def engine(self) -> Optional[SimulationEngine]:
        """The compiled SimulationEngine (None until compile() is called)."""
        return self._engine

    def __repr__(self) -> str:
        status = "compiled" if self._compiled else "not compiled"
        return f"AnalogModel(name={self._graph.name!r}, {self._graph}, {status})"
