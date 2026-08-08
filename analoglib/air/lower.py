"""AIR Lowering Pass — converts an AIRGraph to SimulationEngine.

The lowering pass is the compiler step:

    AIRGraph  ──→  list[Crossbar]  ──→  SimulationEngine

All hardware decisions (quantize, device, mapping, peripherals) are
resolved from the AIR layer configs during this step.

Invariant (must hold in ideal mode)
------------------------------------
    lower(air_graph).run(x, mode="ideal") ≈ chained_matmul(x, W_layers)

to within floating-point precision (ideal crossbar = exact VMM).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .schema import AIRGraph, AIRLayer, LayerType, ActivationFn, PeripheralConfig
from ..crossbar.crossbar import Crossbar
from ..devices.base import Device
from ..devices.ideal import IdealDevice
from ..mapping.base import MappingStrategy
from ..mapping.differential import DifferentialMapping
from ..adc_dac.adc import ADC
from ..adc_dac.dac import DAC
from ..simulation.engine import SimulationEngine


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def lower(air_graph: AIRGraph, quantize: bool = True) -> SimulationEngine:
    """Lower an AIRGraph to a runnable SimulationEngine.

    Parameters
    ----------
    air_graph : AIRGraph
        A validated AIR graph with weights loaded on all CROSSBAR layers.
    quantize : bool
        If True, conductances are quantized to device levels during
        ``load_weights``. Set False for ideal mathematical verification.

    Returns
    -------
    SimulationEngine
        Ready-to-run engine. ADC/DAC are attached when the first CROSSBAR
        layer specifies a ``PeripheralConfig`` with non-zero bits.

    Raises
    ------
    ValueError
        If the graph is invalid or a CROSSBAR layer has no weights.
    """
    air_graph.validate()

    crossbars: List[Crossbar] = []
    adc: Optional[ADC] = None
    dac: Optional[DAC] = None

    for layer in air_graph.layers:
        if layer.layer_type == LayerType.CROSSBAR:
            xbar, layer_adc, layer_dac = _lower_crossbar(layer, quantize=quantize)
            crossbars.append(xbar)
            # Use first ADC/DAC found — for multi-layer models each layer
            # may have its own peripherals; for now the engine uses one pair.
            # Per-layer ADC/DAC is tracked in the AIRLayer and will be
            # consumed by the future per-layer SimulationEngine (Phase 4+).
            if layer_adc is not None and adc is None:
                adc = layer_adc
            if layer_dac is not None and dac is None:
                dac = layer_dac

        elif layer.layer_type == LayerType.ACTIVATION:
            # Activation layers are not yet handled inside SimulationEngine.
            # They are stored in the AIRLayer and will be wired in Phase 2
            # when the AnalogModel builds its own forward pass.
            # For now, lowering silently skips activations.
            pass

        elif layer.layer_type == LayerType.INLINE:
            pass  # future use

    if not crossbars:
        raise ValueError("AIRGraph contains no CROSSBAR layers; cannot lower to SimulationEngine.")

    return SimulationEngine(crossbars=crossbars, adc=adc, dac=dac)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_device(device_config: Optional[dict]) -> Device:
    """Reconstruct a Device from its serialized config dict."""
    if device_config is None:
        return IdealDevice()
    from ..devices.base import Device as DeviceBase
    return DeviceBase.from_dict(device_config)


def _resolve_mapping(mapping_config: Optional[dict]) -> MappingStrategy:
    """Reconstruct a MappingStrategy from its serialized config dict."""
    if mapping_config is None:
        return DifferentialMapping()
    from ..mapping.base import MappingStrategy as MappingBase
    return MappingBase.from_dict(mapping_config)


def _resolve_peripherals(
    peripherals: Optional[PeripheralConfig],
) -> Tuple[Optional[DAC], Optional[ADC]]:
    """Build DAC/ADC from PeripheralConfig. Returns (dac, adc)."""
    if peripherals is None:
        return None, None
    dac = DAC(bits=peripherals.dac_bits,
              v_min=peripherals.dac_v_min,
              v_max=peripherals.dac_v_max) if peripherals.dac_bits > 0 else None
    adc = ADC(bits=peripherals.adc_bits,
              v_min=peripherals.adc_v_min,
              v_max=peripherals.adc_v_max) if peripherals.adc_bits > 0 else None
    return dac, adc


def _lower_crossbar(
    layer: AIRLayer,
    quantize: bool,
) -> Tuple[Crossbar, Optional[ADC], Optional[DAC]]:
    """Lower a single CROSSBAR AIRLayer to a Crossbar + optional peripherals."""
    rows, cols = layer.matrix_shape
    device  = _resolve_device(layer.device_config)
    mapping = _resolve_mapping(layer.mapping_config)

    xbar = Crossbar(rows=rows, cols=cols, device=device, mapping=mapping, differential=True)
    xbar.load_weights(layer.weights, quantize=quantize)

    dac, adc = _resolve_peripherals(layer.peripherals)
    return xbar, adc, dac
