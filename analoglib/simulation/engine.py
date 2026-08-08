"""Simulation engine — runs analog inference at selected fidelity.

Provides a single ``SimulationEngine.run()`` method that pushes an input
through one or more crossbars, applying the appropriate level of
non-idealities based on the selected ``SimulationMode``.

Modes
-----
* **IDEAL** — pure matrix math, no noise, no quantization.
* **DEVICE** — quantized conductances + read noise + variation.
* **HARDWARE** — additionally applies DAC (input quantization) and
  ADC (output quantization).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..core.backend import to_numpy
from ..core.types import SimulationMode
from ..crossbar.crossbar import Crossbar
from ..adc_dac.adc import ADC
from ..adc_dac.dac import DAC


class SimulationEngine:
    """Orchestrates analog forward passes through crossbar(s).

    Parameters
    ----------
    crossbars : list of Crossbar
        Ordered list of crossbars representing successive layers.
    adc : ADC or None
        ADC model (used in HARDWARE mode).
    dac : DAC or None
        DAC model (used in HARDWARE mode).
    """

    def __init__(
        self,
        crossbars: List[Crossbar] | None = None,
        adc: ADC | None = None,
        dac: DAC | None = None,
    ) -> None:
        self.crossbars = crossbars or []
        self.adc = adc
        self.dac = dac

    def add_crossbar(self, xbar: Crossbar) -> None:
        """Append a crossbar to the layer stack."""
        self.crossbars.append(xbar)

    def run(
        self,
        x: Any,
        mode: SimulationMode | str = SimulationMode.IDEAL,
    ) -> np.ndarray:
        """Execute a forward pass through all crossbars.

        Parameters
        ----------
        x : array-like
            Input vector/batch.
        mode : SimulationMode or str
            Simulation fidelity.  Accepts enum or string like ``"ideal"``.

        Returns
        -------
        ndarray
            Output after passing through all crossbar layers.
        """
        if isinstance(mode, str):
            mode = SimulationMode[mode.upper()]

        x = to_numpy(x)

        for xbar in self.crossbars:
            # --- DAC: quantize input voltages (HARDWARE mode) ---
            if mode == SimulationMode.HARDWARE and self.dac is not None:
                x = self.dac.convert(x)

            # --- Crossbar VMM ---
            use_noise = mode in (SimulationMode.DEVICE, SimulationMode.HARDWARE)
            x = xbar.vmm(x, noise=use_noise, mode=mode)

            # --- ADC: quantize output currents (HARDWARE mode) ---
            if mode == SimulationMode.HARDWARE and self.adc is not None:
                x = self.adc.convert(x)

        return x

    def run_comparison(
        self,
        x: Any,
        modes: List[SimulationMode | str] | None = None,
    ) -> Dict[str, np.ndarray]:
        """Run inference in multiple modes and return results for comparison.

        Parameters
        ----------
        x : array-like
            Input data.
        modes : list, optional
            Modes to compare.  Defaults to ``["ideal", "device", "hardware"]``.

        Returns
        -------
        dict
            ``{mode_name: output_array}`` for each requested mode.
        """
        if modes is None:
            modes = [SimulationMode.IDEAL, SimulationMode.DEVICE, SimulationMode.HARDWARE]

        results = {}
        for m in modes:
            if isinstance(m, str):
                m = SimulationMode[m.upper()]
            results[m.name.lower()] = self.run(x, mode=m)

        return results

    def __repr__(self) -> str:
        n = len(self.crossbars)
        return f"SimulationEngine({n} crossbar{'s' if n != 1 else ''})"
