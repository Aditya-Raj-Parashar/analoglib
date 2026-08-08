"""AnalogProfiler — hardware performance analytics.

Computes power, energy, area, and efficiency metrics from crossbar
simulation results.

Physical formulas
-----------------
All formulas use real voltages (not ideal assumptions):

    P_array = sum_{i,j} V_i^2 * G_{i,j}    [Watts]
    E_read  = P_array * t_read              [Joules/VMM]
    E_adc   = P_adc * (1/f_adc)            [Joules/conversion] (estimated)
    ops     = 2 * M * N                    [int multiplies + adds]
    TOPS/W  = (ops / E_total) * 1e-12

Note: Once IR-drop support is enabled, V_i should be replaced by V_{ij}
per-cell voltages from the hardware solver. The current implementation
uses nominal row voltages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

from ..crossbar.crossbar import Crossbar
from ..adc_dac.adc import ADC
from ..adc_dac.dac import DAC


# ---------------------------------------------------------------------------
# AnalogReport
# ---------------------------------------------------------------------------

@dataclass
class AnalogReport:
    """Container for hardware performance metrics.

    All energy/power units are SI (Watts, Joules).
    All area units are µm² (micrometer squared).
    """
    # Model summary
    n_layers:      int = 0
    total_params:  int = 0
    n_tiles:       int = 0

    # Per-VMM metrics
    ops_per_vmm:       int   = 0      # 2 * M * N per layer
    array_power_W:     float = 0.0    # P = sum V²G
    read_energy_J:     float = 0.0    # P * t_read
    adc_energy_J:      float = 0.0    # estimated ADC cost
    dac_energy_J:      float = 0.0    # estimated DAC cost
    total_energy_J:    float = 0.0    # sum of above
    tops_per_watt:     float = 0.0    # TOPS/W
    throughput_tops:   float = 0.0    # TOPS at given latency
    latency_s:         float = 0.0    # estimated read latency

    # Area
    cell_area_um2:     float = 0.0    # crossbar array area
    peripheral_area_um2: float = 0.0  # ADC+DAC area estimate

    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted summary string."""
        def _fmt(v: float, unit: str) -> str:
            if v == 0:
                return "N/A"
            for prefix, scale in [("T", 1e12), ("G", 1e9), ("M", 1e6), ("k", 1e3),
                                   ("", 1), ("m", 1e-3), ("u", 1e-6), ("n", 1e-9),
                                   ("p", 1e-12), ("f", 1e-15)]:
                if abs(v) >= scale or scale == 1e-15:
                    return f"{v/scale:.3f} {prefix}{unit}"
            return str(v)

        lines = [
            "=" * 52,
            "  AnalogLib Hardware Performance Report",
            "=" * 52,
            f"  Layers:          {self.n_layers}",
            f"  Parameters:      {self.total_params:,}",
            f"  Ops/VMM:         {self.ops_per_vmm:,}",
            f"  Array Power:     {_fmt(self.array_power_W, 'W')}",
            f"  Read Energy:     {_fmt(self.read_energy_J, 'J')}/VMM",
            f"  ADC Energy:      {_fmt(self.adc_energy_J, 'J')}/VMM",
            f"  Total Energy:    {_fmt(self.total_energy_J, 'J')}/VMM",
            f"  TOPS/W:          {self.tops_per_watt:.3f}",
            f"  Latency:         {_fmt(self.latency_s, 's')}",
            f"  Cell Area:       {_fmt(self.cell_area_um2, 'm2')}",
            "=" * 52,
        ]
        return "\n".join(lines)

    def print(self) -> None:
        """Print the formatted summary."""
        print(self.summary())


# ---------------------------------------------------------------------------
# AnalogProfiler
# ---------------------------------------------------------------------------

class AnalogProfiler:
    """Compute hardware metrics from a SimulationEngine.

    Parameters
    ----------
    t_read : float
        Array readout time in seconds (default 10 ns).
    cell_feature_F : float
        Cell feature size in nm (default 10 nm = 10nm process node, 4F² cell).
    V_supply : float
        Nominal supply voltage in Volts (default 1.0 V).
    """

    # Energy constants from literature
    _ADC_ENERGY_PER_BIT_PJ = 0.5   # pJ/bit (typical for SAR ADC)
    _DAC_ENERGY_PER_BIT_PJ = 0.1   # pJ/bit (typical DAC)

    def __init__(
        self,
        t_read: float = 10e-9,
        cell_feature_F: float = 10.0,
        V_supply: float = 1.0,
    ) -> None:
        self.t_read = t_read
        self.cell_feature_F = cell_feature_F
        self.V_supply = V_supply

    def profile(
        self,
        crossbars: List[Crossbar],
        V_input: np.ndarray,
        adc: Optional[ADC] = None,
        dac: Optional[DAC] = None,
    ) -> AnalogReport:
        """Compute hardware metrics for a multi-layer crossbar inference.

        Parameters
        ----------
        crossbars : list of Crossbar
            The crossbar layers.
        V_input : ndarray
            Input voltage vector (used for power estimation).
        adc : ADC, optional
            ADC specification (for energy estimation).
        dac : DAC, optional
            DAC specification (for energy estimation).

        Returns
        -------
        AnalogReport
        """
        report = AnalogReport()
        report.n_layers = len(crossbars)

        total_ops = 0
        total_power_W = 0.0
        total_cell_area = 0.0

        V = V_input.copy()
        for i, xbar in enumerate(crossbars):
            rows, cols = xbar.rows, xbar.cols
            report.total_params += rows * cols
            total_ops += 2 * rows * cols  # MAC = multiply + add

            # Power: P = sum V_i^2 * G_ij (nom: uses G+ - G- mag)
            G_pos, G_neg = xbar.get_conductance()
            G_eff = np.abs(G_pos - G_neg)  # effective conductance magnitude

            # Broadcast row voltages over column dimension
            if V.ndim == 1 and V.shape[0] == rows:
                V_sq = V ** 2  # (rows,)
                power_layer = float(np.sum(V_sq[:, np.newaxis] * G_eff))
            else:
                power_layer = float(np.sum(G_eff) * (self.V_supply ** 2))

            total_power_W += power_layer

            # Cell area: 4F² per cell (1-R1-T crossbar cell)
            F_m = self.cell_feature_F * 1e-9
            total_cell_area += rows * cols * 4 * (F_m * 1e6) ** 2  # µm²

            # For multi-layer: pass ideal output as next V approximation
            V = xbar.vmm(V if V.ndim == 1 else V[0])

        report.ops_per_vmm = total_ops
        report.array_power_W = total_power_W
        report.read_energy_J = total_power_W * self.t_read
        report.cell_area_um2 = total_cell_area
        report.latency_s = self.t_read * len(crossbars)

        # ADC/DAC energy estimate
        if adc is not None:
            # Each output column needs 1 ADC conversion
            n_adc_conversions = sum(xb.cols for xb in crossbars)
            report.adc_energy_J = (
                n_adc_conversions * adc.bits * self._ADC_ENERGY_PER_BIT_PJ * 1e-12
            )
        if dac is not None:
            n_dac_conversions = crossbars[0].rows if crossbars else 0
            report.dac_energy_J = (
                n_dac_conversions * dac.bits * self._DAC_ENERGY_PER_BIT_PJ * 1e-12
            )

        report.total_energy_J = (
            report.read_energy_J + report.adc_energy_J + report.dac_energy_J
        )

        # TOPS/W = ops / (energy_per_VMM × 1e12)
        if report.total_energy_J > 0:
            report.tops_per_watt = report.ops_per_vmm / (report.total_energy_J * 1e12)

        # Throughput = ops / latency_s → in TOPS
        if report.latency_s > 0:
            report.throughput_tops = report.ops_per_vmm / (report.latency_s * 1e12)

        # Peripheral area (lookup by ADC bits from published data)
        if adc is not None:
            report.peripheral_area_um2 = self._adc_area_um2(adc.bits) * sum(xb.cols for xb in crossbars)

        return report

    def _adc_area_um2(self, bits: int) -> float:
        """Estimated ADC area per converter by bit precision (µm²).
        Based on published 6T1C SAR ADC area vs. resolution trend.
        """
        _lookup = {4: 0.5, 6: 1.2, 8: 3.5, 10: 9.0, 12: 25.0}
        closest = min(_lookup, key=lambda k: abs(k - bits))
        return _lookup[closest]
