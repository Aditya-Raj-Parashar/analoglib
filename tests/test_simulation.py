"""Tests for the simulation engine."""

import numpy as np
import pytest

import analoglib as al
from analoglib.simulation import SimulationEngine
from analoglib.crossbar import Crossbar
from analoglib.devices import IdealDevice, ReRAM
from analoglib.mapping import DifferentialMapping
from analoglib.adc_dac import ADC, DAC
from analoglib.core.types import SimulationMode


class TestSimulationEngine:
    def setup_method(self):
        al.set_seed(42)
        self.dev = IdealDevice(g_min=0.0, g_max=1.0)
        self.mapping = DifferentialMapping(w_max=1.0)
        self.rng = np.random.default_rng(42)

    def _make_loaded_xbar(self, rows, cols, device=None):
        dev = device or self.dev
        xbar = Crossbar(rows, cols, device=dev, mapping=self.mapping)
        W = self.rng.uniform(-1, 1, (rows, cols))
        xbar.load_weights(W, quantize=False)
        return xbar, W

    def test_single_layer_ideal(self):
        xbar, W = self._make_loaded_xbar(4, 3)
        engine = SimulationEngine(crossbars=[xbar])
        V = self.rng.uniform(0, 1, 4)
        result = engine.run(V, mode="ideal")
        expected = V @ W
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_multi_layer(self):
        xbar1, W1 = self._make_loaded_xbar(4, 3)
        xbar2, W2 = self._make_loaded_xbar(3, 2)
        engine = SimulationEngine(crossbars=[xbar1, xbar2])
        V = self.rng.uniform(0, 1, 4)
        result = engine.run(V, mode="ideal")
        expected = (V @ W1) @ W2
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_hardware_mode_with_adc_dac(self):
        xbar, W = self._make_loaded_xbar(4, 3)
        adc = ADC(bits=8, v_min=-2.0, v_max=2.0)
        dac = DAC(bits=8, v_min=0.0, v_max=1.0)
        engine = SimulationEngine(crossbars=[xbar], adc=adc, dac=dac)
        V = self.rng.uniform(0, 1, 4)
        result = engine.run(V, mode="hardware")
        # Should produce valid output (not NaN)
        assert not np.any(np.isnan(result))
        assert result.shape == (3,)

    def test_run_comparison(self):
        xbar, _ = self._make_loaded_xbar(4, 3)
        engine = SimulationEngine(crossbars=[xbar])
        V = self.rng.uniform(0, 1, 4)
        results = engine.run_comparison(V, modes=["ideal", "device"])
        assert "ideal" in results
        assert "device" in results
        assert results["ideal"].shape == (3,)

    def test_add_crossbar(self):
        engine = SimulationEngine()
        assert len(engine.crossbars) == 0
        xbar, _ = self._make_loaded_xbar(4, 3)
        engine.add_crossbar(xbar)
        assert len(engine.crossbars) == 1

    def test_string_mode_parsing(self):
        xbar, _ = self._make_loaded_xbar(4, 3)
        engine = SimulationEngine(crossbars=[xbar])
        V = self.rng.uniform(0, 1, 4)
        # Should accept string modes
        result = engine.run(V, mode="IDEAL")
        assert result.shape == (3,)
