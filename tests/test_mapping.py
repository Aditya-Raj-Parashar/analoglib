"""Tests for mapping strategies."""

import numpy as np
import pytest

import analoglib as al
from analoglib.devices import IdealDevice, ReRAM
from analoglib.mapping import DifferentialMapping, OffsetMapping, MappingStrategy


# -----------------------------------------------------------------------
# DifferentialMapping
# -----------------------------------------------------------------------

class TestDifferentialMapping:
    def setup_method(self):
        self.dev = IdealDevice(g_min=0.0, g_max=1.0)
        self.mapping = DifferentialMapping()

    def test_positive_weight_g_pos_gt_g_neg(self):
        W = np.array([[0.5]])
        g_pos, g_neg = self.mapping.weights_to_conductance(W, self.dev)
        assert g_pos[0, 0] > g_neg[0, 0]

    def test_negative_weight_g_pos_lt_g_neg(self):
        W = np.array([[-0.5]])
        g_pos, g_neg = self.mapping.weights_to_conductance(W, self.dev)
        assert g_pos[0, 0] < g_neg[0, 0]

    def test_zero_weight_g_pos_eq_g_neg(self):
        W = np.array([[0.0]])
        g_pos, g_neg = self.mapping.weights_to_conductance(W, self.dev)
        assert g_pos[0, 0] == pytest.approx(g_neg[0, 0])

    def test_conductance_in_bounds(self):
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, size=(64, 32))
        g_pos, g_neg = self.mapping.weights_to_conductance(W, self.dev)
        assert np.all(g_pos >= self.dev.g_min)
        assert np.all(g_pos <= self.dev.g_max)
        assert np.all(g_neg >= self.dev.g_min)
        assert np.all(g_neg <= self.dev.g_max)

    def test_roundtrip_ideal(self):
        """W → G → W' should be nearly identical for ideal device."""
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, size=(16, 8))
        mapping = DifferentialMapping(w_max=1.0)
        g_pos, g_neg = mapping.weights_to_conductance(W, self.dev)
        W_reconstructed = mapping.conductance_to_weights(g_pos, g_neg, device=self.dev)
        np.testing.assert_allclose(W_reconstructed, W, atol=1e-10)

    def test_roundtrip_with_reram(self):
        """Roundtrip with quantized ReRAM should have bounded error."""
        dev = ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, size=(16, 8))
        mapping = DifferentialMapping(w_max=1.0)
        g_pos, g_neg = mapping.weights_to_conductance(W, dev)
        g_pos = dev.quantize(g_pos)
        g_neg = dev.quantize(g_neg)
        W_reconstructed = mapping.conductance_to_weights(g_pos, g_neg, device=dev)
        # Quantization error should be bounded
        max_error = np.max(np.abs(W - W_reconstructed))
        assert max_error < 0.02  # < 2% for 256 levels

    def test_w_max_auto_detection(self):
        """When w_max=None, infer from data."""
        W = np.array([[3.0, -2.0]])
        mapping = DifferentialMapping(w_max=None)
        g_pos, g_neg = mapping.weights_to_conductance(W, self.dev)
        # Largest weight (3.0) should map near g_max
        assert g_pos[0, 0] > 0.9 * self.dev.g_max


# -----------------------------------------------------------------------
# OffsetMapping
# -----------------------------------------------------------------------

class TestOffsetMapping:
    def setup_method(self):
        self.dev = IdealDevice(g_min=0.0, g_max=1.0)
        self.mapping = OffsetMapping(w_max=1.0)

    def test_zero_weight_maps_to_midpoint(self):
        W = np.array([[0.0]])
        (g,) = self.mapping.weights_to_conductance(W, self.dev)
        expected_mid = (self.dev.g_min + self.dev.g_max) / 2.0
        assert g[0, 0] == pytest.approx(expected_mid)

    def test_positive_weight_above_mid(self):
        W = np.array([[0.5]])
        (g,) = self.mapping.weights_to_conductance(W, self.dev)
        mid = (self.dev.g_min + self.dev.g_max) / 2.0
        assert g[0, 0] > mid

    def test_roundtrip_ideal(self):
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, size=(16, 8))
        (g,) = self.mapping.weights_to_conductance(W, self.dev)
        W_reconstructed = self.mapping.conductance_to_weights(g, device=self.dev)
        np.testing.assert_allclose(W_reconstructed, W, atol=1e-10)

    def test_wrong_arg_count_raises(self):
        g1 = np.zeros((4, 4))
        g2 = np.zeros((4, 4))
        with pytest.raises(ValueError, match="expects 1"):
            self.mapping.conductance_to_weights(g1, g2, device=self.dev)


# -----------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------

class TestMappingRegistry:
    def test_differential_registered(self):
        assert "DifferentialMapping" in MappingStrategy.registry()

    def test_offset_registered(self):
        assert "OffsetMapping" in MappingStrategy.registry()

    def test_from_dict_roundtrip(self):
        mapping = DifferentialMapping(w_max=2.5)
        d = mapping.to_dict()
        restored = MappingStrategy.from_dict(d)
        assert isinstance(restored, DifferentialMapping)
        assert restored.w_max == 2.5
