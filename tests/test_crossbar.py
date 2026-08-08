"""Tests for the crossbar engine."""

import numpy as np
import pytest

import analoglib as al
from analoglib.crossbar import Crossbar
from analoglib.devices import IdealDevice, ReRAM
from analoglib.mapping import DifferentialMapping


class TestCrossbarCreation:
    def test_basic_creation(self):
        xbar = Crossbar(128, 64)
        assert xbar.rows == 128
        assert xbar.cols == 64
        assert xbar.differential is True

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError, match="positive"):
            Crossbar(0, 64)

    def test_default_device_is_ideal(self):
        xbar = Crossbar(4, 4)
        assert isinstance(xbar.device, IdealDevice)

    def test_default_mapping_is_differential(self):
        xbar = Crossbar(4, 4)
        assert isinstance(xbar.mapping, DifferentialMapping)


class TestCrossbarLoadWeights:
    def setup_method(self):
        self.dev = IdealDevice(g_min=0.0, g_max=1.0)
        self.xbar = Crossbar(4, 3, device=self.dev)

    def test_load_weights_shape(self):
        W = np.random.default_rng(42).uniform(-1, 1, (4, 3))
        self.xbar.load_weights(W)
        g_pos, g_neg = self.xbar.get_conductance()
        assert g_pos.shape == (4, 3)
        assert g_neg.shape == (4, 3)

    def test_wrong_shape_raises(self):
        W = np.zeros((5, 3))
        with pytest.raises(ValueError, match="doesn't match"):
            self.xbar.load_weights(W)

    def test_no_weights_raises_on_vmm(self):
        with pytest.raises(RuntimeError, match="no weights"):
            self.xbar.vmm(np.zeros(4))


class TestCrossbarVMM:
    def setup_method(self):
        self.dev = IdealDevice(g_min=0.0, g_max=1.0)
        self.rows, self.cols = 4, 3
        self.xbar = Crossbar(self.rows, self.cols, device=self.dev,
                             mapping=DifferentialMapping(w_max=1.0))

    def test_vmm_correctness_ideal(self):
        """VMM should match numpy matmul for ideal device."""
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, (self.rows, self.cols))
        V = rng.uniform(0, 1, self.rows)

        self.xbar.load_weights(W, quantize=False)
        result = self.xbar.vmm(V)

        # For ideal device with differential mapping:
        # I = V @ (G+ - G-) should be proportional to V @ W
        expected = V @ W
        # The proportionality factor depends on the mapping
        # For differential: G+ - G- = w_norm * g_range - (1-w_norm) * g_range
        #                            = (2*w_norm - 1) * g_range
        # where w_norm = (W + w_max) / (2*w_max)
        # So G+ - G- = W/w_max * g_range = W * 1.0 = W (for w_max=1, g_range=1)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_vmm_batch(self):
        """VMM should handle batch inputs."""
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, (self.rows, self.cols))
        V = rng.uniform(0, 1, (5, self.rows))  # batch of 5

        self.xbar.load_weights(W, quantize=False)
        result = self.xbar.vmm(V)

        assert result.shape == (5, self.cols)
        expected = V @ W
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_vmm_wrong_dim_raises(self):
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, (self.rows, self.cols))
        self.xbar.load_weights(W)
        with pytest.raises(ValueError, match="doesn't match"):
            self.xbar.vmm(np.zeros(99))

    def test_vmm_1d_returns_1d(self):
        W = np.eye(self.rows, self.cols)
        self.xbar.load_weights(W, quantize=False)
        V = np.ones(self.rows)
        result = self.xbar.vmm(V)
        assert result.ndim == 1
        assert result.shape == (self.cols,)


class TestCrossbarWeightReconstruction:
    def test_reconstruct_ideal(self):
        dev = IdealDevice(g_min=0.0, g_max=1.0)
        mapping = DifferentialMapping(w_max=1.0)
        xbar = Crossbar(8, 4, device=dev, mapping=mapping)

        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, (8, 4))
        xbar.load_weights(W, quantize=False)

        W_recon = xbar.reconstruct_weights()
        np.testing.assert_allclose(W_recon, W, atol=1e-10)

    def test_reconstruct_quantized_bounded_error(self):
        dev = ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)
        mapping = DifferentialMapping(w_max=1.0)
        xbar = Crossbar(8, 4, device=dev, mapping=mapping)

        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, (8, 4))
        xbar.load_weights(W, quantize=True)

        W_recon = xbar.reconstruct_weights()
        max_error = np.max(np.abs(W - W_recon))
        # 256 levels → error < 1/128 ≈ 0.008, but with double quantization ≈ 0.016
        assert max_error < 0.02


class TestCrossbarSerialization:
    def test_to_dict(self):
        dev = ReRAM(g_min=1e-6, g_max=100e-6)
        xbar = Crossbar(8, 4, device=dev, differential=True)
        d = xbar.to_dict()
        assert d["rows"] == 8
        assert d["cols"] == 4
        assert d["differential"] is True
        assert d["device"]["type"] == "ReRAM"
