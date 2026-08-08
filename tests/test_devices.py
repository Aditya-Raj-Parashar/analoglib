"""Tests for device models (IdealDevice, ReRAM)."""

import numpy as np
import pytest

import analoglib as al
from analoglib.devices import Device, IdealDevice, ReRAM


# -----------------------------------------------------------------------
# Device registry
# -----------------------------------------------------------------------

class TestDeviceRegistry:
    def test_reram_registered(self):
        assert "ReRAM" in Device.registry()

    def test_ideal_registered(self):
        assert "IdealDevice" in Device.registry()

    def test_get_by_name(self):
        assert Device.get("ReRAM") is ReRAM

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="No .* registered"):
            Device.get("FakeDevice")


# -----------------------------------------------------------------------
# IdealDevice
# -----------------------------------------------------------------------

class TestIdealDevice:
    def setup_method(self):
        self.dev = IdealDevice(g_min=0.0, g_max=1.0)

    def test_g_range(self):
        assert self.dev.g_range == 1.0

    def test_no_quantization(self):
        g = np.array([0.123456789, 0.5, 0.999])
        result = self.dev.quantize(g)
        np.testing.assert_array_almost_equal(result, g)

    def test_clamps_to_bounds(self):
        g = np.array([-0.5, 0.5, 1.5])
        result = self.dev.quantize(g)
        np.testing.assert_array_almost_equal(result, [0.0, 0.5, 1.0])

    def test_no_noise(self):
        g = np.array([0.3, 0.7])
        result = self.dev.add_noise(g)
        np.testing.assert_array_equal(result, g)

    def test_no_variation(self):
        g = np.array([0.3, 0.7])
        result = self.dev.add_variation(g)
        np.testing.assert_array_equal(result, g)

    def test_continuous_no_levels(self):
        levels = self.dev.conductance_levels()
        assert len(levels) == 0


# -----------------------------------------------------------------------
# ReRAM
# -----------------------------------------------------------------------

class TestReRAM:
    def setup_method(self):
        self.dev = ReRAM(
            g_min=1e-6,
            g_max=100e-6,
            num_states=256,
            read_noise_sigma=0.0,
            programming_error_sigma=0.0,
        )

    def test_conductance_levels_count(self):
        levels = self.dev.conductance_levels()
        assert len(levels) == 256

    def test_conductance_levels_range(self):
        levels = self.dev.conductance_levels()
        assert levels[0] == pytest.approx(1e-6)
        assert levels[-1] == pytest.approx(100e-6)

    def test_quantize_snaps_to_levels(self):
        levels = self.dev.conductance_levels()
        # Value between level 0 and level 1
        mid = (levels[0] + levels[1]) / 2.0
        result = self.dev.quantize(np.array([mid]))
        # Should snap to one of the two nearest
        assert result[0] in (levels[0], levels[1])

    def test_quantize_clamps(self):
        result = self.dev.quantize(np.array([0.0, 1.0]))
        assert result[0] >= self.dev.g_min
        assert result[1] <= self.dev.g_max

    def test_quantize_output_in_bounds(self):
        rng = np.random.default_rng(42)
        g = rng.uniform(0, 200e-6, size=1000)
        result = self.dev.quantize(g)
        assert np.all(result >= self.dev.g_min)
        assert np.all(result <= self.dev.g_max)

    def test_noise_zero_sigma_is_identity(self):
        g = np.array([50e-6, 75e-6])
        result = self.dev.add_noise(g)
        np.testing.assert_array_equal(result, g)

    def test_noise_nonzero_sigma_alters_values(self):
        dev = ReRAM(g_min=1e-6, g_max=100e-6, num_states=256,
                    read_noise_sigma=0.05)
        al.set_seed(42)
        g = np.full(1000, 50e-6)
        result = dev.add_noise(g)
        # Should be different from input
        assert not np.allclose(result, g)
        # Should still be in bounds
        assert np.all(result >= dev.g_min)
        assert np.all(result <= dev.g_max)

    def test_variation_nonzero_alters(self):
        dev = ReRAM(g_min=1e-6, g_max=100e-6, num_states=256,
                    d2d_variation_sigma=0.05)
        al.set_seed(123)
        g = np.full(1000, 50e-6)
        result = dev.add_variation(g)
        assert not np.allclose(result, g)

    def test_stuck_at_faults(self):
        dev = ReRAM(g_min=1e-6, g_max=100e-6, num_states=256,
                    stuck_at_fault_rate=0.5)
        al.set_seed(99)
        g = np.full(10000, 50e-6)
        result = dev.add_variation(g)
        # Some should be stuck at min or max
        stuck_min = np.sum(result == dev.g_min)
        stuck_max = np.sum(result == dev.g_max)
        total_stuck = stuck_min + stuck_max
        # Expect roughly 50% to be stuck
        assert total_stuck > 2000  # at least 20%


# -----------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------

class TestDeviceValidation:
    def test_g_min_negative_raises(self):
        with pytest.raises(ValueError, match="g_min"):
            ReRAM(g_min=-1e-6, g_max=100e-6)

    def test_g_max_less_than_g_min_raises(self):
        with pytest.raises(ValueError, match="g_max"):
            ReRAM(g_min=100e-6, g_max=1e-6)

    def test_negative_states_raises(self):
        with pytest.raises(ValueError, match="num_states"):
            ReRAM(g_min=1e-6, g_max=100e-6, num_states=-1)


# -----------------------------------------------------------------------
# Serialization
# -----------------------------------------------------------------------

class TestDeviceSerialization:
    def test_reram_roundtrip(self):
        dev = ReRAM(g_min=1e-6, g_max=100e-6, num_states=128,
                    read_noise_sigma=0.02, d2d_variation_sigma=0.01)
        d = dev.to_dict()
        restored = Device.from_dict(d)
        assert isinstance(restored, ReRAM)
        assert restored.g_min == dev.g_min
        assert restored.g_max == dev.g_max
        assert restored.num_states == dev.num_states
        assert restored.read_noise_sigma == dev.read_noise_sigma

    def test_ideal_roundtrip(self):
        dev = IdealDevice(g_min=0.0, g_max=1.0)
        d = dev.to_dict()
        restored = Device.from_dict(d)
        assert isinstance(restored, IdealDevice)
