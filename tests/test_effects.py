"""Tests for all hardware effects: IRDrop, Thermal, Drift.

Each test class covers:
  - Construction and repr
  - apply() output shape preservation
  - Physical correctness (direction of effect)
  - Edge cases (zero params, extreme temps)
  - to_dict / from_dict roundtrip via Effect.from_dict
"""

import numpy as np
import pytest

import analoglib as al
from analoglib.effects.base import Effect, EffectContext
from analoglib.effects.ir_drop import IRDrop
from analoglib.effects.thermal import Thermal
from analoglib.effects.drift import Drift


@pytest.fixture
def sample_G():
    rng = np.random.default_rng(42)
    return rng.uniform(1e-6, 100e-6, (8, 4))


@pytest.fixture
def sample_ctx(sample_G):
    return EffectContext(
        V_row=np.ones(8) * 0.5,
        G=sample_G,
        T_kelvin=300.0,
        t_seconds=0.0,
    )


class TestEffectRegistry:

    def test_irdrop_registered(self):
        assert "IRDrop" in Effect.registry()

    def test_thermal_registered(self):
        assert "Thermal" in Effect.registry()

    def test_drift_registered(self):
        assert "Drift" in Effect.registry()

    def test_get_by_name(self):
        klass = Effect.get("IRDrop")
        assert klass is IRDrop

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown effect"):
            Effect.get("Nonexistent")


class TestIRDrop:

    def test_creation(self):
        e = IRDrop(r_wire=1.0)
        assert e.r_wire == 1.0

    def test_negative_rwire_raises(self):
        with pytest.raises(ValueError, match="r_wire"):
            IRDrop(r_wire=-1.0)

    def test_apply_output_shape(self, sample_G, sample_ctx):
        e = IRDrop(r_wire=2.0)
        result = e.apply(sample_G, sample_ctx)
        assert result.shape == sample_G.shape

    def test_zero_rwire_identity(self, sample_G, sample_ctx):
        e = IRDrop(r_wire=0.0)
        result = e.apply(sample_G, sample_ctx)
        np.testing.assert_array_equal(result, sample_G)

    def test_irdrop_reduces_conductance(self, sample_G, sample_ctx):
        """IR drop should always reduce effective conductance."""
        e = IRDrop(r_wire=5.0)
        result = e.apply(sample_G, sample_ctx)
        assert np.all(result <= sample_G + 1e-15)

    def test_irdrop_bounded_above_zero(self, sample_G, sample_ctx):
        e = IRDrop(r_wire=100.0)
        result = e.apply(sample_G, sample_ctx)
        assert np.all(result >= 0)

    def test_serialization_roundtrip(self):
        e = IRDrop(r_wire=2.5)
        d = e.to_dict()
        assert d["type"] == "IRDrop"
        assert d["r_wire"] == 2.5
        e2 = Effect.from_dict(d)
        assert isinstance(e2, IRDrop)
        assert e2.r_wire == 2.5

    def test_repr(self):
        e = IRDrop(r_wire=3.0)
        assert "IRDrop" in repr(e)
        assert "3.0" in repr(e)


class TestThermal:

    def test_creation(self):
        e = Thermal(E_a=0.2, T_ref=300.0)
        assert e.E_a == 0.2

    def test_negative_Ea_raises(self):
        with pytest.raises(ValueError, match="E_a"):
            Thermal(E_a=-1.0)

    def test_nonpositive_Tref_raises(self):
        with pytest.raises(ValueError, match="T_ref"):
            Thermal(T_ref=0.0)

    def test_apply_output_shape(self, sample_G, sample_ctx):
        e = Thermal(E_a=0.1)
        result = e.apply(sample_G, sample_ctx)
        assert result.shape == sample_G.shape

    def test_at_reference_temperature_identity(self, sample_G):
        e = Thermal(E_a=0.1, T_ref=300.0)
        ctx = EffectContext(V_row=np.ones(8), G=sample_G, T_kelvin=300.0)
        result = e.apply(sample_G, ctx)
        np.testing.assert_array_equal(result, sample_G)

    def test_high_temperature_increases_G(self, sample_G):
        """Higher T → higher G (thermal activation)."""
        e = Thermal(E_a=0.1, T_ref=300.0)
        ctx_hot = EffectContext(V_row=np.ones(8), G=sample_G, T_kelvin=400.0)
        result_hot = e.apply(sample_G, ctx_hot)
        assert np.all(result_hot > sample_G - 1e-20)

    def test_low_temperature_decreases_G(self, sample_G):
        """Lower T → lower G."""
        e = Thermal(E_a=0.1, T_ref=300.0)
        ctx_cold = EffectContext(V_row=np.ones(8), G=sample_G, T_kelvin=200.0)
        result_cold = e.apply(sample_G, ctx_cold)
        assert np.all(result_cold < sample_G + 1e-20)

    def test_serialization_roundtrip(self):
        e = Thermal(E_a=0.15, T_ref=320.0)
        d = e.to_dict()
        assert d["type"] == "Thermal"
        e2 = Effect.from_dict(d)
        assert isinstance(e2, Thermal)
        assert e2.E_a == 0.15
        assert e2.T_ref == 320.0


class TestDrift:

    def test_creation(self):
        e = Drift(nu=0.05, t_0=1.0)
        assert e.nu == 0.05

    def test_negative_nu_raises(self):
        with pytest.raises(ValueError, match="nu"):
            Drift(nu=-0.1)

    def test_nonpositive_t0_raises(self):
        with pytest.raises(ValueError, match="t_0"):
            Drift(t_0=0.0)

    def test_apply_output_shape(self, sample_G, sample_ctx):
        e = Drift(nu=0.05)
        ctx = EffectContext(V_row=np.ones(8), G=sample_G, t_seconds=100.0)
        result = e.apply(sample_G, ctx)
        assert result.shape == sample_G.shape

    def test_zero_time_identity(self, sample_G):
        """At t=0, no drift should occur."""
        e = Drift(nu=0.05)
        ctx = EffectContext(V_row=np.ones(8), G=sample_G, t_seconds=0.0)
        result = e.apply(sample_G, ctx)
        np.testing.assert_array_equal(result, sample_G)

    def test_zero_nu_identity(self, sample_G):
        """nu=0 means no drift regardless of time."""
        e = Drift(nu=0.0)
        ctx = EffectContext(V_row=np.ones(8), G=sample_G, t_seconds=1000.0)
        result = e.apply(sample_G, ctx)
        np.testing.assert_array_equal(result, sample_G)

    def test_drift_decreases_conductance(self, sample_G):
        """Drift always reduces G for t > t_0."""
        e = Drift(nu=0.05, t_0=1.0)
        ctx = EffectContext(V_row=np.ones(8), G=sample_G, t_seconds=1000.0)
        result = e.apply(sample_G, ctx)
        assert np.all(result <= sample_G + 1e-20)

    def test_larger_nu_more_drift(self, sample_G):
        """Higher nu exponent → greater conductance loss."""
        ctx = EffectContext(V_row=np.ones(8), G=sample_G, t_seconds=100.0)
        e_small = Drift(nu=0.01)
        e_large = Drift(nu=0.20)
        r_small = e_small.apply(sample_G, ctx)
        r_large = e_large.apply(sample_G, ctx)
        assert np.all(r_large <= r_small + 1e-20)

    def test_serialization_roundtrip(self):
        e = Drift(nu=0.03, t_0=5.0)
        d = e.to_dict()
        assert d["type"] == "Drift"
        e2 = Effect.from_dict(d)
        assert isinstance(e2, Drift)
        assert e2.nu == 0.03
        assert e2.t_0 == 5.0
