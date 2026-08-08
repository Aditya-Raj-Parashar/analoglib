"""End-to-end integration tests."""

import os
import tempfile

import numpy as np
import pytest

import analoglib as al


class TestEndToEnd:
    """Full pipeline: weights → crossbar → VMM → save → load → VMM."""

    def test_full_pipeline_ideal(self):
        """Ideal device should produce exact roundtrip."""
        al.set_seed(42)
        rng = np.random.default_rng(42)

        # 1. Create device and crossbar
        device = al.IdealDevice(g_min=0.0, g_max=1.0)
        mapping = al.DifferentialMapping(w_max=1.0)
        xbar = al.Crossbar(16, 8, device=device, mapping=mapping)

        # 2. Load weights
        W = rng.uniform(-1, 1, (16, 8))
        xbar.load_weights(W, quantize=False)

        # 3. Run VMM
        V = rng.uniform(0, 1, 16)
        out1 = xbar.vmm(V)
        expected = V @ W
        np.testing.assert_allclose(out1, expected, atol=1e-10)

        # 4. Save
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "pipeline.analog")
        al.save(path, [xbar], model_name="integration_test")

        # 5. Load
        result = al.load(path)
        loaded_xbar = result["crossbars"][0]

        # 6. VMM on loaded model
        out2 = loaded_xbar.vmm(V)
        np.testing.assert_allclose(out1, out2, atol=1e-12)

    def test_full_pipeline_reram(self):
        """ReRAM device — quantized, bounded error."""
        al.set_seed(123)
        rng = np.random.default_rng(123)

        # 1. Device
        device = al.ReRAM(
            g_min=1e-6, g_max=100e-6,
            num_states=256,
            read_noise_sigma=0.0,
        )
        mapping = al.DifferentialMapping(w_max=1.0)
        xbar = al.Crossbar(32, 16, device=device, mapping=mapping)

        # 2. Load weights (with quantization)
        W = rng.uniform(-1, 1, (32, 16))
        xbar.load_weights(W, quantize=True)

        # 3. Reconstruct weights — should be close
        W_recon = xbar.reconstruct_weights()
        max_err = np.max(np.abs(W - W_recon))
        assert max_err < 0.02  # < 2% for 256 levels

        # 4. VMM
        V = rng.uniform(0, 1, 32)
        out1 = xbar.vmm(V)

        # 5. Save/Load roundtrip
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "reram.analog")
        al.save(path, [xbar])
        result = al.load(path)
        loaded_xbar = result["crossbars"][0]

        # 6. VMM should be identical (same conductances, no noise)
        out2 = loaded_xbar.vmm(V)
        np.testing.assert_allclose(out1, out2, atol=1e-12)

    def test_simulation_engine_pipeline(self):
        """Multi-layer simulation engine with comparison."""
        al.set_seed(42)
        rng = np.random.default_rng(42)

        dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)
        mapping = al.DifferentialMapping(w_max=1.0)

        xbar1 = al.Crossbar(8, 4, device=dev, mapping=mapping)
        xbar1.load_weights(rng.uniform(-1, 1, (8, 4)))

        xbar2 = al.Crossbar(4, 2, device=dev, mapping=mapping)
        xbar2.load_weights(rng.uniform(-1, 1, (4, 2)))

        engine = al.SimulationEngine(crossbars=[xbar1, xbar2])
        V = rng.uniform(0, 1, 8)

        # Run in ideal mode
        out = engine.run(V, mode="ideal")
        assert out.shape == (2,)
        assert not np.any(np.isnan(out))

        # Comparison across modes
        results = engine.run_comparison(V)
        assert "ideal" in results
        assert results["ideal"].shape == (2,)

    def test_api_matches_spec_example(self):
        """Verify the API matches the spec's quick-start example."""
        import analoglib as al

        device = al.devices.ReRAM(
            g_min=1e-6,
            g_max=100e-6,
            num_states=256,
        )

        crossbar = al.Crossbar(
            rows=128,
            cols=64,
            device=device,
            differential=True,
        )

        W = np.random.default_rng(42).uniform(-1, 1, (128, 64))
        crossbar.load_weights(W)

        V = np.random.default_rng(42).uniform(0, 1, 128)
        output = crossbar.vmm(V)

        assert output.shape == (64,)
        assert not np.any(np.isnan(output))
