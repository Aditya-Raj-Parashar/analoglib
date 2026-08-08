"""Tests for Phase 3: TiledCrossbar.

Tests:
  1. TiledCrossbar creation and grid calculation
  2. from_matrix factory
  3. Invalid dimensions raise errors
  4. KEY INVARIANT: TiledCrossbar.vmm == Crossbar.vmm (ideal, unquantized)
  5. Batch VMM
  6. Partial tile (non-divisible shapes)
  7. load_weights shape mismatch raises error
  8. Weight reconstruction from tiles
  9. tile_grid_info
  10. n_tiles property
"""

import numpy as np
import pytest

import analoglib as al
from analoglib.crossbar.tiled import TiledCrossbar


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestTiledCrossbarCreation:

    def test_basic_creation(self):
        tc = TiledCrossbar(64, 32, tile_rows=16, tile_cols=8)
        assert tc.rows == 64
        assert tc.cols == 32
        assert tc.n_tile_rows == 4
        assert tc.n_tile_cols == 4

    def test_tile_grid_exact_divisible(self):
        tc = TiledCrossbar(128, 64, tile_rows=64, tile_cols=32)
        assert tc.n_tile_rows == 2
        assert tc.n_tile_cols == 2

    def test_tile_grid_non_divisible(self):
        # 10 / 3 = ceil → 4 tiles high, 7 / 4 = ceil → 2 tiles wide
        tc = TiledCrossbar(10, 7, tile_rows=3, tile_cols=4)
        assert tc.n_tile_rows == 4
        assert tc.n_tile_cols == 2

    def test_n_tiles(self):
        tc = TiledCrossbar(16, 16, tile_rows=8, tile_cols=8)
        assert tc.n_tiles == 4

    def test_invalid_rows_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TiledCrossbar(0, 4, tile_rows=2, tile_cols=2)

    def test_invalid_tile_rows_raises(self):
        with pytest.raises(ValueError, match="Tile"):
            TiledCrossbar(4, 4, tile_rows=0, tile_cols=2)

    def test_vmm_before_load_raises(self, rng):
        tc = TiledCrossbar(4, 4, tile_rows=2, tile_cols=2)
        with pytest.raises(RuntimeError, match="no weights"):
            tc.vmm(rng.uniform(0, 1, 4))

    def test_repr_empty(self):
        tc = TiledCrossbar(8, 4, tile_rows=4, tile_cols=2)
        r = repr(tc)
        assert "TiledCrossbar" in r
        assert "empty" in r


class TestTiledCrossbarFromMatrix:

    def test_from_matrix(self, rng):
        W = rng.uniform(-1, 1, (16, 8))
        tc = TiledCrossbar.from_matrix(W, tile_shape=(8, 4))
        assert tc.rows == 16
        assert tc.cols == 8

    def test_from_matrix_wrong_ndim(self, rng):
        W = rng.uniform(-1, 1, (16,))
        with pytest.raises(ValueError, match="2-D"):
            TiledCrossbar.from_matrix(W, tile_shape=(8, 4))

    def test_load_weights_wrong_shape(self, rng):
        W = rng.uniform(-1, 1, (4, 4))
        tc = TiledCrossbar(8, 4, tile_rows=4, tile_cols=2)
        with pytest.raises(ValueError, match="shape"):
            tc.load_weights(W)


class TestTiledCrossbarInvariant:
    """Primary invariant: TiledCrossbar.vmm == Crossbar.vmm in ideal mode."""

    def _make_ref(self, W, device, mapping):
        rows, cols = W.shape
        xbar = al.Crossbar(rows, cols, device=device, mapping=mapping)
        xbar.load_weights(W, quantize=False)
        return xbar

    def test_invariant_exact_tiles(self, rng):
        """16x8 matrix, 8x4 tiles → 2x2 grid."""
        W = rng.uniform(-1, 1, (16, 8))
        x = rng.uniform(0, 1, 16)

        device = al.IdealDevice()
        mapping = al.DifferentialMapping()

        tc = TiledCrossbar.from_matrix(W, tile_shape=(8, 4), device=device,
                                        mapping=mapping, quantize=False)
        ref = self._make_ref(W, device, mapping)

        out_tiled = tc.vmm(x)
        out_ref   = ref.vmm(x)
        np.testing.assert_allclose(out_tiled, out_ref, atol=1e-10)

    def test_invariant_non_divisible(self, rng):
        """13x7 matrix with 4x3 tiles (non-divisible)."""
        W = rng.uniform(-1, 1, (13, 7))
        x = rng.uniform(0, 1, 13)

        device = al.IdealDevice()
        mapping = al.DifferentialMapping()

        tc = TiledCrossbar.from_matrix(W, tile_shape=(4, 3), device=device,
                                        mapping=mapping, quantize=False)
        ref = self._make_ref(W, device, mapping)

        out_tiled = tc.vmm(x)
        out_ref   = ref.vmm(x)
        np.testing.assert_allclose(out_tiled, out_ref, atol=1e-10)

    def test_invariant_single_tile(self, rng):
        """Matrix fits in one tile — should be exact."""
        W = rng.uniform(-1, 1, (8, 4))
        x = rng.uniform(0, 1, 8)

        device = al.IdealDevice()
        tc  = TiledCrossbar.from_matrix(W, tile_shape=(16, 8), device=device, quantize=False)
        ref = self._make_ref(W, device, al.DifferentialMapping())

        np.testing.assert_allclose(tc.vmm(x), ref.vmm(x), atol=1e-10)

    def test_invariant_large(self, rng):
        """Large 128x64 matrix with 32x16 tiles."""
        W = rng.uniform(-1, 1, (128, 64))
        x = rng.uniform(0, 1, 128)

        device = al.IdealDevice()
        mapping = al.DifferentialMapping()

        tc  = TiledCrossbar.from_matrix(W, tile_shape=(32, 16), device=device,
                                         mapping=mapping, quantize=False)
        ref = self._make_ref(W, device, mapping)

        np.testing.assert_allclose(tc.vmm(x), ref.vmm(x), atol=1e-10)


class TestTiledCrossbarVMM:

    def test_output_shape_1d(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        x = rng.uniform(0, 1, 8)
        tc = TiledCrossbar.from_matrix(W, tile_shape=(4, 2))
        out = tc.vmm(x)
        assert out.shape == (4,)

    def test_output_shape_batch(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        x = rng.uniform(0, 1, (3, 8))
        tc = TiledCrossbar.from_matrix(W, tile_shape=(4, 2))
        out = tc.vmm(x)
        assert out.shape == (3, 4)

    def test_reram_device_vmm_runs(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        x = rng.uniform(0, 1, 8)
        reram = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=64)
        tc = TiledCrossbar.from_matrix(W, tile_shape=(4, 2), device=reram)
        out = tc.vmm(x, mode=al.SimulationMode.DEVICE)
        assert out.shape == (4,)


class TestTiledCrossbarReconstruct:

    def test_reconstruct_shape(self, rng):
        W = rng.uniform(-1, 1, (16, 8))
        tc = TiledCrossbar.from_matrix(W, tile_shape=(8, 4))
        W_recon = tc.reconstruct_weights()
        assert W_recon.shape == (16, 8)

    def test_reconstruct_ideal_no_quantize(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        device = al.IdealDevice()
        tc = TiledCrossbar.from_matrix(W, tile_shape=(4, 2),
                                        device=device, quantize=False)
        W_recon = tc.reconstruct_weights()
        np.testing.assert_allclose(W_recon, W, atol=1e-10)

    def test_tile_grid_info(self):
        tc = TiledCrossbar(128, 64, tile_rows=32, tile_cols=16)
        info = tc.tile_grid_info()
        assert info["n_tile_rows"] == 4
        assert info["n_tile_cols"] == 4
        assert info["n_tiles"] == 16
