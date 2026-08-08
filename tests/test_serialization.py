"""Tests for .analog file serialization (encrypted format)."""

import os
import tempfile

import numpy as np
import pytest

import analoglib as al
from analoglib.crossbar import Crossbar
from analoglib.devices import IdealDevice, ReRAM
from analoglib.mapping import DifferentialMapping
from analoglib.serialization import save, load


class TestAnalogFormat:
    def setup_method(self):
        al.set_seed(42)
        self.tmpdir = tempfile.mkdtemp()

    def _make_loaded_xbar(self, device=None, rows=8, cols=4):
        dev = device or IdealDevice(0.0, 1.0)
        mapping = DifferentialMapping(w_max=1.0)
        xbar = Crossbar(rows, cols, device=dev, mapping=mapping)
        rng = np.random.default_rng(42)
        W = rng.uniform(-1, 1, (rows, cols))
        xbar.load_weights(W, quantize=False)
        return xbar, W

    def test_save_creates_file(self):
        xbar, _ = self._make_loaded_xbar()
        path = os.path.join(self.tmpdir, "test.analog")
        result_path = save(path, [xbar])
        assert os.path.exists(result_path)
        assert os.path.getsize(result_path) > 0

    def test_file_starts_with_magic(self):
        xbar, _ = self._make_loaded_xbar()
        path = os.path.join(self.tmpdir, "test.analog")
        save(path, [xbar])
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\xAE\x4C\x49\x42"

    def test_file_not_readable_as_text(self):
        """Verify the file is not human-readable."""
        xbar, _ = self._make_loaded_xbar()
        path = os.path.join(self.tmpdir, "test.analog")
        save(path, [xbar])
        with open(path, "rb") as f:
            content = f.read()
        # Should not contain readable JSON or plaintext metadata
        assert b'"format_version"' not in content
        assert b'"model_name"' not in content

    def test_save_load_roundtrip(self):
        xbar, W = self._make_loaded_xbar()
        path = os.path.join(self.tmpdir, "roundtrip.analog")
        save(path, [xbar], model_name="test_model", description="unit test")
        result = load(path)

        assert result["meta"]["model_name"] == "test_model"
        assert result["meta"]["description"] == "unit test"
        assert len(result["crossbars"]) == 1

        loaded_xbar = result["crossbars"][0]
        assert loaded_xbar.rows == 8
        assert loaded_xbar.cols == 4
        assert loaded_xbar.differential is True

    def test_conductance_preserved(self):
        dev = ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)
        xbar, _ = self._make_loaded_xbar(device=dev)
        path = os.path.join(self.tmpdir, "conductance.analog")
        save(path, [xbar])
        result = load(path)

        loaded_xbar = result["crossbars"][0]
        g_orig = xbar.get_conductance()
        g_loaded = loaded_xbar.get_conductance()

        np.testing.assert_array_equal(g_orig[0], g_loaded[0])
        np.testing.assert_array_equal(g_orig[1], g_loaded[1])

    def test_vmm_preserved(self):
        """VMM results should be identical before and after save/load."""
        xbar, _ = self._make_loaded_xbar()
        V = np.random.default_rng(99).uniform(0, 1, 8)
        out_before = xbar.vmm(V)

        path = os.path.join(self.tmpdir, "vmm.analog")
        save(path, [xbar])
        result = load(path)
        loaded_xbar = result["crossbars"][0]
        out_after = loaded_xbar.vmm(V)

        np.testing.assert_allclose(out_before, out_after, atol=1e-12)

    def test_multi_layer_roundtrip(self):
        xbar1, _ = self._make_loaded_xbar(rows=8, cols=4)
        xbar2, _ = self._make_loaded_xbar(rows=4, cols=2)
        path = os.path.join(self.tmpdir, "multi.analog")
        save(path, [xbar1, xbar2])
        result = load(path)
        assert len(result["crossbars"]) == 2
        assert result["crossbars"][0].rows == 8
        assert result["crossbars"][1].rows == 4

    def test_invalid_file_raises(self):
        path = os.path.join(self.tmpdir, "fake.analog")
        with open(path, "wb") as f:
            f.write(b"not a real analog file")
        with pytest.raises(ValueError, match="Not a valid"):
            load(path)

    def test_auto_extension(self):
        xbar, _ = self._make_loaded_xbar()
        path = os.path.join(self.tmpdir, "noext")
        result_path = save(path, [xbar])
        assert str(result_path).endswith(".analog")

    def test_extra_meta_preserved(self):
        xbar, _ = self._make_loaded_xbar()
        path = os.path.join(self.tmpdir, "meta.analog")
        save(path, [xbar], extra_meta={"experiment": "test_42", "seed": 42})
        result = load(path)
        assert result["meta"]["experiment"] == "test_42"
        assert result["meta"]["seed"] == 42
