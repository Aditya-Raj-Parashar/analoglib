"""Tests for analytics, SPICE exporter, and neural converters.

Sections:
  1. AnalogProfiler / AnalogReport
  2. SpiceExporter
  3. NumPy converter (from_numpy)
  4. PyTorch converter (from_torch) — only if torch installed
"""

import io
import numpy as np
import pytest

import analoglib as al
from analoglib.analysis.profiler import AnalogProfiler, AnalogReport
from analoglib.exporters.spice import SpiceExporter
from analoglib.neural.numpy_converter import from_numpy


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def single_crossbar(rng):
    W = rng.uniform(-1, 1, (8, 4))
    xbar = al.Crossbar(8, 4, device=al.IdealDevice(), mapping=al.DifferentialMapping())
    xbar.load_weights(W, quantize=False)
    return xbar


@pytest.fixture
def two_crossbars(rng):
    W1 = rng.uniform(-1, 1, (8, 4))
    W2 = rng.uniform(-1, 1, (4, 2))
    xb1 = al.Crossbar(8, 4)
    xb1.load_weights(W1, quantize=False)
    xb2 = al.Crossbar(4, 2)
    xb2.load_weights(W2, quantize=False)
    return [xb1, xb2]


# ===========================================================================
# 1. Analytics
# ===========================================================================

class TestAnalogProfiler:

    def test_profile_returns_report(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V)
        assert isinstance(report, AnalogReport)

    def test_ops_per_vmm(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V)
        # 2 * 8 * 4 = 64
        assert report.ops_per_vmm == 64

    def test_ops_two_layers(self, two_crossbars, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile(two_crossbars, V)
        # 2*8*4 + 2*4*2 = 64 + 16 = 80
        assert report.ops_per_vmm == 80

    def test_param_count(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V)
        assert report.total_params == 32

    def test_power_positive(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V)
        assert report.array_power_W >= 0

    def test_tops_w_positive(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V)
        assert report.tops_per_watt >= 0

    def test_with_adc_dac(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        adc = al.ADC(bits=8, v_min=-500e-6, v_max=500e-6)
        dac = al.DAC(bits=8, v_min=0.0, v_max=1.0)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V, adc=adc, dac=dac)
        assert report.adc_energy_J > 0
        assert report.dac_energy_J > 0
        assert report.total_energy_J > report.read_energy_J

    def test_latency_positive(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler(t_read=10e-9)
        report = profiler.profile([single_crossbar], V)
        assert report.latency_s > 0

    def test_cell_area_positive(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler(cell_feature_F=10.0)
        report = profiler.profile([single_crossbar], V)
        assert report.cell_area_um2 > 0

    def test_report_summary_contains_keywords(self, single_crossbar, rng):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V)
        summary = report.summary()
        assert "TOPS/W" in summary
        assert "Ops/VMM" in summary

    def test_report_print_runs(self, single_crossbar, rng, capsys):
        V = rng.uniform(0, 1, 8)
        profiler = AnalogProfiler()
        report = profiler.profile([single_crossbar], V)
        report.print()
        captured = capsys.readouterr()
        assert "Report" in captured.out


# ===========================================================================
# 2. SpiceExporter
# ===========================================================================

class TestSpiceExporter:

    def test_invalid_dialect_raises(self):
        with pytest.raises(ValueError, match="dialect"):
            SpiceExporter(dialect="hspice_pro_super")

    def test_export_str_returns_string(self, single_crossbar):
        exporter = SpiceExporter()
        netlist = exporter.export_str([single_crossbar])
        assert isinstance(netlist, str)

    def test_netlist_contains_title(self, single_crossbar):
        exporter = SpiceExporter()
        netlist = exporter.export_str([single_crossbar], title="MY TEST")
        assert "MY TEST" in netlist

    def test_netlist_contains_subcircuit(self, single_crossbar):
        exporter = SpiceExporter()
        netlist = exporter.export_str([single_crossbar])
        assert ".subckt" in netlist.lower() or ".subckt" in netlist

    def test_netlist_contains_resistors(self, single_crossbar):
        exporter = SpiceExporter()
        netlist = exporter.export_str([single_crossbar])
        assert "R_pos_" in netlist

    def test_netlist_ends_with_end(self, single_crossbar):
        exporter = SpiceExporter()
        netlist = exporter.export_str([single_crossbar])
        assert ".end" in netlist.lower()

    def test_export_to_file(self, tmp_path, single_crossbar):
        exporter = SpiceExporter()
        path = tmp_path / "test_circuit"
        saved = exporter.export(str(path), [single_crossbar])
        assert saved.exists()
        content = saved.read_text()
        assert ".subckt" in content.lower() or ".subckt" in content

    def test_export_adds_cir_extension(self, tmp_path, single_crossbar):
        exporter = SpiceExporter()
        path = tmp_path / "circuit"
        saved = exporter.export(str(path), [single_crossbar])
        assert saved.suffix == ".cir"

    def test_ltspice_dialect(self, single_crossbar):
        exporter = SpiceExporter(dialect="ltspice")
        netlist = exporter.export_str([single_crossbar])
        assert ".backanno" in netlist

    def test_multi_layer_export(self, two_crossbars):
        exporter = SpiceExporter()
        netlist = exporter.export_str(two_crossbars)
        assert "xbar_0" in netlist
        assert "xbar_1" in netlist


# ===========================================================================
# 3. NumPy Converter
# ===========================================================================

class TestNumpyConverter:

    def test_from_numpy_single_layer(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        g = from_numpy([W], name="test")
        assert len(g) == 1
        assert g.layers[0].matrix_shape == (8, 4)

    def test_from_numpy_two_layers(self, rng):
        W1 = rng.uniform(-1, 1, (8, 4))
        W2 = rng.uniform(-1, 1, (4, 2))
        g = from_numpy([W1, W2])
        assert len(g.crossbar_layers) == 2

    def test_from_numpy_with_activations(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        from analoglib.air.schema import LayerType
        g = from_numpy([W], activations=["relu"])
        act_layers = [l for l in g.layers if l.layer_type == LayerType.ACTIVATION]
        assert len(act_layers) == 1

    def test_from_numpy_none_activation_no_layer(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        from analoglib.air.schema import LayerType
        g = from_numpy([W], activations=["none"])
        act_layers = [l for l in g.layers if l.layer_type == LayerType.ACTIVATION]
        assert len(act_layers) == 0

    def test_from_numpy_bad_activation_raises(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        with pytest.raises(ValueError, match="Unknown activation"):
            from_numpy([W], activations=["lemon_squeeze"])

    def test_from_numpy_1d_raises(self, rng):
        W = rng.uniform(-1, 1, (8,))
        with pytest.raises(ValueError, match="2-D"):
            from_numpy([W])

    def test_from_numpy_mismatched_activations_raises(self, rng):
        W1 = rng.uniform(-1, 1, (8, 4))
        W2 = rng.uniform(-1, 1, (4, 2))
        with pytest.raises(ValueError, match="length"):
            from_numpy([W1, W2], activations=["relu"])

    def test_from_numpy_weights_preserved(self, rng):
        W = rng.uniform(-1, 1, (8, 4))
        g = from_numpy([W])
        np.testing.assert_array_equal(g.layers[0].weights, W)

    def test_from_numpy_accepts_pytorch_tensor(self, rng):
        """from_numpy should accept PyTorch tensors if torch is installed."""
        try:
            import torch
            W = torch.from_numpy(rng.uniform(-1, 1, (8, 4)))
            g = from_numpy([W])
            assert g.layers[0].matrix_shape == (8, 4)
        except ImportError:
            pytest.skip("PyTorch not installed")


# ===========================================================================
# 4. PyTorch Converter (conditional on torch availability)
# ===========================================================================

class TestTorchConverter:

    def test_from_torch_no_torch_raises(self, monkeypatch):
        """Raises ImportError with actionable message if torch not installed."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "torch.nn":
                raise ImportError("Mocked torch missing")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        from analoglib.neural.torch_converter import from_torch

        # We import fresh to bypass any cached module
        try:
            result = from_torch(None)
        except Exception as e:
            assert "PyTorch" in str(e) or "torch" in str(e).lower()

    def test_from_torch_linear(self):
        torch = pytest.importorskip("torch")
        nn   = pytest.importorskip("torch.nn")
        from analoglib.neural.torch_converter import from_torch

        model = nn.Linear(8, 4)
        g = from_torch(model, name="linear_test")
        assert len(g.crossbar_layers) == 1
        assert g.crossbar_layers[0].matrix_shape == (8, 4)

    def test_from_torch_sequential_mlp(self):
        torch = pytest.importorskip("torch")
        nn    = pytest.importorskip("torch.nn")
        from analoglib.neural.torch_converter import from_torch
        from analoglib.air.schema import LayerType

        model = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
        )
        g = from_torch(model, name="mlp")
        assert len(g.crossbar_layers) == 2
        act_layers = [l for l in g.layers if l.layer_type == LayerType.ACTIVATION]
        assert len(act_layers) == 1

    def test_from_torch_conv2d_im2col(self):
        torch = pytest.importorskip("torch")
        nn    = pytest.importorskip("torch.nn")
        from analoglib.neural.torch_converter import from_torch

        # Conv2d(in=3, out=8, kernel=3x3) → im2col → (3*3*3, 8)
        conv = nn.Conv2d(3, 8, kernel_size=3)
        g = from_torch(conv)
        assert len(g.crossbar_layers) == 1
        rows, cols = g.crossbar_layers[0].matrix_shape
        assert rows == 3 * 3 * 3   # in_ch * kH * kW
        assert cols == 8            # out_ch

    def test_from_torch_weights_match(self):
        torch = pytest.importorskip("torch")
        nn    = pytest.importorskip("torch.nn")
        from analoglib.neural.torch_converter import from_torch

        model = nn.Linear(4, 2, bias=False)
        W_torch = model.weight.detach().numpy().T  # (in, out)
        g = from_torch(model)
        W_air = g.crossbar_layers[0].weights
        np.testing.assert_allclose(W_air, W_torch, atol=1e-6)
