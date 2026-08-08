# AnalogLib — Analog In-Memory Computing Library

[![PyPI Version](https://img.shields.io/pypi/v/analoglib.svg)](https://pypi.org/project/analoglib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-v0.1.0-blue)](https://github.com/Aditya-Raj-Parashar/analoglib)

**AnalogLib** is an open-source Python library for simulating **analog in-memory computing (IMC)** and neural network inference on resistive crossbar architectures (ReRAM, PCM, and memristive arrays).

It models Ohm's Law ($I = V \cdot G$) and Kirchhoff's Current Law array execution, mapping high-level ML models (NumPy, PyTorch) onto physical crossbar arrays while accounting for device state quantization, read noise, wire IR drop, thermal scaling, retention drift, and peripheral ADC/DAC converters.

---

## ⚡ 5-Minute Quickstart

```python
import analoglib as al
import numpy as np

# 1. Define model weights
W1 = np.random.uniform(-0.5, 0.5, (128, 64))
W2 = np.random.uniform(-0.5, 0.5, (64, 10))

# 2. Build AnalogModel via AIR (Analog Intermediate Representation)
model = al.AnalogModel.from_numpy([W1, W2], activations=["relu", "softmax"])

# 3. Target physical ReRAM crossbars + 8-bit peripherals
model.compile(
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01),
    adc_bits=8,
    dac_bits=8,
)

# 4. Simulate hardware inference
x_input = np.random.uniform(0.0, 1.0, 128)
result = model.simulate(x_input, mode="hardware")

# 5. Print hardware metrics report
result.report()
```

---

## 📊 v0.1.0 Capability Matrix

| Component | Supported in v0.1.0 | Planned / Roadmap |
| :--- | :--- | :--- |
| **Frameworks** | ✅ **NumPy**, ✅ **PyTorch (`nn.Module`)** | 🚧 TensorFlow / Keras (v0.2.0), 🚧 ONNX (v0.3.0) |
| **Device Models** | ✅ **IdealDevice**, ✅ **ReRAM** | 🚧 Non-linear J4 memristor model |
| **Mapping** | ✅ **Differential ($G^+, G^-$)**, ✅ **Offset ($G_{\text{mid}}$)** | 🚧 Multi-bit cell slicing |
| **Array Architecture** | ✅ **Crossbar**, ✅ **TiledCrossbar** | 🚧 3D Crossbar stack |
| **Peripherals** | ✅ **DAC**, ✅ **ADC** | 🚧 Non-uniform logarithmic ADC |
| **Simulation Modes** | ✅ **"ideal"**, ✅ **"device"**, ✅ **"hardware"** | 🚧 Transient pulse-based simulation |
| **Physical Effects** | ✅ **IRDrop**, ✅ **Thermal**, ✅ **Drift** | 🚧 MNA full nodal solver |
| **Exporters & Format** | ✅ **Encrypted `.analog`**, ✅ **SPICE (ngspice/LTspice)** | 🚧 Verilog-A behavioral model export |
| **Analytics & CLI** | ✅ **AnalogProfiler**, ✅ **`analog` CLI** | 🚧 Graphical dashboard UI |

---

## 📦 Installation

```bash
# Base package
pip install analoglib

# With PyTorch support
pip install "analoglib[torch]"

# With Visualization support
pip install "analoglib[viz]"

# Full developer installation
pip install "analoglib[all]"
```

---

## 📖 Complete Documentation System

Explore the full documentation system:

- 🚀 **[Getting Started](docs/getting-started/quickstart.md)** — Installation, Quickstart, First VMM, Core Concepts
- 🧠 **[Physics & Equations](docs/concepts/analog-computing.md)** — Conductance mapping, state quantization, device noise, hardware modes
- 🔬 **[Capabilities Matrix](docs/capabilities.md)** — Full v0.1.0 compatibility breakdown
- ❓ **[FAQ & AI Discoverability](docs/faq.md)** — Answers to PyTorch, Keras, VMM, SPICE, and ReRAM questions
- ⚡ **[Framework Converters](docs/models/pytorch.md)** — PyTorch (`from_torch`) & NumPy (`from_numpy`) guides
- 🔒 **[File Format Spec](docs/serialization/analog-format.md)** — Encrypted `.analog` binary format layout
- 🔌 **[SPICE Exporter](docs/exporters/spice.md)** — ngspice & LTspice netlist generation
- 📊 **[Analytics & Profiler](docs/analytics/profiler.md)** — TOPS/W, power, energy, area estimation
- 🛠️ **[Troubleshooting Reference](docs/troubleshooting/common-errors.md)** — Error codes and fix guide
- 📚 **[API Reference Index](docs/api/index.md)** — Complete module, class, and method index
- 🗺️ **[Release Roadmap](docs/roadmap/roadmap.md)** — Feature roadmap for v0.2.0 and beyond

---

## 📜 License

AnalogLib is released under the [MIT License](LICENSE).
