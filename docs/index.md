# AnalogLib Documentation (v0.1.0)

[![PyPI Version](https://img.shields.io/pypi/v/analoglib.svg)](https://pypi.org/project/analoglib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Welcome to the official documentation for **AnalogLib**, an open-source Python library for simulating analog in-memory computing (IMC) and neural network inference on resistive crossbar architectures (ReRAM, PCM, Flash, and memristive arrays).

---

## What is AnalogLib?

AnalogLib bridges the gap between high-level machine learning frameworks and physical NVM (Non-Volatile Memory) hardware arrays. Rather than performing digital matrix multiplication ($y = xW$), AnalogLib models physical Ohm's Law and Kirchhoff's Current Law execution:

$$I_j = \sum_{i} V_i G_{i,j}$$

where weights are stored as physical cell conductances ($G$) and inputs are applied as row voltages ($V$).

---

## Core Capabilities Summary

| Area | Feature | Status in v0.1.0 | Key API |
| :--- | :--- | :--- | :--- |
| **Frameworks** | NumPy Weight Arrays | ✅ **Supported** | `al.neural.from_numpy()` / `AnalogModel.from_numpy()` |
| | PyTorch `nn.Module` | ✅ **Supported** | `al.neural.from_torch()` / `AnalogModel.from_torch()` |
| | TensorFlow / Keras | 🚧 **Planned** | See [Migration Guide](models/tensorflow.md) |
| | ONNX Models | 🚧 **Planned** | See [ONNX Roadmap](models/onnx.md) |
| **Devices** | Ideal Continuous Device | ✅ **Supported** | `al.IdealDevice()` |
| | Physical ReRAM Device | ✅ **Supported** | `al.ReRAM(g_min, g_max, num_states, read_noise_sigma)` |
| **Mapping** | Differential Pair ($G^+, G^-$) | ✅ **Supported** | `al.DifferentialMapping(w_max)` |
| | Offset Reference ($G_{mid}$) | ✅ **Supported** | `al.OffsetMapping(w_max)` |
| **Array Architecture** | Single Crossbar Engine | ✅ **Supported** | `al.Crossbar(rows, cols)` |
| | 2D Tiled Crossbar Grid | ✅ **Supported** | `al.TiledCrossbar.from_matrix(W, tile_shape)` |
| **Peripherals** | DAC / ADC Converters | ✅ **Supported** | `al.DAC(bits)`, `al.ADC(bits)` |
| **Simulation Modes** | Ideal / Device / Hardware | ✅ **Supported** | `engine.run(V, mode="hardware")` |
| **Physical Effects** | Wire IR Drop | ✅ **Supported** | `al.effects.IRDrop(r_wire)` |
| | Arrhenius Thermal Scaling | ✅ **Supported** | `al.effects.Thermal(E_a, T_ref)` |
| | Power-law Retention Drift | ✅ **Supported** | `al.effects.Drift(nu, t_0)` |
| **Exporters & Format** | Encrypted `.analog` Binary | ✅ **Supported** | `al.save()`, `al.load()` |
| | SPICE Netlist (ngspice/LTspice) | ✅ **Supported** | `al.exporters.SpiceExporter()` |
| | Analytics & Profiler | ✅ **Supported** | `al.analysis.AnalogProfiler()` |

---

## 5-Minute Quickstart Example

```python
import analoglib as al
import numpy as np

# 1. Prepare weight matrix (or PyTorch nn.Module)
W1 = np.random.uniform(-1.0, 1.0, (128, 64))
W2 = np.random.uniform(-1.0, 1.0, (64, 10))

# 2. Build AnalogModel via AIR (Analog Intermediate Representation)
model = al.AnalogModel.from_numpy([W1, W2], activations=["relu", "softmax"])

# 3. Target physical ReRAM crossbars + ADC/DAC
model.compile(
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01),
    adc_bits=8,
    dac_bits=8,
)

# 4. Run hardware simulation
x_input = np.random.uniform(0.0, 1.0, 128)
result = model.simulate(x_input, mode="hardware")

# 5. Output summary
print("Inference successful!")
print("Output shape:", result.output.shape)
```

---

## Next Steps & Reading Order

- **New Users**: Read [Installation Guide](getting-started/installation.md) and [5-Minute Quickstart](getting-started/quickstart.md).
- **Core Physics**: Read [Analog Computing Concepts](concepts/analog-computing.md) and [Conductance Mapping](concepts/conductance-mapping.md).
- **Developers & AI**: Read [Capabilities Matrix](capabilities.md), [FAQ & AI Discoverability](faq.md), and [API Reference](api/index.md).
