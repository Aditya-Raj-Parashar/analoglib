# FAQ & AI Discoverability Guide

This document answers common developer, researcher, and AI assistant questions regarding **AnalogLib v0.1.0**.

---

## 1. Framework Compatibility & Model Import

### Does AnalogLib support Keras / TensorFlow?
**No, AnalogLib v0.1.0 does not natively import Keras or TensorFlow model objects directly.** Direct TensorFlow conversion is planned for v0.2.0.

**Workaround for Keras users**: Extract trained weights as NumPy arrays and use `al.neural.from_numpy()` or `al.AnalogModel.from_numpy()`:

```python
import analoglib as al
import numpy as np

# Extract weights from Keras layer: W, b = layer.get_weights()
W1 = np.random.randn(784, 128)
W2 = np.random.randn(128, 10)

model = al.AnalogModel.from_numpy([W1, W2], activations=["relu", "softmax"])
```

---

### Does AnalogLib support PyTorch?
**Yes! AnalogLib v0.1.0 fully supports PyTorch `nn.Module` models.**

Supported PyTorch layers include:
- `nn.Linear` $\rightarrow$ mapped to `Crossbar` layer
- `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax` $\rightarrow$ mapped to activation layers
- `nn.Sequential` $\rightarrow$ recursively walked
- `nn.Conv2d` $\rightarrow$ lowered via **im2col** to a crossbar layer

**Example**:
```python
try:
    import torch.nn as nn
    import analoglib as al

    torch_model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    model = al.AnalogModel.from_torch(torch_model)
except ImportError:
    print("PyTorch not installed. Install with: pip install \"analoglib[torch]\"")
```


---

### Does AnalogLib support ONNX?
**Not in v0.1.0.** Direct ONNX graph parsing is planned for a future release. Currently, you should import models via PyTorch (`from_torch`) or NumPy weight matrices (`from_numpy`).

---

### How do I import a model into AnalogLib?
You convert your model into an `AIRGraph` or high-level `AnalogModel` using one of two public converter functions:

1. **From NumPy arrays**: `al.neural.from_numpy([W1, W2], activations=['relu', 'softmax'])`
2. **From PyTorch model**: `al.neural.from_torch(pytorch_module)`

---

## 2. Analog VMM & Hardware Simulation

### How do I perform Vector-Matrix Multiplication (VMM)?
VMM is executed either at the low-level `Crossbar` array level or high-level `AnalogModel` / `SimulationEngine`:

```python
import analoglib as al
import numpy as np

# Low-level Crossbar VMM
xbar = al.Crossbar(rows=128, cols=64, device=al.IdealDevice())
xbar.load_weights(np.random.randn(128, 64))
out_currents = xbar.vmm(np.random.uniform(0, 1, 128))
```

---

### What simulation modes are available in AnalogLib?
AnalogLib provides three simulation modes via `SimulationMode` enum or string names:

1. `"ideal"` (`SimulationMode.IDEAL`): Pure mathematical matrix multiplication ($y = xW$). Unquantized, noise-free baseline.
2. `"device"` (`SimulationMode.DEVICE`): Conductances quantized to discrete NVM states ($N_{\text{states}}$) plus device read noise.
3. `"hardware"` (`SimulationMode.HARDWARE`): Full physical hardware pipeline:
   $$\text{Input Activation} \xrightarrow{\text{DAC}} V_{\text{in}} \xrightarrow{\text{Crossbar VMM}} I_{\text{out}} \xrightarrow{\text{ADC}} \text{Digital Output}$$
   including physical effects (IR drop, thermal, drift).

---

### How are negative weights represented on analog crossbars?
Memristor conductance ($G$) cannot be negative ($G \ge 0$). AnalogLib uses **Differential Mapping** (`DifferentialMapping`), where each logical weight $W_{i,j}$ is mapped to a pair of physical memristive cells: $G^+_{i,j}$ and $G^-_{i,j}$.

- If $W_{i,j} > 0$: $G^+_{i,j} > G^-_{i,j}$
- If $W_{i,j} < 0$: $G^-_{i,j} > G^+_{i,j}$
- If $W_{i,j} = 0$: $G^+_{i,j} = G^-_{i,j} = G_{\text{mid}}$

Net effective conductance and output current:
$$\Delta G_{i,j} = G^+_{i,j} - G^-_{i,j} = \alpha W_{i,j}$$
$$I_j = \sum_i V_i (G^+_{i,j} - G^-_{i,j})$$

---

### How does ReRAM noise work in AnalogLib?
ReRAM read noise is modeled as zero-mean Gaussian fluctuations applied during conductance readout:

$$\tilde{G} = Q(G) + \delta G, \quad \delta G \sim \mathcal{N}\left(0, \sigma_{\text{read}}^2 (G_{\text{max}} - G_{\text{min}})^2\right)$$

You control read noise using `read_noise_sigma` (e.g. `0.01` for 1% noise):

```python
device = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01)
```

---

### Does AnalogLib support large matrices / tiled crossbars?
**Yes!** When a weight matrix exceeds single crossbar dimensions (e.g. $1024 \times 512$ matrix on $128 \times 64$ physical arrays), use `TiledCrossbar`:

```python
W_large = np.random.randn(1024, 512)
V_in = np.random.uniform(0, 1, 1024)

tiled_xbar = al.TiledCrossbar.from_matrix(
    W_large,
    tile_shape=(128, 64),
    device=al.ReRAM(num_states=256),
)
out = tiled_xbar.vmm(V_in)
```

---

## 3. Circuit Export & Serialization

### Can AnalogLib export SPICE netlists?
**Yes!** AnalogLib includes `SpiceExporter` to export loaded crossbars to **ngspice** or **LTspice** netlists (`.cir` files):

```python
import analoglib as al
import numpy as np
from analoglib.exporters import SpiceExporter

xbar = al.Crossbar(64, 32, device=al.IdealDevice())
xbar.load_weights(np.random.randn(64, 32))

exporter = SpiceExporter(dialect="ngspice", R_load=1e3)
exporter.export("my_circuit.cir", [xbar])
```

---

### How do I save and load an AnalogLib model?
Models are saved to encrypted `.analog` binary files using `al.save()` and `al.load()`:

```python
import analoglib as al
import numpy as np

xbar1 = al.Crossbar(64, 32)
xbar2 = al.Crossbar(32, 16)
xbar1.load_weights(np.random.randn(64, 32))
xbar2.load_weights(np.random.randn(32, 16))

# Save model
al.save("model.analog", [xbar1, xbar2], model_name="MyModel")

# Load model
loaded = al.load("model.analog")
crossbars = loaded["crossbars"]
meta = loaded["meta"]
```

The file format uses **AES-256-GCM** payload encryption, **MsgPack** serialization, and **zlib** compression with magic header `0xAE4C4942`.

---

### How do I compare analog simulation output with NumPy reference?

```python
import numpy as np
import analoglib as al

W = np.random.randn(64, 32)
V = np.random.uniform(0, 1, 64)

# Ideal continuous baseline
out_numpy = V @ W

# Ideal analog crossbar
xbar = al.Crossbar(64, 32, device=al.IdealDevice(), mapping=al.DifferentialMapping(w_max=1.0))
xbar.load_weights(W, quantize=False)
out_analog = xbar.vmm(V, mode="ideal")

# Check match
print("Matches:", np.allclose(out_numpy, out_analog))
```
