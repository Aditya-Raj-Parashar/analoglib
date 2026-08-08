# ONNX Model Compatibility

> [!WARNING]
> **Status: 🚧 Planned / Not Implemented in AnalogLib v0.1.0**
> Direct ONNX file parsing (`al.neural.from_onnx`) is **not implemented** in v0.1.0. This feature is scheduled for the v0.3.0 roadmap.

---

## Migration Workaround for ONNX Models

To simulate ONNX models in AnalogLib v0.1.0:

1. Load the ONNX model into PyTorch using `onnx2torch` or extract tensor initializers via `onnx.numpy_helper`.
2. Pass the resulting PyTorch module to `al.neural.from_torch()` or pass extracted NumPy weights to `al.neural.from_numpy()`:

```python
import analoglib as al
import numpy as np

# Extracted ONNX initializers as NumPy arrays
W1 = np.random.randn(256, 128)
W2 = np.random.randn(128, 64)

# Build AnalogModel via NumPy converter
model = al.AnalogModel.from_numpy(weights=[W1, W2], activations=["relu", "none"])
model.compile(device=al.ReRAM(num_states=256))
```
