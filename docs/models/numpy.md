# Loading NumPy Weight Arrays (`al.neural.from_numpy`)

The NumPy converter `al.neural.from_numpy()` and `AnalogModel.from_numpy()` convert raw NumPy weight matrices into an `AIRGraph` or compiled `AnalogModel`.

---

## 1. Function Signature

```text
al.neural.from_numpy(
    weights: List[np.ndarray],
    name: str = "numpy_model",
    activations: Optional[List[str]] = None,
) -> AIRGraph
```

### Parameters
- `weights` (`List[np.ndarray]`): List of 2D weight matrices $[(M_1, N_1), (M_2, N_2), \dots]$.
- `name` (`str`): Model identifier name.
- `activations` (`List[str]`, optional): Activation names after each layer (`"relu"`, `"sigmoid"`, `"tanh"`, `"softmax"`, `"none"`).


---

## 2. Code Example

```python
import analoglib as al
import numpy as np

# Create 2 weight matrices
W1 = np.random.randn(64, 32)
W2 = np.random.randn(32, 10)

# Build AnalogModel
model = al.AnalogModel.from_numpy([W1, W2], activations=["relu", "softmax"])

# Compile and simulate
model.compile(device=al.ReRAM(num_states=256), adc_bits=8, dac_bits=8)
out = model.simulate(np.random.uniform(0, 1, 64), mode="hardware")
print("Output shape:", out.output.shape)
```
