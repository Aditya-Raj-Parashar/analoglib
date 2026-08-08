# Tutorial: Multi-Layer Perceptron (MLP) Hardware Simulation

This tutorial demonstrates end-to-end hardware simulation of a 3-layer Multi-Layer Perceptron (MLP) targeting ReRAM crossbars.

---

## 1. Step-by-Step Code Walkthrough

```python
import analoglib as al
import numpy as np

# Set RNG seed for reproducibility
al.set_seed(42)

# 1. Define synthetic weights for 64 -> 32 -> 16 -> 10 MLP
W1 = np.random.uniform(-0.5, 0.5, (64, 32))
W2 = np.random.uniform(-0.5, 0.5, (32, 16))
W3 = np.random.uniform(-0.5, 0.5, (16, 10))

# 2. Construct AnalogModel
model = al.AnalogModel.from_numpy(
    weights=[W1, W2, W3],
    activations=["relu", "relu", "softmax"],
    name="3LayerMLP",
)

# 3. Target ReRAM device with 8-bit peripherals
model.compile(
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01),
    adc_bits=8,
    dac_bits=8,
)


# 4. Simulate batch of input samples
X_batch = np.random.uniform(0, 1, 64)
result = model.simulate(X_batch, mode="hardware")

# 5. Output summary
print("MLP Simulation Successful!")
print("Output Probabilities (first 5):", result.output[:5])
```
