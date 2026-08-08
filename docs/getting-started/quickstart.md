# 5-Minute Quickstart

This tutorial walks through building and running your first multi-layer analog crossbar model using AnalogLib's high-level `AnalogModel` API.

---

## Complete Self-Contained Quickstart Example

```python
import analoglib as al
import numpy as np

# 1. Reproducibility & Weight generation (784 -> 128 -> 10 network)
al.set_seed(42)
W1 = np.random.uniform(-0.5, 0.5, (784, 128))
W2 = np.random.uniform(-0.5, 0.5, (128, 10))

# 2. Build AnalogModel wrapper via AIR
model = al.AnalogModel.from_numpy([W1, W2], activations=["relu", "softmax"])

# 3. Compile model targeting ReRAM crossbar with 8-bit DAC/ADC peripherals
model.compile(
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01),
    adc_bits=8,
    dac_bits=8,
)

# 4. Run hardware simulation inference
x_input = np.random.uniform(0.0, 1.0, 784)
result = model.simulate(x_input, mode="hardware")

# 5. Print hardware summary report
result.report()

# 6. Save and Reload model artifact
saved_path = al.save("my_first_model.analog", result.engine.crossbars, model_name="QuickstartMLP")
loaded = al.load(saved_path)
print("Saved and successfully reloaded crossbar layers:", len(loaded["crossbars"]))
```
