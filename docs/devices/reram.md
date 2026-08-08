# ReRAM Device Model (`analoglib.ReRAM`)

`ReRAM` (Resistive Random Access Memory) models a multi-level oxide-based memristive cell with discrete state levels, thermal read noise, programming errors, device-to-device structural variations, and stuck-at defect faults.

---

## 1. Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `g_min` | `float` | `1e-6` ($1 \, \mu\text{S}$) | Minimum (HRS) conductance in Siemens. |
| `g_max` | `float` | `100e-6` ($100 \, \mu\text{S}$) | Maximum (LRS) conductance in Siemens. |
| `num_states` | `int` | `256` (8-bit) | Number of discrete programmable conductance levels. |
| `read_noise_sigma` | `float` | `0.01` (1%) | Gaussian read noise standard deviation (fraction of $G_{\text{max}} - G_{\text{min}}$). |
| `programming_error_sigma` | `float` | `0.005` (0.5%) | Programming write variability standard deviation. |
| `d2d_variation_sigma` | `float` | `0.0` | Structural device-to-device conductance variation standard deviation. |
| `stuck_at_fault_rate` | `float` | `0.0` | Probability of cell stuck-at defect fault (half HRS, half LRS). |

---

## 2. Usage Code Example

```python
import analoglib as al
import numpy as np

# Instantiate 4-bit (16 state) ReRAM device model
device = al.ReRAM(
    g_min=1e-6,
    g_max=100e-6,
    num_states=16,
    read_noise_sigma=0.01,
    programming_error_sigma=0.005,
    stuck_at_fault_rate=0.01,
)

# Test conductance quantization
g_continuous = np.array([10e-6, 45e-6, 88e-6])
g_quantized = device.quantize(g_continuous)

print("Nominal Continuous: ", g_continuous)
print("Quantized States:   ", g_quantized)
```
