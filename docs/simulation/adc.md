# Analog-to-Digital Converter Model (`al.ADC`)

`ADC` (`analoglib.adc_dac.adc.ADC`) models uniform analog-to-digital current/voltage quantization and hard clipping.

---

## 1. Parameters & Clipping

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bits` | `int` | `8` | Resolution in bits ($2^N$ levels). |
| `v_min` | `float` | `0.0` | Minimum representable input value. |
| `v_max` | `float` | `1.0` | Maximum representable input value. |

Step resolution:

$$\Delta I = \frac{v_{\text{max}} - v_{\text{min}}}{2^{\text{bits}} - 1}$$

Values outside $[v_{\text{min}}, v_{\text{max}}]$ are hard-clipped.

---

## 2. Code Example

```python
import analoglib as al
import numpy as np

adc = al.ADC(bits=8, v_min=-500e-6, v_max=500e-6)
print("ADC Resolution:", adc.resolution)

i_analog = np.array([-600e-6, 0.0, 250e-6, 800e-6])
y_quantized = adc.convert(i_analog)
print("Clipped & Quantized Output:", y_quantized)
```
