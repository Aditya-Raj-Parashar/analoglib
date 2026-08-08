# Digital-to-Analog Converter Model (`al.DAC`)

`DAC` (`analoglib.adc_dac.dac.DAC`) models uniform digital-to-analog voltage quantization for driving crossbar row inputs.

---

## 1. Parameters & Resolution

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `bits` | `int` | `8` | Resolution in bits ($2^N$ levels). |
| `v_min` | `float` | `0.0` | Minimum output voltage (V). |
| `v_max` | `float` | `1.0` | Maximum output voltage (V). |

Voltage step per LSB:

$$\Delta V = \frac{v_{\text{max}} - v_{\text{min}}}{2^{\text{bits}} - 1}$$

---

## 2. Code Example

```python
import analoglib as al
import numpy as np

dac = al.DAC(bits=8, v_min=0.0, v_max=1.0)
print("DAC LSB Resolution:", dac.resolution)

x_digital = np.array([0.12345, 0.67891, 0.99999])
v_analog = dac.convert(x_digital)
print("Quantized DAC Voltages:", v_analog)
```
