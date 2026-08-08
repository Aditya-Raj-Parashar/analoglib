# Tutorial: Quantization & Precision Sweeps

This tutorial performs an ADC resolution bit-width sweep to compute SQNR (Signal-to-Quantization-Noise Ratio).

---

## Code Example

```python
import analoglib as al
import numpy as np

al.set_seed(42)

W = np.random.uniform(-1, 1, (64, 32))
V = np.random.uniform(0, 1, 64)

xb = al.Crossbar(64, 32, device=al.IdealDevice())
xb.load_weights(W, quantize=False)
I_ideal = xb.vmm(V, mode=al.SimulationMode.IDEAL)

print(f"{'ADC Bits':>10}  {'Levels':>8}  {'SQNR (dB)':>12}")
print("-" * 34)

for bits in [4, 6, 8, 10, 12, 14, 16]:
    adc = al.ADC(bits=bits, v_min=float(I_ideal.min()), v_max=float(I_ideal.max()))
    I_quant = adc.convert(I_ideal)

    diff = I_quant - I_ideal
    sqnr_db = 20 * np.log10(np.linalg.norm(I_ideal) / np.linalg.norm(diff + 1e-15))
    print(f"{bits:>10}  {2**bits:>8}  {sqnr_db:>12.2f}")
```
