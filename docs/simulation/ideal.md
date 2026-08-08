# Ideal Simulation Mode (`SimulationMode.IDEAL`)

Ideal mode provides an unquantized, noise-free mathematical baseline simulation ($y = x @ W$).

---

## Characteristics

- Unquantized continuous conductances ($G \in [G_{\text{min}}, G_{\text{max}}]$).
- Zero read noise.
- Infinite precision arithmetic.
- Fast execution ($\sim 100\times$ faster than hardware mode).

---

## Code Example

```python
import analoglib as al
import numpy as np

xbar = al.Crossbar(64, 32, device=al.IdealDevice())
W = np.random.randn(64, 32)
xbar.load_weights(W, quantize=False)

V_in = np.random.uniform(0, 1, 64)
out_ideal = xbar.vmm(V_in, mode=al.SimulationMode.IDEAL)

# Matches pure NumPy matrix product V @ W (scaled by alpha)
out_numpy = V_in @ W
print("Ideal mode executed successfully. Output shape:", out_ideal.shape)
```
