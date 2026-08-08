# Conductance Quantization & State Levels

Physical NVM memory devices (ReRAM, PCM, Flash) possess a finite number of programmable conductance levels $N_{\text{states}}$.

---

## 1. Conductance State Step Size

For a device with $N_{\text{states}}$ discrete states spanning $[G_{\text{min}}, G_{\text{max}}]$, the conductance resolution per level is:

$$\Delta G = \frac{G_{\text{max}} - G_{\text{min}}}{N_{\text{states}} - 1}$$

The relation between state bit precision $B_{\text{device}}$ and number of levels $N_{\text{states}}$ is:

$$N_{\text{states}} = 2^{B_{\text{device}}}$$

For example:
- **1-bit device** (SLC ReRAM): $2^1 = 2$ states ($G_{\text{min}}$ and $G_{\text{max}}$)
- **4-bit device** (MLC ReRAM): $2^4 = 16$ states
- **8-bit device** (High-precision ReRAM): $2^8 = 256$ states

---

## 2. Quantization Function

Ideal continuous conductance values $G \in [G_{\text{min}}, G_{\text{max}}]$ are snapped to the nearest discrete state level using the round-to-nearest quantization operator $Q(G)$:

$$Q(G) = G_{\text{min}} + \text{round}\left( \frac{G - G_{\text{min}}}{\Delta G} \right) \cdot \Delta G$$

Values outside $[G_{\text{min}}, G_{\text{max}}]$ are hard-clipped before quantization.

---

## 3. Quantization Error Analysis & Code Sweep

```python
import analoglib as al
import numpy as np

# Generate continuous weight matrix
al.set_seed(42)
W = np.random.uniform(-1.0, 1.0, (64, 32))
mapping = al.DifferentialMapping(w_max=1.0)

print(f"{'States':>8}  {'Bits':>6}  {'Max Error':>12}  {'Mean Error':>12}")
print("-" * 44)

for n_states in [2, 4, 8, 16, 32, 64, 128, 256]:
    device = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=n_states, read_noise_sigma=0.0)
    xbar = al.Crossbar(64, 32, device=device, mapping=mapping)
    xbar.load_weights(W, quantize=True)
    W_recon = xbar.reconstruct_weights()

    max_err = np.max(np.abs(W - W_recon))
    mean_err = np.mean(np.abs(W - W_recon))
    bits = np.log2(n_states)
    print(f"{n_states:>8}  {bits:>6.1f}  {max_err:>12.6f}  {mean_err:>12.6f}")
```
