# Device Noise & Physical Faults

In real physical NVM hardware, conductance readings fluctuate due to thermal noise, random telegraph noise (RTN), programming inaccuracy, and fabrication defects.

AnalogLib models four distinct physical noise and defect mechanisms in `ReRAM`.

---

## 1. Read Noise (Gaussian Conductance Noise)

Read noise represents instantaneous thermal fluctuations during array readout:

$$\tilde{G} = Q(G) + \delta G$$

$$\delta G \sim \mathcal{N}\left(0, \sigma_{\text{read}}^2 \cdot (G_{\text{max}} - G_{\text{min}})^2\right)$$

where `read_noise_sigma` ($\sigma_{\text{read}}$) is specified as a fraction of full conductance span.

---

## 2. Programming Inaccuracy Error

Programming error represents cycle-to-cycle variability during state write/programming pulses:

$$G_{\text{prog}} = Q(G) + \delta G_{\text{prog}}$$

$$\delta G_{\text{prog}} \sim \mathcal{N}\left(0, \sigma_{\text{prog}}^2 \cdot (G_{\text{max}} - G_{\text{min}})^2\right)$$

---

## 3. Device-to-Device (D2D) Structural Variation

D2D variation models fabrication line non-uniformity across different physical cells on a chip:

$$G_{\text{max}, i, j} = G_{\text{max}} \cdot (1 + \eta_{i,j}), \quad \eta_{i,j} \sim \mathcal{N}(0, \sigma_{\text{d2d}}^2)$$

---

## 4. Stuck-at Faults (Defect Defects)

Physical manufacturing defects can render cells permanently stuck at High Conductance State (Stuck-at-1 / HRS) or Low Conductance State (Stuck-at-0 / LRS).

Given `stuck_at_fault_rate` ($p_{\text{fault}}$):
- $p_{\text{fault}} / 2$ of cells are stuck at $G_{\text{min}}$ (Stuck-at-0)
- $p_{\text{fault}} / 2$ of cells are stuck at $G_{\text{max}}$ (Stuck-at-1)

---

## 5. Code Demonstration

```python
import analoglib as al
import numpy as np

al.set_seed(42)

# Create ReRAM with 1% read noise and 2% stuck-at faults
device = al.ReRAM(
    g_min=1e-6,
    g_max=100e-6,
    num_states=256,
    read_noise_sigma=0.01,
    programming_error_sigma=0.005,
    d2d_variation_sigma=0.02,
    stuck_at_fault_rate=0.02,
)

xbar = al.Crossbar(64, 32, device=device)
W = np.random.uniform(-1, 1, (64, 32))
xbar.load_weights(W, quantize=True)

V_in = np.random.uniform(0, 1, 64)

# Compare noise-free vs noisy VMM
I_clean = xbar.vmm(V_in, noise=False, mode=al.SimulationMode.DEVICE)
I_noisy = xbar.vmm(V_in, noise=True, mode=al.SimulationMode.DEVICE)

snr_db = 20 * np.log10(np.linalg.norm(I_clean) / np.linalg.norm(I_clean - I_noisy))
print(f"Output Signal-to-Noise Ratio (SNR): {snr_db:.2f} dB")
```
