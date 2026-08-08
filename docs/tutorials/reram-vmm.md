# Tutorial: ReRAM VMM Under Device Noise

This tutorial evaluates VMM output degradation under varying ReRAM read noise levels.

---

## Code Example

```python
import analoglib as al
import numpy as np

al.set_seed(42)

W = np.random.uniform(-1, 1, (128, 64))
V = np.random.uniform(0, 1, 128)

print(f"{'Read Noise Sigma':>18}  {'SNR (dB)':>10}  {'Max Rel Error (%)':>20}")
print("-" * 54)

for noise_sigma in [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]:
    device = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=noise_sigma)
    xbar = al.Crossbar(128, 64, device=device)
    xbar.load_weights(W, quantize=True)

    I_clean = xbar.vmm(V, noise=False, mode=al.SimulationMode.DEVICE)
    I_noisy = xbar.vmm(V, noise=True, mode=al.SimulationMode.DEVICE)

    if noise_sigma == 0.0:
        snr_db = float("inf")
        rel_err = 0.0
    else:
        diff = I_noisy - I_clean
        snr_db = 20 * np.log10(np.linalg.norm(I_clean) / np.linalg.norm(diff))
        rel_err = np.max(np.abs(diff) / (np.abs(I_clean) + 1e-12)) * 100

    print(f"{noise_sigma:>18.3f}  {snr_db:>10.2f}  {rel_err:>20.2f}")
```
