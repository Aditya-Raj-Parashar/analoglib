# Device Simulation Mode (`SimulationMode.DEVICE`)

Device mode incorporates NVM device-level physical constraints:

- Conductance state quantization ($N_{\text{states}}$).
- Gaussian thermal read noise.
- Programming write variability.
- Stuck-at defect faults.

---

## Code Example

```python
import analoglib as al
import numpy as np

device = al.ReRAM(num_states=16, read_noise_sigma=0.02)
xbar = al.Crossbar(32, 16, device=device)
xbar.load_weights(np.random.randn(32, 16))

V_in = np.random.uniform(0, 1, 32)
out_device = xbar.vmm(V_in, noise=True, mode=al.SimulationMode.DEVICE)
print("Device mode output shape:", out_device.shape)
```
