# Creating Custom Device Models

You can implement custom memristive NVM device models (e.g. PCM, Flash, FTJ) by subclassing `Device` (`analoglib.devices.base.Device`).

---

## 1. Subclassing `Device`

To implement a custom device, inherit from `Device` and override:

1. `quantize(self, g: np.ndarray) -> np.ndarray`
2. `add_noise(self, g: np.ndarray) -> np.ndarray`
3. `add_variation(self, g: np.ndarray) -> np.ndarray`

---

## 2. Complete Example: Custom PCM Device

```python
import analoglib as al
import numpy as np
from analoglib.devices.base import Device


class PCMDevice(Device):
    """Phase-Change Memory (PCM) model with non-linear crystallization steps."""

    def __init__(self, g_min: float = 0.5e-6, g_max: float = 50e-6, num_states: int = 64):
        super().__init__(g_min=g_min, g_max=g_max, num_states=num_states)
        # Non-linear exponential level spacing characteristic of PCM
        self._levels = g_min + (g_max - g_min) * (np.linspace(0, 1, num_states) ** 1.5)

    def quantize(self, g: np.ndarray) -> np.ndarray:
        g_clipped = np.clip(g, self.g_min, self.g_max)
        # Find nearest PCM level
        idx = np.abs(g_clipped[..., np.newaxis] - self._levels).argmin(axis=-1)
        return self._levels[idx]

    def add_noise(self, g: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0, 0.005 * self.g_range, size=g.shape)
        return np.clip(g + noise, self.g_min, self.g_max)

    def add_variation(self, g: np.ndarray) -> np.ndarray:
        variation = np.random.normal(0, 0.01 * self.g_range, size=g.shape)
        return np.clip(g + variation, self.g_min, self.g_max)


# Verify custom device with Crossbar
pcm = PCMDevice(num_states=16)
xbar = al.Crossbar(16, 8, device=pcm)
xbar.load_weights(np.random.randn(16, 8))
out = xbar.vmm(np.ones(16))
print("PCM VMM Output Shape:", out.shape)
```

