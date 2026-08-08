# Physical Devices Overview

All memristive NVM device models in AnalogLib inherit from the abstract base class `Device` (`analoglib.devices.base.Device`).

---

## 1. Class Hierarchy

```text
               Device (ABC)
                │
         ───────┴───────
        │               │
   IdealDevice        ReRAM
```

---

## 2. Base Class Interface (`Device`)

Every device subclass must implement:

- `g_min` (float): Minimum conductance in Siemens (S).
- `g_max` (float): Maximum conductance in Siemens (S).
- `quantize(g: np.ndarray) -> np.ndarray`: Quantizes conductances to discrete device levels.
- `add_noise(g: np.ndarray) -> np.ndarray`: Injects read noise.
- `add_variation(g: np.ndarray) -> np.ndarray`: Applies device-to-device variation.
- `to_dict() -> dict`: Serializes device configuration.
- `from_dict(d: dict) -> Device`: Deserializes device object.

