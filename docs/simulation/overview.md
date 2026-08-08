# SimulationEngine Overview

`SimulationEngine` (`analoglib.SimulationEngine`) is the primary multi-layer execution runtime in AnalogLib.

---

## 1. Responsibilities

- Multi-layer inference pipeline execution.
- Integration of physical crossbars, peripherals (ADC/DAC), and activation functions.
- Execution modes: `"ideal"`, `"device"`, and `"hardware"`.

---

## 2. API Signature

```text
SimulationEngine(
    crossbars: List[Crossbar],
    adc: Optional[ADC] = None,
    dac: Optional[DAC] = None,
    activations: Optional[List[str]] = None,
)
```

---

## 3. Code Example

```python
import analoglib as al
import numpy as np

xb1 = al.Crossbar(32, 16)
xb2 = al.Crossbar(16, 8)
xb1.load_weights(np.random.randn(32, 16))
xb2.load_weights(np.random.randn(16, 8))

engine = al.SimulationEngine(
    crossbars=[xb1, xb2],
    adc=al.ADC(bits=8),
    dac=al.DAC(bits=8),
)

out = engine.run(np.random.uniform(0, 1, 32), mode="hardware")
print("Engine run completed. Output shape:", out.shape)
```
