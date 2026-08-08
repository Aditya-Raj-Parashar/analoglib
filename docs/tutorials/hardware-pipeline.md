# Tutorial: Hardware Pipeline & Profiling Workflow

This tutorial demonstrates combining multi-layer hardware inference with the `AnalogProfiler` to generate a hardware report.

---

## Code Example

```python
import analoglib as al
import numpy as np
from analoglib.analysis import AnalogProfiler

al.set_seed(42)

# Create 2-layer model
xb1 = al.Crossbar(128, 64, device=al.ReRAM(num_states=256))
xb2 = al.Crossbar(64, 10, device=al.ReRAM(num_states=256))

xb1.load_weights(np.random.randn(128, 64))
xb2.load_weights(np.random.randn(64, 10))

# Attach peripherals
adc = al.ADC(bits=8, v_min=-1e-3, v_max=1e-3)
dac = al.DAC(bits=8, v_min=0.0, v_max=1.0)
engine = al.SimulationEngine(crossbars=[xb1, xb2], adc=adc, dac=dac)

# Run hardware simulation
V_in = np.random.uniform(0, 1, 128)
y_out = engine.run(V_in, mode="hardware")

# Profile hardware metrics
profiler = AnalogProfiler(t_read=10e-9, cell_feature_F=10.0, V_supply=1.0)
report = profiler.profile([xb1, xb2], V_input=V_in, adc=adc, dac=dac)

# Print hardware performance report
report.print()
```
