# Hardware Analytics & Profiler (`al.analysis.AnalogProfiler`)

`AnalogProfiler` computes energy consumption, array power, latency, cell area, and TOPS/W throughput metrics for crossbar models.

---

## 1. Physical Governing Formulas

- **Array Power**: $P_{\text{array}} = \sum_{i,j} V_i^2 \cdot G_{i,j}$ (Watts)
- **Read Energy**: $E_{\text{read}} = P_{\text{array}} \cdot t_{\text{read}}$ (Joules / VMM)
- **ADC Energy**: $E_{\text{adc}} = N_{\text{cols}} \cdot B_{\text{adc}} \cdot 0.5 \times 10^{-12}$ (Joules)
- **Cell Area**: $A_{\text{cell}} = N_{\text{cells}} \cdot 4 F^2$ ($\mu\text{m}^2$)
- **Energy Efficiency**: $\text{TOPS/W} = \frac{\text{MAC Operations}}{E_{\text{total}} \times 10^{12}}$

---

## 2. Code Example

```python
import analoglib as al
import numpy as np
from analoglib.analysis import AnalogProfiler

# Build crossbars
xb1 = al.Crossbar(128, 64, device=al.ReRAM(num_states=256))
xb1.load_weights(np.random.randn(128, 64))

profiler = AnalogProfiler(t_read=10e-9, cell_feature_F=10.0, V_supply=1.0)
report = profiler.profile([xb1], V_input=np.ones(128) * 0.5, adc=al.ADC(bits=8))

report.print()

print(f"Total Energy:  {report.total_energy_J * 1e9:.3f} nJ")
print(f"TOPS/W:        {report.tops_per_watt:.2f}")
```
