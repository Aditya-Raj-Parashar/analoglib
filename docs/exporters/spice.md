# SPICE Netlist Exporter (`al.exporters.SpiceExporter`)

`SpiceExporter` converts loaded crossbars into circuit-level SPICE netlists for validation with **ngspice** or **LTspice**.

---

## 1. Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `dialect` | `str` | `"ngspice"` | Target dialect (`"ngspice"` or `"ltspice"`). |
| `R_load` | `float` | `1000.0` | Output column load sense resistor in Ohms ($1 \, \text{k}\Omega$). |
| `V_dd` | `float` | `1.0` | Driver supply voltage in Volts. |

---

## 2. Code Example

```python
import analoglib as al
import numpy as np
from analoglib.exporters import SpiceExporter

xbar = al.Crossbar(4, 2, device=al.IdealDevice())
xbar.load_weights(np.array([[0.5, -0.2], [0.1, 0.9], [-0.4, 0.3], [0.7, -0.8]]))

# Generate SPICE netlist string
exporter = SpiceExporter(dialect="ngspice", R_load=1e3)
netlist_str = exporter.export_str([xbar], title="TestCrossbarExport")
print("Generated SPICE Netlist Header:\n", netlist_str[:300])

# Export directly to file
cir_file = exporter.export("circuit.cir", [xbar])
print("Saved netlist to:", cir_file)
```
