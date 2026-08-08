# Analog Computing Concepts & Physical Principles

AnalogLib models physical analog in-memory computing (IMC) based on fundamental physical laws of circuit theory.

---

## 1. Physical Foundations: Ohm's & Kirchhoff's Laws

In resistive crossbar arrays, matrix-vector multiplication (VMM) is executed naturally by physics:

### Ohm's Law (Multiplication)
When a voltage $V_i$ is applied to a wordline connected to a cell with conductance $G_{i,j}$, the resulting current $I_{i,j}$ flowing through the element is:

$$I_{i,j} = V_i \cdot G_{i,j}$$

where:
- $V_i$ is input voltage (Volts, V)
- $G_{i,j} = 1 / R_{i,j}$ is device conductance (Siemens, S)
- $I_{i,j}$ is current (Amperes, A)

### Kirchhoff's Current Law (Accumulation)
All cell currents along column $j$ sum together at the bitline output:

$$I_j = \sum_{i=0}^{M-1} I_{i,j} = \sum_{i=0}^{M-1} V_i \cdot G_{i,j}$$

This performs an $M$-element dot product in a single parallel step $\mathcal{O}(1)$ time complexity.

---

## 2. Mathematical vs. Physical Domain Conversions

Digital weights $W \in [-w_{\text{max}}, w_{\text{max}}]$ are dimensionless. Physical crossbars store conductances $G \in [G_{\text{min}}, G_{\text{max}}]$.

The conductance scaling factor $\alpha$ is defined as:

$$\alpha = \frac{G_{\text{max}} - G_{\text{min}}}{2 \cdot w_{\text{max}}}$$

The physical current produced by row voltage $V_i$ is:

$$I_{\text{phys}} = \alpha \sum_{i} V_i W_i$$

To convert physical current $I_{\text{phys}}$ back to dimensionless math output $y_{\text{math}}$:

$$y_{\text{math}} = \frac{I_{\text{phys}}}{\alpha}$$

---

## 3. Code Verification Example

```python
import analoglib as al
import numpy as np

# Physical device parameters
g_min, g_max = 1e-6, 100e-6
device = al.IdealDevice(g_min=g_min, g_max=g_max)
mapping = al.DifferentialMapping(w_max=1.0)

# 4x2 matrix and input voltage
W = np.array([[0.5, -0.8], [0.2, 0.9], [-0.4, 0.1], [0.7, -0.3]])
V = np.array([0.8, 0.5, 0.2, 1.0])

# Physical Crossbar VMM
xbar = al.Crossbar(4, 2, device=device, mapping=mapping)
xbar.load_weights(W, quantize=False)
I_out = xbar.vmm(V, mode=al.SimulationMode.IDEAL)

# Mathematical output conversion
scale_factor = (g_max - g_min) / 1.0
y_math = I_out / scale_factor

# Compare with pure NumPy (V @ W)
y_numpy = V @ W
print("NumPy math output:      ", y_numpy)
print("Reconstructed output:   ", y_math)
print("Matches perfectly:      ", np.allclose(y_numpy, y_math))
```
