# Weight-to-Conductance Mapping Strategies

Weights in neural networks are signed floating-point values $W \in [-w_{\text{max}}, w_{\text{max}}]$. Physical devices store non-negative conductances $G \in [G_{\text{min}}, G_{\text{max}}]$.

AnalogLib provides two conductance mapping strategies: `DifferentialMapping` and `OffsetMapping`.

---

## 1. Differential Conductance Mapping (`DifferentialMapping`)

Each weight entry $W_{i,j}$ is mapped to a pair of physical memristors: $G^+_{i,j}$ and $G^-_{i,j}$.

### Governing Equations

Given weight scaling factor:

$$\alpha = \frac{G_{\text{max}} - G_{\text{min}}}{2 \cdot w_{\text{max}}}$$

And midpoint conductance:

$$G_{\text{mid}} = \frac{G_{\text{max}} + G_{\text{min}}}{2}$$

The positive and negative cell conductances are:

$$G^+_{i,j} = G_{\text{mid}} + \frac{1}{2} \alpha W_{i,j}$$

$$G^-_{i,j} = G_{\text{mid}} - \frac{1}{2} \alpha W_{i,j}$$

Net conductance difference:

$$\Delta G_{i,j} = G^+_{i,j} - G^-_{i,j} = \alpha W_{i,j}$$

### Roundtrip Weight Reconstruction

To extract effective weights from conductance matrices $G^+$ and $G^-$:

$$W_{\text{recon}} = \frac{G^+ - G^-}{\alpha}$$

---

## 2. Offset Conductance Mapping (`OffsetMapping`)

In offset mapping, signed weights are mapped to a single conductance cell $G_{i,j}$, and a static reference zero-offset $G_{\text{mid}}$ is subtracted from output current:

$$G_{i,j} = G_{\text{min}} + \alpha \cdot (W_{i,j} + w_{\text{max}})$$

$$\text{where } \alpha = \frac{G_{\text{max}} - G_{\text{min}}}{2 \cdot w_{\text{max}}}$$

Zero weight ($W = 0$) maps directly to midpoint conductance $G_{\text{mid}}$.

---

## 3. Executable Python Demonstration

```python
import analoglib as al
import numpy as np

# Signed weight matrix
W = np.array([[0.8, -0.5, 0.0], [-0.2, 0.9, -0.7]])
device = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)

# Differential Mapping
diff_mapping = al.DifferentialMapping(w_max=1.0)
g_pos, g_neg = diff_mapping.weights_to_conductance(W, device)
W_recon_diff = diff_mapping.conductance_to_weights(g_pos, g_neg, device=device)

# Offset Mapping
offset_mapping = al.OffsetMapping(w_max=1.0)
(g_single,) = offset_mapping.weights_to_conductance(W, device)
W_recon_off = offset_mapping.conductance_to_weights(g_single, device=device)

print("Original Weights:\n", W)
print("Reconstructed (Differential):\n", np.round(W_recon_diff, 4))
print("Reconstructed (Offset):\n", np.round(W_recon_off, 4))
```
