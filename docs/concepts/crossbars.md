# Crossbar Architecture & Tiling

A **crossbar array** is a 2D grid of intersecting horizontal wordlines (rows) and vertical bitlines (columns), with a memristive device element at each crosspoint.

---

## 1. Array Anatomy

```text
              Bitlines (Columns) -> Current Output I_j
               col_0    col_1    col_2   ...   col_{N-1}
                 │        │        │              │
Wordlines ──V_0──┼─[G₀₀]──┼─[G₀₁]──┼─[G₀₂]────────┼──
 (Rows)   ──V_1──┼─[G₁₀]──┼─[G₁₁]──┼─[G₁₂]────────┼──
 Input    ──V_2──┼─[G₂₀]──┼─[G₂₁]──┼─[G₂₂]────────┼──
 Voltage         │        │        │              │
                 ▼        ▼        ▼              ▼
                I_0      I_1      I_2           I_{N-1}
```

---

## 2. Differential Crossbar Arrays

Because memristors cannot have negative conductance ($G \ge 0$), signed neural network weights $W_{i,j} \in [-w_{\text{max}}, w_{\text{max}}]$ are mapped onto two physical crossbar arrays: $G^+$ (positive weight array) and $G^-$ (negative weight array).

Total current exiting column $j$:

$$I_j = \sum_{i} V_i (G^+_{i,j} - G^-_{i,j})$$

---

## 3. Large Weight Matrix Tiling (`TiledCrossbar`)

When a weight matrix (e.g. $1024 \times 512$) exceeds physical hardware array dimensions (e.g. $128 \times 64$), `TiledCrossbar` partitions the matrix across a 2D grid of physical tiles:

$$\text{Grid Rows} = \left\lceil \frac{M}{T_r} \right\rceil, \quad \text{Grid Columns} = \left\lceil \frac{N}{T_c} \right\rceil$$

```python
import analoglib as al
import numpy as np

# Partition 512x256 matrix into 128x64 tiles
W_large = np.random.randn(512, 256)
tiled = al.TiledCrossbar.from_matrix(W_large, tile_shape=(128, 64))

# Compute tiled VMM
V_in = np.random.uniform(0, 1, 512)
I_out = tiled.vmm(V_in)
print("Tiled output shape:", I_out.shape)
```
