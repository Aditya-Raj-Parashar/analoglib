# Common Errors & Troubleshooting Reference

This searchable guide explains common exception messages, root causes, and resolutions when working with AnalogLib v0.1.0.

---

## 1. `ValueError: Shape mismatch`

### Error Message
```text
ValueError: Weight shape (128, 64) doesn't match Crossbar (64, 32)
```

### Cause
The weight matrix shape $(M, N)$ passed to `load_weights(W)` does not match the initialized `Crossbar(rows, cols)` dimensions.

### Resolution
Ensure that `Crossbar(rows=M, cols=N)` matches the weight matrix dimensions, or use `TiledCrossbar.from_matrix(W, tile_shape=(tile_r, tile_c))` to automatically partition large weight matrices across multiple tiles.

---

## 2. `ValueError: Not a valid .analog file (bad magic)`

### Error Message
```text
ValueError: Not a valid .analog file (bad magic: b'PK\x03\x04'). This file cannot be opened without analoglib.
```

### Cause
The target file is corrupted, encrypted with an incompatible version, or is not a valid `.analog` binary file (e.g. attempting to open a zip or json file).

### Resolution
Ensure the file was generated using `analoglib.save()`. Do not modify `.analog` files manually using external text editors.

---

## 3. `ValueError: v_max must be > v_min`

### Error Message
```text
ValueError: v_max (0.0) must be > v_min (1.0)
```

### Cause
`ADC` or `DAC` initialization parameters have `v_max <= v_min`.

### Resolution
Pass valid range bounds where `v_max > v_min` (e.g. `ADC(v_min=-1e-3, v_max=1e-3)`).

---

## 4. `ImportError: matplotlib is required for visualization`

### Error Message
```text
ImportError: matplotlib is required for visualization. Install it with: pip install matplotlib
```

### Cause
You attempted to call `al.visualization` functions without `matplotlib` installed.

### Resolution
Install matplotlib using pip:
```bash
pip install "analoglib[viz]"
```

---

## 5. ADC Saturation Output Warning

### Problem
All hardware simulation outputs flatline at `v_max` or `v_min`.

### Cause
Current range $[v_{\text{min}}, v_{\text{max}}]$ specified in `ADC(v_min=..., v_max=...)` is too small for the accumulation output current exiting the crossbar columns ($I_j = \sum V_i G_{i,j}$).

### Resolution
Increase the ADC input range bounds to match total expected maximum column current, or reduce row driver voltage `v_max` on the `DAC`.
