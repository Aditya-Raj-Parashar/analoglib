# AnalogLib — Core Concepts

This document explains the physics, mathematics, and design principles behind AnalogLib. It is written for researchers who want to understand **why** the library works the way it does.

---

## 1. Resistive Crossbar Arrays

A crossbar array is a 2D grid of resistive memory devices placed at the intersection of horizontal (row) and vertical (column) wires.

```
        col_0    col_1    col_2
         │        │        │
row_0 ───┼────────┼────────┼───  V₀
         │G₀₀     │G₀₁     │G₀₂
row_1 ───┼────────┼────────┼───  V₁
         │G₁₀     │G₁₁     │G₁₂
row_2 ───┼────────┼────────┼───  V₂
         │        │        │
         I₀       I₁       I₂
```

### Physical operation

1. **Input**: Voltage `V_i` is applied to each row wire
2. **Computation**: At each junction, current flows: `i_{ij} = G_{ij} × V_i` (Ohm's Law)
3. **Summation**: Column currents sum by Kirchhoff's Current Law: `I_j = Σ_i G_{ij} × V_i`

In matrix form:

```
I = V @ G        (vector-matrix multiply)
```

This computes an entire matrix-vector multiplication **in one analog step**, as opposed to O(n²) multiply-accumulate operations in digital.

### Why this matters for neural networks

A fully-connected neural network layer computes `y = W @ x + b`. The weight matrix `W` can be stored as conductances `G` in a crossbar, and input `x` can be encoded as voltages `V`. The output currents `I` directly give the matrix product — no digital multipliers needed.

---

## 2. Device Physics

### 2.1 ReRAM (Resistive RAM)

ReRAM devices store data as resistance states. A **conductive filament** forms or dissolves inside a metal-insulator-metal (MIM) structure, switching between high-resistance (HRS) and low-resistance (LRS) states.

**AnalogLib models these properties:**

| Property | Parameter | Physical origin |
|----------|-----------|-----------------|
| Conductance range | `g_min`, `g_max` | HRS and LRS states |
| Discrete levels | `num_states` | Finite filament configurations |
| Read noise | `read_noise_sigma` | Thermal + 1/f noise during read |
| Programming error | `programming_error_sigma` | Imprecise filament formation |
| D2D variation | `d2d_variation_sigma` | Manufacturing non-uniformity |
| Stuck-at faults | `stuck_at_fault_rate` | Permanent device defects |

#### Quantization model

Conductance levels are modeled as uniformly distributed between `g_min` and `g_max`:

```
G_levels = linspace(g_min, g_max, num_states)
```

This is the standard model used in NeuroSim, CrossSim, and most crossbar simulation literature. Real devices may have non-uniform level spacing — this can be modeled by subclassing `Device`.

> **Scientific note**: The uniform-level model is an approximation. Real ReRAM devices exhibit state-dependent noise (σ varies with G) and non-uniform level spacing. AnalogLib clearly marks this as a simplification.

#### Noise model

Read noise is modeled as additive Gaussian:

```
G_noisy = G + N(0, σ²)
where σ = read_noise_sigma × (g_max - g_min)
```

This is a relative sigma — specifying `read_noise_sigma=0.03` means σ is 3% of the conductance window.

### 2.2 IdealDevice

A mathematically perfect device with:
- Continuous conductance (infinite resolution)
- Zero noise, zero variation
- Useful as a **controlled baseline** for isolating error sources

---

## 3. Weight-to-Conductance Mapping

### 3.1 The sign problem

Neural network weights are signed (positive and negative), but conductance is always non-negative (`G ≥ 0`). This is a fundamental challenge.

### 3.2 Differential mapping (default)

The standard solution uses **two conductances per weight**:

```
W_norm = (W + w_max) / (2 × w_max)        ∈ [0, 1]

G⁺ = g_min + W_norm × (g_max - g_min)
G⁻ = g_min + (1 - W_norm) × (g_max - g_min)
```

The effective weight is proportional to `G⁺ - G⁻`:

```
When W > 0:  G⁺ > G⁻  →  G⁺ - G⁻ > 0  ✓
When W < 0:  G⁺ < G⁻  →  G⁺ - G⁻ < 0  ✓
When W = 0:  G⁺ = G⁻  →  G⁺ - G⁻ = 0  ✓
```

**Cost**: 2× the devices. **Benefit**: Full signed weight support, symmetric behavior.

### 3.3 Offset mapping (alternative)

Uses a single device with mid-conductance as the zero point:

```
G = g_mid + W × scale
where g_mid = (g_min + g_max) / 2
      scale = (g_max - g_min) / (2 × w_max)
```

**Cost**: 1× devices. **Drawback**: Must subtract the offset current, reducing effective dynamic range.

### 3.4 Roundtrip error

Due to quantization, `W → G → W'` introduces error. For `n` conductance levels with differential mapping, the maximum weight error is approximately:

```
ε_max ≈ 2 × w_max / (n - 1)
```

For 256 levels and w_max=1: `ε_max ≈ 0.008` (< 1%).

---

## 4. Simulation Fidelity Levels

AnalogLib defines a hierarchy of simulation modes:

### Level 0: Ideal (mathematical)

```python
I = V @ W    # Pure matrix multiply, no hardware effects
```

Use for: **Algorithm development, correctness testing.**

### Level 1: Device-aware

```python
G = quantize(map(W))        # Quantized conductances
G_noisy = G + noise          # Read noise
I = V @ (G⁺_noisy - G⁻_noisy)
```

Includes: conductance quantization, read noise, D2D variation, programming error, stuck-at faults.

Use for: **Accuracy degradation studies, noise sensitivity analysis.**

### Level 2: Hardware-aware

```python
V = DAC(V_digital)           # Input quantization
I = V @ (G⁺_noisy - G⁻_noisy)
output = ADC(I)              # Output quantization
```

Adds: DAC voltage quantization, ADC current-to-digital quantization, clipping.

Use for: **Full system accuracy estimation, ADC/DAC bit-width optimization.**

---

## 5. ADC/DAC Models

### Analog-to-Digital Converter (ADC)

Converts continuous output current to discrete digital values:

```
Quantized = round((x - v_min) / step) × step + v_min
where step = (v_max - v_min) / (2^bits - 1)
```

Key parameter: `bits` — determines resolution. 8-bit ADC = 256 levels.

### Digital-to-Analog Converter (DAC)

Converts digital input to quantized analog voltage. Same math, applied to inputs.

---

## 6. File Format (.analog)

AnalogLib uses a proprietary encrypted binary format inspired by GGUF:

```
┌──────────────────────────────────────┐
│ Magic: 0xAE4C4942 ("ALIB")          │  4 bytes
│ Format version                       │  4 bytes (uint32 LE)
│ Payload length                       │  4 bytes (uint32 LE)
├──────────────────────────────────────┤
│ Encrypted payload (AES-256-GCM)      │
│   → zlib-compressed MessagePack      │
│   → contains metadata, config, and   │
│     conductance tensor data           │
└──────────────────────────────────────┘
```

### Design decisions

- **Encrypted**: Files are opaque without AnalogLib — protects model IP
- **Self-contained**: All metadata, device config, and conductances in one file
- **Versioned**: Format version in header enables forward compatibility
- **Compact**: zlib compression reduces array storage

---

## 7. Plugin Architecture

AnalogLib uses `__init_subclass__` for automatic class discovery:

```python
# Any subclass of Device is auto-registered:
class MyCustomDevice(al.Device):
    def __init__(self, g_min, g_max, num_states, my_param=0.5):
        super().__init__(g_min, g_max, num_states)
        self.my_param = my_param

    def quantize(self, g): ...
    def add_noise(self, g): ...
    def add_variation(self, g): ...

# It's automatically available:
al.Device.registry()  # {"ReRAM": ..., "IdealDevice": ..., "MyCustomDevice": ...}
```

This works for `Device`, `MappingStrategy`, and all other registry-enabled base classes.

---

## References

1. Hu, M. et al. "Memristor‐Based Analog Computation and Neural Network Classification with a Dot Product Engine." *Adv. Mater.* 2018.
2. Chen, P.Y. et al. "NeuroSim: A Circuit-Level Macro Model for Benchmarking Neuro-Inspired Architectures." *IEEE TCAD* 2018.
3. Agarwal, S. et al. "Achieving Ideal Accuracies in Analog Neuromorphic Computing Using Periodic Carry." *Symp. VLSI Tech.* 2017.
4. Joshi, V. et al. "Accurate deep neural network inference using computational phase-change memory." *Nature Commun.* 2020.
