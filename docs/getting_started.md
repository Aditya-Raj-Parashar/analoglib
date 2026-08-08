# AnalogLib — Getting Started Guide

Welcome to **AnalogLib**, an open-source Python library for simulating analog computing systems. This guide will take you from zero to running your first analog crossbar simulation.

---

## What is Analog Computing?

Traditional (digital) computing represents numbers as binary bits and uses transistor logic gates. **Analog computing** uses continuous physical quantities — like electrical current and conductance — to perform calculations directly in hardware.

### Why does this matter?

Neural networks are essentially giant matrix multiplications. In digital hardware, each multiply-accumulate (MAC) operation costs energy. But in an **analog crossbar array**, you can do an entire matrix-vector multiplication in a single step:

```
Input Voltages  →  Crossbar Array  →  Output Currents
    V[i]         ×   G[i,j]         =    I[j]
```

This is possible because of **Ohm's Law** (`I = G × V`) and **Kirchhoff's Current Law** (currents sum at a node). The crossbar computes `I = G @ V` in one physical step — potentially thousands of times more energy-efficient than digital.

### The catch

Real analog devices aren't perfect:
- Conductances can only take **discrete values** (quantization)
- Devices have **noise** (random fluctuations during read)
- Each device is **slightly different** (device-to-device variation)
- Some devices get **stuck** (stuck-at faults)

AnalogLib lets you model all of these effects in software.

---

## Installation

```bash
# Basic install
pip install -e .

# With development tools (pytest)
pip install -e ".[dev]"

# With visualization support
pip install -e ".[viz]"
```

**Requirements**: Python ≥ 3.10, NumPy ≥ 1.24

---

## Quick Start (5 minutes)

### Step 1: Import the library

```python
import analoglib as al
import numpy as np
```

### Step 2: Create a device

A "device" models a single memory cell. The most common is **ReRAM** (Resistive RAM):

```python
device = al.ReRAM(
    g_min=1e-6,       # Minimum conductance: 1 µS
    g_max=100e-6,     # Maximum conductance: 100 µS
    num_states=256,    # 256 programmable levels (8-bit)
)
```

> **Think of it as**: Each cell is a tiny variable resistor that can be set to 256 different resistance values.

### Step 3: Create a crossbar

A "crossbar" is a grid of devices. Each row is an input, each column is an output:

```python
crossbar = al.Crossbar(
    rows=4,          # 4 inputs
    cols=3,          # 3 outputs
    device=device,
    differential=True,  # Use G+/G- pairs for signed weights
)
```

### Step 4: Load weights

Take any weight matrix (like from a neural network layer) and load it:

```python
# Random weights for demonstration
weights = np.array([
    [ 0.5, -0.3,  0.8],
    [-0.2,  0.7, -0.1],
    [ 0.9, -0.5,  0.4],
    [-0.6,  0.1,  0.3],
])

crossbar.load_weights(weights)
```

Behind the scenes, AnalogLib:
1. Normalizes the weights to [0, 1]
2. Maps them to conductance pairs (G⁺, G⁻)
3. Quantizes to the nearest of 256 device levels

### Step 5: Compute!

Apply input voltages and get output currents:

```python
input_voltage = np.array([0.5, 0.3, 0.8, 0.2])
output = crossbar.vmm(input_voltage)  # Vector-Matrix Multiply

print(f"Output: {output}")
# Output shape: (3,)  — one value per column
```

### Step 6: Save & share

```python
al.save("my_model.analog", [crossbar], model_name="demo")

# Later, or on another machine:
loaded = al.load("my_model.analog")
loaded_crossbar = loaded["crossbars"][0]
```

The `.analog` file is encrypted and self-contained. Only AnalogLib can open it.

---

## Understanding the Pipeline

```
Neural Network Weights    (what you start with)
        ↓
Weight Normalization      (scale to [-1, 1])
        ↓
Conductance Mapping       (W → G⁺, G⁻)
        ↓
Device Quantization       (snap to 256 levels)
        ↓
Crossbar Storage          (conductance matrices)
        ↓
Input Voltage Applied     (your input data)
        ↓
Ohmic Current: I = G·V   (physics does the work)
        ↓
Column Current Summation  (Kirchhoff's Law)
        ↓
Differential Readout      (I⁺ - I⁻ = result)
        ↓
Output                    (your inference result)
```

---

## Key Concepts

### Differential Mapping

Since conductance is always positive (G ≥ 0), but weights can be negative, we use **two devices per weight**:

```
Positive weight (+0.7):  G⁺ = high,  G⁻ = low   →  G⁺ - G⁻ > 0  ✓
Negative weight (-0.4):  G⁺ = low,   G⁻ = high   →  G⁺ - G⁻ < 0  ✓
Zero weight     ( 0.0):  G⁺ = mid,   G⁻ = mid    →  G⁺ - G⁻ = 0  ✓
```

### Simulation Modes

AnalogLib supports 3 levels of realism:

| Mode | What it includes | Use case |
|------|-----------------|----------|
| `ideal` | Perfect math only | Debugging, baseline |
| `device` | + quantization, noise, variation | Research accuracy studies |
| `hardware` | + ADC/DAC quantization | Full system simulation |

```python
engine = al.SimulationEngine(crossbars=[crossbar])

# Compare all modes:
results = engine.run_comparison(input_voltage)
print(results["ideal"])     # Perfect
print(results["device"])    # With noise
print(results["hardware"])  # Full pipeline
```

### Devices

| Device | Description | When to use |
|--------|-------------|-------------|
| `IdealDevice` | Perfect, no noise | Baseline comparisons |
| `ReRAM` | Realistic with all non-idealities | Research simulations |

---

## Common Patterns

### Comparing ideal vs. noisy inference

```python
# Ideal device (baseline)
ideal = al.Crossbar(32, 16, device=al.IdealDevice())
ideal.load_weights(W, quantize=False)
out_ideal = ideal.vmm(V)

# ReRAM with noise
noisy = al.Crossbar(32, 16, device=al.ReRAM(
    g_min=1e-6, g_max=100e-6, num_states=64,
    read_noise_sigma=0.03,
    d2d_variation_sigma=0.02,
))
noisy.load_weights(W)
out_noisy = noisy.vmm(V, noise=True, mode=al.SimulationMode.DEVICE)

error = np.max(np.abs(out_ideal - out_noisy))
print(f"Max error from noise: {error:.6f}")
```

### Sweeping ADC precision

```python
for bits in [4, 6, 8, 10]:
    adc = al.ADC(bits=bits, v_min=-1.0, v_max=1.0)
    quantized = adc.convert(output)
    error = np.mean(np.abs(output - quantized))
    print(f"ADC {bits}-bit → mean error: {error:.6f}")
```

### Checking quantization loss

```python
W_reconstructed = crossbar.reconstruct_weights()
max_error = np.max(np.abs(weights - W_reconstructed))
print(f"Weight quantization error: {max_error:.6f}")
```

---

## What's Next?

- **API Reference** → `docs/api_reference.md` — every function, every parameter
- **Core Concepts** → `docs/core_concepts.md` — the physics and math behind it
- **Examples** → `examples/demo.py` — interactive runnable demo
