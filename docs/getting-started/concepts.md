# Core Concepts for Beginners

AnalogLib simulates analog in-memory computing (IMC) architectures by modeling physical electrical circuits instead of ideal floating-point arithmetic.

---

## 1. Digital vs. Analog Computing

| Parameter | Digital Computing (CPU/GPU) | Analog In-Memory Computing |
| :--- | :--- | :--- |
| **Data Representation** | 0s and 1s (binary registers) | Continuous/Discrete voltage & conductance |
| **Computation Location** | Separate ALU and Memory (Von Neumann) | Directly inside NVM crossbar memory array |
| **VMM Kernel** | Sequential multiply-accumulate loop | Parallel Ohm's & Kirchhoff's current accumulation |
| **Energy Consumption** | Memory bus transfer bound ($pJ$) | Current accumulation ($fJ$) |

---

## 2. Mathematical vs. Physical Domain

Understanding the boundary between dimensionless mathematics and physical units is essential:

```text
Digital Model Domain (Dimensionless)
  • Weights: W ∈ [-w_max, w_max]
  • Input Activations: x ∈ [0.0, 1.0]
  • Math Result: y = x @ W
                    │
                    ▼ Conductance Mapping & DAC
Physical Hardware Domain (Physical Units)
  • Cell Conductances: G ∈ [g_min, g_max] (Siemens, S)
  • Wordline Voltages: V ∈ [v_min, v_max] (Volts, V)
  • Column Currents: I = V @ (G+ - G-) (Amperes, A)
                    │
                    ▼ ADC & Reconstruction
Reconstructed Digital Output
  • Output y_recon = I / alpha
```

---

## 3. The AIR Contract

**Analog Intermediate Representation (AIR)** decouples high-level machine learning frameworks from low-level NVM hardware simulators:

```text
PyTorch / NumPy ───► AIRGraph (Schema) ───► lower() ───► Crossbar / SPICE / CLI
```

Every converter target emits an `AIRGraph`. Every simulator engine consumes an `AIRGraph`.
