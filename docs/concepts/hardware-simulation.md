# Hardware Simulation Modes & Execution Pipeline

`SimulationEngine` orchestrates multi-layer inference across physical crossbar arrays, peripherals (ADC/DAC), and physical non-ideality solvers.

---

## 1. Simulation Pipeline Diagram

```text
Digital Input Activations  x_in  ∈ [0, 1]
           │
           ▼
┌──────────────────────────────────────┐
│  DAC Converter  (Quantizes to V_in)   │
└──────────────────┬───────────────────┘
                   │ V_in (Volts)
                   ▼
┌──────────────────────────────────────┐
│  Crossbar VMM Array (G+ and G-)      │
│  Applied Effects: IRDrop, Thermal,   │
│  Drift, Read Noise                   │
└──────────────────┬───────────────────┘
                   │ I_out (Amperes)
                   ▼
┌──────────────────────────────────────┐
│  ADC Converter (Clipping & Quant)    │
└──────────────────┬───────────────────┘
                   │ y_digital
                   ▼
       Activation Function (ReLU/Softmax)
```

---

## 2. Comparison of Simulation Modes

| Mode Enum | String | Speed | Physical Fidelity | Operations Simulated |
| :--- | :--- | :--- | :--- | :--- |
| `SimulationMode.IDEAL` | `"ideal"` | $\sim 100\times$ faster | Unquantized math | Ideal matrix multiplication $y = x @ W$ |
| `SimulationMode.DEVICE` | `"device"` | Standard | Medium | NVM conductance state quantization ($N_{\text{states}}$) + Gaussian read noise |
| `SimulationMode.HARDWARE` | `"hardware"` | Rigorous | High | Full pipeline: DAC $\rightarrow$ Crossbar $\rightarrow$ IR Drop / Thermal / Drift $\rightarrow$ Read Noise $\rightarrow$ ADC |

---

## 3. Code Example

```python
import analoglib as al
import numpy as np

# Build 2-layer engine
device = al.ReRAM(num_states=256, read_noise_sigma=0.01)
xb1 = al.Crossbar(64, 32, device=device)
xb2 = al.Crossbar(32, 10, device=device)

xb1.load_weights(np.random.randn(64, 32))
xb2.load_weights(np.random.randn(32, 10))

engine = al.SimulationEngine(
    crossbars=[xb1, xb2],
    adc=al.ADC(bits=8),
    dac=al.DAC(bits=8),
)

V_in = np.random.uniform(0, 1, 64)

# Run in all three modes
y_ideal = engine.run(V_in, mode="ideal")
y_device = engine.run(V_in, mode="device")
y_hardware = engine.run(V_in, mode="hardware")

print("Ideal output:", y_ideal.shape)
print("Device output:", y_device.shape)
print("Hardware output:", y_hardware.shape)
```
