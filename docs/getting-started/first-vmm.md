# Performing Your First Analog VMM

Vector-Matrix Multiplication (VMM) is the fundamental kernel of analog in-memory computing. This guide demonstrates how to initialize a crossbar array, load weights, and execute VMM across simulation modes.

---

## Complete Self-Contained VMM Example

```python
import analoglib as al
import numpy as np

# 1. Define physical device (ReRAM with 256 states)
device = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01)

# 2. Instantiate a 32x16 differential crossbar
xbar = al.Crossbar(rows=32, cols=16, device=device, differential=True)

# 3. Generate and load weight matrix
W = np.random.uniform(-1.0, 1.0, (32, 16))
xbar.load_weights(W, quantize=True)

# 4. Generate input voltage vector (32 rows)
V_in = np.random.uniform(0.0, 1.0, 32)

# 5. Execute VMM in three simulation modes:
# Mode 1: IDEAL (pure mathematical V @ W)
y_ideal = xbar.vmm(V_in, mode=al.SimulationMode.IDEAL)

# Mode 2: DEVICE (conductance quantization + read noise)
y_device = xbar.vmm(V_in, noise=True, mode=al.SimulationMode.DEVICE)

# Mode 3: HARDWARE (peripheral ADC/DAC pipeline)
engine = al.SimulationEngine(
    crossbars=[xbar],
    adc=al.ADC(bits=8, v_min=-500e-6, v_max=500e-6),
    dac=al.DAC(bits=8, v_min=0.0, v_max=1.0),
)
y_hardware = engine.run(V_in, mode="hardware")

# 6. Reconstruct effective weights from conductances
W_recon = xbar.reconstruct_weights()
max_error = np.max(np.abs(W - W_recon))

print("Ideal output shape:", y_ideal.shape)
print("Device output (first 3):", y_device[:3])
print("Hardware output (first 3):", y_hardware[:3])
print(f"Max Weight Quantization Error: {max_error:.6f}")
```
