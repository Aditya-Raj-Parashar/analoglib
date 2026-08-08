# Getting Started with AnalogLib

Welcome to **AnalogLib**, an open-source Python library for simulating analog in-memory computing (IMC) and neural network inference on resistive crossbar arrays (ReRAM, PCM, Flash).

This guide covers everything from your first single crossbar simulation to multi-layer neural network lowering, physical hardware non-idealities, profiling, and SPICE netlist export.

---

## 1. Installation

### Requirements
- Python 3.10+
- NumPy, PyCryptodome, MsgPack, PyTest

### Install from Source
```bash
git clone https://github.com/Aditya-Raj-Parashar/analoglib.git
cd analoglib
pip install -e .
```

### Optional Dependencies
- **PyTorch**: For converting PyTorch `nn.Module` models (`pip install torch`)
- **Matplotlib**: For visualization plots (`pip install matplotlib`)

---

## 2. Starter Tutorial: 5-Minute Quick Start

### Step 1: Create a ReRAM Crossbar
```python
import analoglib as al
import numpy as np

# Define physical ReRAM device (1 µS to 100 µS, 256 discrete states)
device = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01)

# Initialize a 128x64 differential crossbar
xbar = al.Crossbar(rows=128, cols=64, device=device, differential=True)

# Load weight matrix (mapped automatically to G+ and G- conductances)
W = np.random.uniform(-1.0, 1.0, (128, 64))
xbar.load_weights(W, quantize=True)
```

### Step 2: Perform Analog Matrix-Vector Multiplication (VMM)
```python
# Create input voltage vector
V_in = np.random.uniform(0.0, 1.0, 128)

# Execute VMM in three different simulation modes:
# 1. IDEAL: Mathematical V @ W (unquantized, no noise)
y_ideal = xbar.vmm(V_in, mode=al.SimulationMode.IDEAL)

# 2. DEVICE: Quantized conductances with read noise
y_device = xbar.vmm(V_in, mode=al.SimulationMode.DEVICE)

# 3. HARDWARE: Complete peripheral circuit pipeline (DAC -> VMM -> ADC)
engine = al.SimulationEngine(
    crossbars=[xbar],
    adc=al.ADC(bits=8, v_min=-500e-6, v_max=500e-6),
    dac=al.DAC(bits=8, v_min=0.0, v_max=1.0)
)
y_hardware = engine.run(V_in, mode="hardware")
```

---

## 3. High-Level Workflow: PyTorch to Analog Simulation

Using **AIR (Analog Intermediate Representation)**, you can convert PyTorch models directly to analog simulation engines with zero manual mapping required:

```python
import torch
import torch.nn as nn
import analoglib as al

# 1. Define and train your PyTorch model
torch_model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

# 2. Convert PyTorch model to AnalogModel
model = al.AnalogModel.from_torch(torch_model)

# 3. Compile targeting physical ReRAM crossbars + ADC/DAC + Hardware Effects
model.compile(
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01),
    adc_bits=8,
    dac_bits=8,
    r_wire=1.0,    # Parasitic IR drop (wire resistance per cell)
    E_a=0.1,       # Thermal Arrhenius conductance scaling
    nu=0.05,       # Temporal retention drift
)

# 4. Simulate inference
x_input = np.random.uniform(0, 1, 784)
result = model.simulate(x_input, mode="hardware")

# 5. Generate Hardware Performance & Accuracy Report
result.report()
```

---

## 4. Large Matrix Weight Tiling (`TiledCrossbar`)

When your weight matrix exceeds the physical dimensions of a single crossbar array (e.g., a 1024x512 matrix on 128x64 physical tiles), use `TiledCrossbar`:

```python
from analoglib import TiledCrossbar, ReRAM

# Partition 1024x512 weight matrix into 128x64 physical crossbar tiles
tiled_xbar = TiledCrossbar.from_matrix(
    W_large,
    tile_shape=(128, 64),
    device=ReRAM(g_min=1e-6, g_max=100e-6, num_states=256),
)

# Performs VMM across all 64 tiles (8x8 grid) and accumulates output currents
y_out = tiled_xbar.vmm(V_in)
```

---

## 5. Circuit Export to SPICE

Export your loaded crossbar arrays into a standalone SPICE netlist for circuit validation in **ngspice** or **LTspice**:

```python
from analoglib.exporters import SpiceExporter

exporter = SpiceExporter(dialect="ngspice", R_load=1e3)
exporter.export("my_circuit.cir", [xbar])
```

---

## 6. Command Line Interface (CLI)

AnalogLib includes a powerful CLI utility for inspecting, simulating, and profiling `.analog` file artifacts:

```bash
# View model architecture and device metadata
analog info model.analog

# Run inference simulation
analog simulate model.analog --mode hardware

# Profile array power, latency, area, and TOPS/W
analog profile model.analog

# Export to SPICE netlist
analog export-spice model.analog --out circuit.cir --dialect ngspice
```
