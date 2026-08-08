# AIR (Analog Intermediate Representation) Architecture Guide

## Overview

**Analog Intermediate Representation (AIR)** is AnalogLib's standardized internal schema for neural network models targeting analog in-memory computing (IMC) hardware.

AIR serves as the clean decoupling contract between high-level neural network framework converters (PyTorch, TensorFlow, ONNX, NumPy) and low-level physical crossbar simulation engines (ReRAM, Phase-Change Memory, Flash, ideal crossbars).

```text
 ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
 │ PyTorch nn.Module    │   │ TensorFlow Model     │   │ ONNX Graph           │
 └──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       ↓
                            ┌──────────────────────┐
                            │      AIRGraph        │  <-- JSON/Dict Serializable Schema
                            └──────────┬───────────┘
                                       ↓ Lowering Pass (air.lower)
                            ┌──────────────────────┐
                            │   SimulationEngine   │  <-- Executes Ideal/Device/Hardware VMM
                            └──────────────────────┘
```

---

## Core Schema Data Structures (`analoglib.air.schema`)

### 1. `AIRGraph`
The top-level container representing an end-to-end model graph.

- **Attributes**:
  - `name: str` — Model graph identifier.
  - `layers: List[AIRLayer]` — Ordered sequence of AIR layers.
  - `metadata: Dict[str, Any]` — Arbitrary global metadata.

- **Methods**:
  - `add_layer(layer: AIRLayer) -> AIRGraph` — Append a layer and return self for chaining.
  - `get_layer(name: str) -> AIRLayer` — Retrieve layer by name.
  - `validate() -> None` — Enforce graph invariants (non-empty, unique layer names, weight matrix presence).
  - `to_dict() -> Dict[str, Any]` — Serialize graph to pure Python dictionary / JSON.
  - `from_dict(d: Dict[str, Any]) -> AIRGraph` — Deserialize from dictionary.

### 2. `AIRLayer`
Represents an individual computational step in the network.

- **Supported Layer Types (`LayerType`)**:
  - `LayerType.CROSSBAR` — Vector-matrix multiplication mapped to physical/ideal crossbar array.
  - `LayerType.ACTIVATION` — Non-linear activation function (`relu`, `sigmoid`, `tanh`, `softmax`).
  - `LayerType.INLINE` — Passthrough node or custom mathematical operation.

- **Peripheral & Effect Configs**:
  - `PeripheralConfig`: ADC/DAC precision (bits) and voltage bounds (`v_min`, `v_max`).
  - `EffectConfig`: Physical non-idealities (IR drop wire resistance `r_wire`, Arrhenius thermal parameters `E_a`, `T_ref`, power-law drift `nu`, `t_0`).

---

## Lowering Compiler Pass (`analoglib.air.lower`)

The lowering pass converts an abstract `AIRGraph` into an executable `SimulationEngine`.

```python
from analoglib.air import lower, AIRGraph, AIRLayer, LayerType
import numpy as np

# 1. Construct AIRGraph
g = AIRGraph(name="demo")
g.add_layer(AIRLayer(
    layer_type=LayerType.CROSSBAR,
    name="fc0",
    matrix_shape=(128, 64),
    weights=np.random.randn(128, 64),
))

# 2. Lower graph to SimulationEngine
engine = lower(
    g,
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256),
    adc_bits=8,
    dac_bits=8,
)

# 3. Run simulation
output = engine.run(x, mode="hardware")
```

### Compiler Invariants Guaranteed

1. **Ideal Equivalence**:
   $$\text{lower}(\text{AIRGraph}).\text{run}(x, \text{"ideal"}) \equiv \text{Crossbar}(W).\text{vmm}(x, \text{"ideal"})$$
2. **Determinism**: Lowering the same AIR graph with identical seeds yields identical conductances and VMM outputs.
3. **Tile Invariance**: When `tile_shape` is specified, `TiledCrossbar.vmm(x)` produces the exact same physical result as a unified `Crossbar.vmm(x)` under ideal conditions.

---

## High-Level `AnalogModel` API (`analoglib.air.model`)

`AnalogModel` wraps `AIRGraph` and `SimulationEngine` into a fluent, user-friendly interface:

```python
import analoglib as al
import numpy as np

# Load weights from framework or NumPy
W1 = np.random.randn(784, 128)
W2 = np.random.randn(128, 10)

# Build, Compile, and Simulate
model = al.AnalogModel.from_numpy([W1, W2], activations=["relu", "softmax"])

model.compile(
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.01),
    adc_bits=8,
    dac_bits=8,
    r_wire=1.0,      # IR Drop effect
    E_a=0.1,         # Thermal effect
    nu=0.05,         # Retention drift effect
)

result = model.simulate(x_input, mode="hardware")

# Print hardware profiling & error analysis report
result.report()
```
