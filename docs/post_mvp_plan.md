# AnalogLib — Post-MVP Architecture & Implementation Roadmap (v2)

> **Key architectural change from v1**: AIR (Analog Intermediate Representation) is the **central contract** of the post-MVP system, not a final phase. All converters, simulators, exporters, and analyzers speak AIR. This avoids converter proliferation and enforces a clean dependency graph.

---

## System Architecture

```text
                     ┌─────────────────────────┐
                     │  PyTorch / TF / ONNX     │
                     └───────────┬─────────────┘
                                 ↓
                     ┌─────────────────────────┐
                     │    Model Converters       │
                     └───────────┬─────────────┘
                                 ↓
                     ┌─────────────────────────┐
                     │           AIR            │
                     │  Analog Intermediate     │
                     │  Representation          │
                     └───────────┬─────────────┘
                                 ↓
                  ┌──────────────┴──────────────┐
                  ↓                             ↓
           Crossbar                     TiledCrossbar
                  │                             │
                  └──────────────┬──────────────┘
                                 ↓
                       Hardware Effects Pipeline
              ┌──────────────────┼──────────────────┐
              ↓                  ↓                  ↓
           IR Drop            Thermal           Drift/Noise
              └──────────────────┼──────────────────┘
                                 ↓
                     ┌─────────────────────────┐
                     │    Simulation Engine      │
                     │   Ideal / Device / HW    │
                     └───────────┬─────────────┘
                                 ↓
                  ┌──────────────┼──────────────┐
                  ↓              ↓              ↓
              Analytics    Visualization      Export
               Power/Area    Dashboard     SPICE / VA
                  └──────────────┼──────────────┘
                                 ↓
                               CLI
```

---

## Interface Contract (Freeze Before Implementing Any Phase)

> **Before starting Phase 1**, the coding implementation must honor these interface contracts. This prevents rewriting Phase 1 code at Phase 3 when exporters and tiled arrays are added.

| Interface | Description |
|-----------|-------------|
| `Device` | Analog memory cell (already implemented) |
| `MappingStrategy` | Weight ↔ conductance (already implemented) |
| `Effect` | Pluggable physical effect applied before/during VMM |
| `Crossbar` | Single-array VMM engine (already implemented) |
| `TiledCrossbar` | Multi-tile VMM, exposes same `.vmm()` interface |
| `SimulationEngine` | Orchestrates layers + effects + peripherals |
| `SimulationMode` | Ideal / Device / Hardware / SPICE (already implemented) |
| `AIRGraph` | Framework-neutral model graph |
| `AIRLayer` | Node in AIRGraph (Crossbar, Activation, etc.) |
| `AnalogModel` | High-level user-facing model wrapping AIR |
| `Exporter` | Produces SPICE/Verilog-A from AIR |
| `Analyzer` | Computes power/area/latency from simulation results |

---

## Dependency Order

```text
Phase 0 (done):   Core MVP — Device, Mapping, Crossbar, Engine, Serialization
Phase 1:          AIR Core Schema (defines all future interfaces)
Phase 2:          Model Converters (PyTorch, TF, ONNX → AIR)
Phase 3:          TiledCrossbar (AIR → tiled execution)
Phase 4:          Hardware Effects (IR Drop, Thermal, Drift as pluggable Effects)
Phase 5:          Analytics (power, area, latency — consumes simulation results)
Phase 6:          Exporters (AIR → SPICE, Verilog-A)
Phase 7:          Visualization Dashboard
Phase 8:          CLI
```

---

## Phase 1: AIR Core Schema (`analoglib.air`)

### Goal
Define the intermediate representation of an analog hardware graph. All converters and exporters target this — not raw `Crossbar` objects. The schema replaces the need for N separate framework-specific paths.

```text
PyTorch ──┐
ONNX ─────┼──→  AIR  ──→  Crossbar / TiledCrossbar / SPICE / CLI
TensorFlow┘
```

### Task 1.1: `AIRGraph` and `AIRLayer` schema
- [ ] Define `AIRLayer` types: `CrossbarLayer`, `ActivationLayer`, `InlineLayer`
- [ ] Define YAML/dict schema: `matrix`, `device`, `mapping`, `tiles`, `peripherals`, `effects`, `simulation_config`
- [ ] Implement `AIRGraph` container with ordered layer list and metadata
- [ ] Implement `AIRGraph.to_dict()` / `AIRGraph.from_dict()` serialization

### Task 1.2: AIR → Crossbar Lowering Pass
- [ ] Implement `air.lower(air_graph) → list[Crossbar | TiledCrossbar]`
- [ ] Populate `SimulationEngine` from a lowered AIR graph
- [ ] Write invariant test: `lower(to_air(W)).vmm(x) ≈ W @ x`

### Task 1.3: `AnalogModel` high-level API
- [ ] Implement `AnalogModel` wrapping an `AIRGraph` with `.compile()`, `.simulate()`, `.report()`, `.save()` / `.load()` methods
- [ ] `.compile(device=..., tile=..., mapping=..., ...)` produces a lowered `SimulationEngine`
- [ ] `.simulate(mode=..., effects=..., adc_bits=..., dac_bits=...)` runs the engine
- [ ] `.report()` prints structured hardware summary

### AIR YAML Example

```yaml
model:
  name: mlp
layers:
  - type: crossbar
    matrix: [128, 256]
    device: ReRAM
    mapping: differential
    tiles: [128, 128]
    peripherals:
      dac: {bits: 8}
      adc: {bits: 8}
    effects:
      ir_drop: {r_wire: 1.0}
      noise: {sigma: 0.03}
```

---

## Phase 2: Model Converters (`analoglib.neural`)

### Goal
Convert trained framework model graphs to AIR, then lower to Crossbar. Converters produce AIR — not Crossbars directly. This keeps the converter logic completely decoupled from hardware implementation.

### Implementation order within Phase 2

1. `nn.Linear` → `CrossbarLayer` AIR node
2. Sequential MLP (stack of Linear + Activations)
3. Activations (ReLU, Sigmoid, Softmax) as `ActivationLayer` nodes
4. `nn.Conv2d` lowered via **im2col → GEMM → CrossbarLayer** (not a direct crossbar model)

### Task 2.1: PyTorch Converter (`analoglib.neural.torch_converter`)
- [ ] Graph walk over `nn.Module` children
- [ ] `nn.Linear` → `AIRLayer(type="crossbar", matrix=[in, out], weights=W)`
- [ ] `nn.ReLU / Sigmoid / Softmax` → `AIRLayer(type="activation", fn=...)`
- [ ] `nn.Conv2d` → im2col lowering → `AIRLayer(type="crossbar")`
- [ ] Unit tests: MNIST MLP forward pass error < 1e-5 in dev mode

### Task 2.2: TensorFlow Converter (`analoglib.neural.tf_converter`)
- [ ] Keras `Dense` → `CrossbarLayer`
- [ ] Keras `Conv2D` → im2col → `CrossbarLayer`
- [ ] Functional + Sequential API support

### Task 2.3: ONNX Importer (`analoglib.neural.onnx_importer`)  
- [ ] Parse `Gemm` / `MatMul` / `Conv` nodes in ONNX graph
- [ ] Emit `AIRLayer` nodes from ONNX nodes
- [ ] Tests: export from both PyTorch and TF, validate AIR equivalence

---

## Phase 3: TiledCrossbar (`analoglib.crossbar.tiled`)

### Goal
Handle weight matrices larger than a single crossbar tile. Expose **identical VMM interface** to `Crossbar` so the simulator doesn't care.

### Primary Invariant Test (must pass before shipping)

```python
W = np.random.randn(512, 256)
x = np.random.randn(512)

ref = Crossbar(512, 256).vmm(x)
tiled = TiledCrossbar.from_matrix(W, tile_shape=(128, 128), device=ReRAM(...))
assert np.allclose(ref, tiled.vmm(x), atol=1e-6)  # ideal mode
```

### Task 3.1: `TiledCrossbar`
- [ ] Split weight matrix into `(ceil(rows/tile_r), ceil(cols/tile_c))` tile grid
- [ ] Pad final tiles to tile dimensions with 0s
- [ ] VMM: partial current accumulation across row-tiles, concatenation across col-tiles
- [ ] Expose `get_conductance()` and `reconstruct_weights()` across all tiles

### Task 3.2: AIR integration
- [ ] `AIRLayer(tiles=[128,128])` lowers to `TiledCrossbar` not `Crossbar`
- [ ] `TiledCrossbar` saves/loads natively in `.analog` format

---

## Phase 4: Hardware Effects (`analoglib.effects`)

### Goal
Rename and reorganize physical effects into a separate pluggable pipeline. IR drop is **not** a device property — it is a system/interconnect effect.

### New module layout

```text
analoglib/
    effects/
        __init__.py
        base.py       ← Effect ABC
        ir_drop.py
        thermal.py
        drift.py
        variation.py
        noise.py      ← (migrated from devices/noise.py)
```

### Effect ABC

```python
class Effect(ABC):
    def apply(self, g: np.ndarray, *, context: EffectContext) -> np.ndarray:
        ...
```

### Crossbar API change (additive, non-breaking)

```python
# New optional parameter
Crossbar(rows, cols, device=reram, effects=[IRDrop(r_wire=1.0), Thermal(T=350)])
```

### Task 4.1: Effect ABC + registry
- [ ] Define `Effect` base class and `EffectContext` (crossbar geometry, voltages, temperature)
- [ ] Implement registry via `__init_subclass__` (same pattern as `Device`)

### Task 4.2: IR Drop (most valuable, implement first)
- [ ] Parasitic wire resistance model (r_wire per cell segment)
- [ ] Node voltage solver via MNA matrix for a tile
- [ ] Benchmark: 64×64 in < 50ms, 128×128 in < 500ms

### Task 4.3: Thermal
- [ ] Temperature-dependent G: `G(T) = G_0 * exp(-E_a / (k*T))`
- [ ] `Thermal(T_kelvin=300, E_a=...)` Effect

### Task 4.4: Temporal Drift
- [ ] `G(t) = G_0 * (t / t_0)^(-nu)` power-law retention drift
- [ ] `Drift(t_seconds=..., nu=...)` Effect

> **Note**: `SimulationMode.HARDWARE` will automatically apply all `effects` registered on a `Crossbar` during VMM. No new enum value needed.

---

## Phase 5: Analytics (`analoglib.analysis`)

> **Important correction**: All power formulas must consume **actual cell voltages from the hardware solver** (not assume ideal input voltage), especially after IR drop is enabled.

### Core metrics to track internally

| Metric | Symbol | Unit |
|--------|--------|------|
| Operations per VMM | ops | 2MN |
| Crossbar read energy | E_read | J |
| ADC conversion energy | E_adc | J |
| DAC conversion energy | E_dac | J |
| Total energy per pass | E_total | J |
| Latency per VMM | t_vmm | s |
| Static array power | P_static | W |
| Dynamic total power | P_total | W |
| Throughput | T | ops/s |
| Energy efficiency | η | TOPS/W |

### Correct power formula (IR drop-aware)

```text
P = Σ_{i,j} V_ij² × G_ij          (uses actual cell voltage, not input voltage)
```

### Task 5.1: `AnalogProfiler`
- [ ] `.profile(simulation_result)` → `AnalogReport`
- [ ] All metrics derived from simulation pass results (not re-computed from scratch)
- [ ] `AnalogReport.summary()` prints structured table

### Task 5.2: Area estimator
- [ ] Cell area from `F²` technology node parameter
- [ ] ADC/DAC peripheral area lookup table by precision

---

## Phase 6: Exporters (`analoglib.exporters`)

### Goal: AIR → circuit representation

```text
AnalogModel
    ↓
   AIR
    ↓
SPICE netlist ─→ ngspice / LTspice
```

This provides a **validation story**: compare AnalogLib simulation against SPICE simulation.

### SPICE dialect support order (implement and validate individually)

1. ngspice (first, open source, easiest to automate)
2. LTspice
3. Spectre (do not claim compatibility without test)
4. HSPICE (do not claim compatibility without test)

### Task 6.1: `SpiceExporter`
- [ ] Cell subcircuit: `R_cell = 1/G`, plus optional C_line per cell
- [ ] Wire parasitics: `R_wire` per metal segment
- [ ] Voltage source drivers on row lines
- [ ] Sense amplifier load resistor on column lines
- [ ] `.export(path, dialect="ngspice")` writes `.cir` file

### Task 6.2: Verilog-A
- [ ] Behavioral ReRAM cell module
- [ ] Top-level crossbar module with embedded conductance parameters

---

## Phase 7: Visualization (`analoglib.visualization`)

> Build the underlying plot functions first. Hold the interactive dashboard until the rest of the API stabilizes.

### Task 7.1: Core plot API (`analoglib.visualization.plots`)
- [ ] `plot_conductance_matrix(crossbar)` — heatmap of G+ and G−
- [ ] `plot_weight_error_histogram(W_orig, W_recon)` — quantization error distribution
- [ ] `plot_noise_sweep(sigmas, snr_list)` — read noise vs SNR
- [ ] `plot_adc_sweep(bits_list, sqnr_list)` — ADC precision vs SQNR

### Task 7.2: Interactive dashboard (after API freeze)
- [ ] Streamlit / Dash interface for live parameter sweeps

---

## Phase 8: CLI (`analoglib.cli`)

```bash
analog compile model.onnx --device reram --tile 128x128 --out model.analog
analog simulate model.analog --mode hardware --adc 8 --dac 8
analog analyze model.analog
analog export-spice model.analog --dialect ngspice --out circuit.cir
analog info model.analog
```

### Task 8.1: CLI entry point
- [ ] `click`-based CLI registered as `analoglib.cli:main` in `pyproject.toml`
- [ ] `analog compile` → calls `neural.{torch|onnx|tf}_converter` → `AIR` → `.analog`
- [ ] `analog simulate` → loads `.analog` → `SimulationEngine.run()`
- [ ] `analog analyze` → loads `.analog` + simulation log → `AnalogProfiler.profile()`
- [ ] `analog export-spice` → loads `.analog` → `SpiceExporter.export()`
- [ ] `analog info` → decrypts metadata and prints layer summary

---

## Target Vision: End-to-End Researcher Workflow

```python
import analoglib as al

model = al.import_model("mnist.onnx")

model.compile(
    device=al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256),
    tile=(128, 128),
    mapping="differential",
)

result = model.simulate(
    mode="hardware",
    effects={"ir_drop": True, "noise": True, "drift": True},
    adc_bits=8,
    dac_bits=8,
)

result.report()
# → structured report: accuracy, energy, latency, TOPS/W
```
