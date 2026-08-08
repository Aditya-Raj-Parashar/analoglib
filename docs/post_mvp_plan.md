# AnalogLib — Post-MVP Detailed Implementation Plan & Roadmap

This document outlines the detailed architectural plans, technical tasks, and subtasks for expanding **AnalogLib** beyond the MVP.

---

## Roadmap Overview

```
Phase 1: Neural Network Importers & Conversion (PyTorch / TensorFlow / ONNX)
Phase 2: Advanced Scalable Hardware Simulation (TiledCrossbar, IR Drop, Thermal)
Phase 3: Circuit & Hardware Exporters (SPICE Netlists, Verilog-A)
Phase 4: Analytics & Performance Profiler (Power, Energy, Area, Latency)
Phase 5: Visualization & Interactive Dashboard (Matplotlib, Bokeh, Web UI)
Phase 6: CLI & Tooling Pipeline (`analog` CLI, compilation targets)
Phase 7: Analog Intermediate Representation (AIR)
```

---

## Phase 1: Neural Network Importers & Model Conversion

### Goal
Allow researchers to pass any trained PyTorch or TensorFlow model directly into AnalogLib, converting standard `Linear` / `nn.Linear` / `Dense` layers into simulated analog crossbar arrays automatically.

### Tasks & Subtasks

#### Task 1.1: PyTorch Model Converter (`analoglib.neural.torch_converter`)
- [ ] **Subtask 1.1.1**: Implement layer inspection for `nn.Linear` and `nn.Conv2d` layers.
- [ ] **Subtask 1.1.2**: Implement weight extraction and conversion to `Crossbar` objects using specified `Device` and `MappingStrategy`.
- [ ] **Subtask 1.1.3**: Implement PyTorch custom autograd module (`AnalogLinear` / `AnalogConv2d`) for PyTorch-native forward passes using `Crossbar.vmm`.
- [ ] **Subtask 1.1.4**: Add support for activation functions (`ReLU`, `Sigmoid`, `Softmax`) between crossbar layers.
- [ ] **Subtask 1.1.5**: Write unit tests for PyTorch converter using standard models (e.g., MNIST MLP).

#### Task 1.2: TensorFlow / Keras Model Converter (`analoglib.neural.tf_converter`)
- [ ] **Subtask 1.2.1**: Implement Keras `Dense` and `Conv2D` layer weight extractor.
- [ ] **Subtask 1.2.2**: Implement custom Keras layer (`AnalogDenseLayer`) wrapping AnalogLib VMM.
- [ ] **Subtask 1.2.3**: Support functional and sequential Keras model conversion.
- [ ] **Subtask 1.2.4**: Write unit tests for TensorFlow/Keras model conversion.

#### Task 1.3: ONNX Model Importer (`analoglib.neural.onnx_importer`)
- [ ] **Subtask 1.3.1**: Parse ONNX model graph to extract `Gemm` and `Conv` nodes and tensor weights.
- [ ] **Subtask 1.3.2**: Map ONNX computational graph to sequential `SimulationEngine` crossbar layers.
- [ ] **Subtask 1.3.3**: Unit test ONNX model loading.

---

## Phase 2: Advanced Scalable Hardware Simulation

### Goal
Simulate large neural network weights that exceed single crossbar dimensions (e.g., 2048×2048 weight matrix mapped to 128×128 tiles), plus physical circuit non-idealities like wire resistance (IR drop) and sneak paths.

### Tasks & Subtasks

#### Task 2.1: Tiled Crossbar Engine (`analoglib.crossbar.tiled`)
- [ ] **Subtask 2.1.1**: Implement `TiledCrossbar(matrix_rows, matrix_cols, tile_rows, tile_cols)`.
- [ ] **Subtask 2.1.2**: Implement automatic weight splitting across a 2D grid of `Crossbar` tiles.
- [ ] **Subtask 2.1.3**: Implement block matrix-vector multiplication with partial current accumulation across column tiles.
- [ ] **Subtask 2.1.4**: Unit tests for weight tiling and exact mathematical equivalence.

#### Task 2.2: Parasitic & IR Drop Simulator (`analoglib.hardware.ir_drop`)
- [ ] **Subtask 2.2.1**: Implement parasitic wire resistance model (`r_wire` per cell segment).
- [ ] **Subtask 2.2.2**: Build node voltage solver (iterative linear solver / MNA matrix solver) to compute line voltage drops.
- [ ] **Subtask 2.2.3**: Integrate IR drop degradation into `Crossbar.vmm` in `SimulationMode.HARDWARE`.
- [ ] **Subtask 2.2.4**: Benchmark performance of IR drop solver for 64×64 and 128×128 crossbars.

#### Task 2.3: Thermal & Drift Noise Models (`analoglib.devices.thermal`)
- [ ] **Subtask 2.3.1**: Implement temperature-dependent conductance drift `G(T) = G_0 * exp(-E_a / (k * T))`.
- [ ] **Subtask 2.3.2**: Implement temporal conductance relaxation/drift `G(t) = G_0 * (t / t_0)^(-nu)`.
- [ ] **Subtask 2.3.3**: Add thermal/drift noise models into `ReRAM` device class.

---

## Phase 3: Circuit Exporters (SPICE & Verilog-A)

### Goal
Export crossbars and trained weights into industry-standard circuit simulation files (SPICE netlists, Verilog-A behavioral models) for chip design verification in Cadence Spectre, LTspice, or ngspice.

### Tasks & Subtasks

#### Task 3.1: SPICE Netlist Generator (`analoglib.exporters.spice`)
- [ ] **Subtask 3.1.1**: Implement `SpiceExporter(crossbar, format="ngspice" | "ltspice" | "spectre")`.
- [ ] **Subtask 3.1.2**: Generate subcircuit subblocks for individual ReRAM cells with specified conductance values (`R_cell = 1/G`).
- [ ] **Subtask 3.1.3**: Add parasitics (wire resistors `R_wire` and line capacitors `C_line`).
- [ ] **Subtask 3.1.4**: Add input voltage source drivers and output sense amplifier load resistors.
- [ ] **Subtask 3.1.5**: Write netlist validation tests and automated ngspice execution runner.

#### Task 3.2: Verilog-A Model Generator (`analoglib.exporters.veriloga`)
- [ ] **Subtask 3.2.1**: Template behavioral Verilog-A module for compact ReRAM cell simulation.
- [ ] **Subtask 3.2.2**: Generate top-level Verilog-A crossbar module with embedded conductance parameters.
- [ ] **Subtask 3.2.3**: Test output syntax compatibility with Synopsys HSPICE and Cadence Spectre.

---

## Phase 4: Analytics & Performance Profiler

### Goal
Provide hardware researchers with accurate estimations of power consumption, energy efficiency (TOPS/W), chip area (μm²), and latency (ns).

### Tasks & Subtasks

#### Task 4.1: Power & Energy Estimator (`analoglib.analysis.power`)
- [ ] **Subtask 4.1.1**: Calculate static array power `P_static = Σ (V_i^2 * G_ij)`.
- [ ] **Subtask 4.1.2**: Calculate dynamic peripheral power (ADC/DAC conversion energy per bit).
- [ ] **Subtask 4.1.3**: Compute total energy per VMM pass `E_total = P_total * t_read`.
- [ ] **Subtask 4.1.4**: Calculate energy efficiency metric: `TOPS/W = (2 * M * N) / (E_total * 10^12)`.

#### Task 4.2: Area & Layout Profiler (`analoglib.analysis.area`)
- [ ] **Subtask 4.2.1**: Estimate crossbar array area based on cell feature size (`F²` units, e.g., 4F² per 1R cell).
- [ ] **Subtask 4.2.2**: Estimate ADC/DAC peripheral circuit area based on bit precision lookup tables.
- [ ] **Subtask 4.2.3**: Produce summary hardware report (`print_hardware_summary(xbar)`).

---

## Phase 5: Visualization & Interactive Dashboard

### Goal
Provide graphical tools to plot conductance distributions, weight error heatmaps, noise sweeps, and interactive web dashboards.

### Tasks & Subtasks

#### Task 5.1: Matplotlib Plotting Suite (`analoglib.visualization.plots`)
- [ ] **Subtask 5.1.1**: `plot_conductance_matrix(crossbar)` — heatmap of G⁺ and G⁻ values.
- [ ] **Subtask 5.1.2**: `plot_weight_error_histogram(w_original, w_reconstructed)` — distribution of quantization errors.
- [ ] **Subtask 5.1.3**: `plot_noise_sweep(snr_list, accuracy_list)` — accuracy vs. read noise curve.
- [ ] **Subtask 5.1.4**: `plot_adc_precision_sweep(bits_list, accuracy_list)` — accuracy vs. ADC bits curve.

#### Task 5.2: Web Dashboard (`analoglib.visualization.dashboard`)
- [ ] **Subtask 5.2.1**: Build lightweight Streamlit / Dash web interface for interactive crossbar tuning.
- [ ] **Subtask 5.2.2**: Allow real-time slider controls for `g_min`, `g_max`, `num_states`, `read_noise_sigma`, and `adc_bits`.

---

## Phase 6: CLI & Tooling Pipeline

### Goal
Command-line interface for batch compiling models, running simulations, and generating hardware reports without writing Python code.

### Tasks & Subtasks

#### Task 6.1: Command-Line Interface (`analoglib.cli`)
- [ ] **Subtask 6.1.1**: Implement `analog compile --input model.onnx --device reram --out model.analog`.
- [ ] **Subtask 6.1.2**: Implement `analog simulate --model model.analog --input data.npy --mode hardware`.
- [ ] **Subtask 6.1.3**: Implement `analog export-spice --model model.analog --out circuit.cir`.
- [ ] **Subtask 6.1.4**: Implement `analog info model.analog` — inspect encrypted metadata and layer info.

---

## Phase 7: Analog Intermediate Representation (AIR)

### Goal
A standardized intermediate representation (AIR) defining crossbar topology, device mapping, routing, and peripheral configurations, enabling interoperability between hardware compilers and analog simulators.

### Tasks & Subtasks

#### Task 7.1: AIR Specification & Compiler (`analoglib.air`)
- [ ] **Subtask 7.1.1**: Define JSON/YAML schema for AIR graph representations.
- [ ] **Subtask 7.1.2**: Implement graph lowering pass from PyTorch/ONNX to AIR graph nodes.
- [ ] **Subtask 7.1.3**: Implement AIR runtime interpreter executing VMM passes through `SimulationEngine`.
