# AnalogLib API Reference

Complete reference for every package, module, class, function, parameter, return type, and exception in `analoglib`.

---

## Table of Contents
1. [`analoglib` Top-Level Package](#1-analoglib-top-level-package)
2. [`analoglib.air` — Analog Intermediate Representation](#2-analoglibair-analog-intermediate-representation)
3. [`analoglib.crossbar` — Crossbars & Tiling](#3-analoglibcrossbar-crossbars--tiling)
4. [`analoglib.devices` — ReRAM & Memristive Devices](#4-analoglibdevices-reram--memristive-devices)
5. [`analoglib.effects` — Physical Non-Idealities](#5-analoglibeffects-physical-non-idealities)
6. [`analoglib.mapping` — Weight Conductance Mapping](#6-analoglibmapping-weight-conductance-mapping)
7. [`analoglib.adc_dac` — Quantized Peripheral Converters](#7-analoglibadc_dac-quantized-peripheral-converters)
8. [`analoglib.simulation` — Execution Engine](#8-analoglibsimulation-execution-engine)
9. [`analoglib.analysis` — Hardware Profiler & Analytics](#9-analoglibanalysis-hardware-profiler--analytics)
10. [`analoglib.exporters` — SPICE Circuit Exporter](#10-analoglibexporters-spice-circuit-exporter)
11. [`analoglib.neural` — Model Converters](#11-analoglibneural-model-converters)
12. [`analoglib.visualization` — Plotting Utilities](#12-analoglibvisualization-plotting-utilities)
13. [`analoglib.cli` — Command Line Interface](#13-analoglibcli-command-line-interface)

---

## 1. `analoglib` Top-Level Package

```python
import analoglib as al
```

### Global Functions
- `al.save(filepath: str, crossbars: List[Crossbar], meta: Dict[str, Any] = None) -> None`: Serializes crossbar models to encrypted `.analog` binary format (AES-256-GCM + MsgPack).
- `al.load(filepath: str) -> Dict[str, Any]`: Loads and decrypts `.analog` file, returning dictionary with keys `"crossbars"` and `"meta"`.
- `al.set_seed(seed: int) -> None`: Sets global random seed for reproducibility across device noise models and variation.

---

## 2. `analoglib.air` (Analog Intermediate Representation)

### `AIRGraph`
- `AIRGraph(name: str = "model", description: str = "")`: Graph schema representing an analog neural network.
- `.add_layer(layer: AIRLayer) -> AIRGraph`: Append layer.
- `.get_layer(name: str) -> AIRLayer`: Lookup layer by name.
- `.validate() -> None`: Verify graph structural invariants.
- `.to_dict() -> Dict[str, Any]`: Export graph as dictionary.
- `.from_dict(d: Dict[str, Any]) -> AIRGraph`: Load graph from dictionary.

### `AIRLayer`
- `AIRLayer(layer_type: LayerType, name: str, matrix_shape: Tuple[int, int] = None, weights: np.ndarray = None, activation_fn: ActivationFn = None, peripherals: PeripheralConfig = None, effects: List[EffectConfig] = None)`: Node in AIRGraph.

### `lower` Compiler Pass
- `al.lower(air_graph: AIRGraph, device: Device = None, adc_bits: int = None, dac_bits: int = None, quantize: bool = True) -> SimulationEngine`: Compiles AIRGraph into an executable `SimulationEngine`.

### `AnalogModel`
- `AnalogModel(air_graph: AIRGraph)`: High-level model wrapper.
- `AnalogModel.from_numpy(weights: List[np.ndarray], activations: List[str] = None) -> AnalogModel`: Factory from NumPy weight matrices.
- `AnalogModel.from_torch(module: nn.Module) -> AnalogModel`: Factory from PyTorch model.
- `.compile(...) -> AnalogModel`: Target specific physical devices and peripherals.
- `.simulate(x: np.ndarray, mode: str = "ideal") -> SimulationResult`: Run inference.

---

## 3. `analoglib.crossbar` (Crossbars & Tiling)

### `Crossbar`
- `Crossbar(rows: int, cols: int, device: Device = None, mapping: MappingStrategy = None, differential: bool = True, effects: List[Effect] = None)`: Single crossbar array engine.
- `.load_weights(W: np.ndarray, quantize: bool = True) -> None`: Map weights to physical conductances $G^+$ and $G^-$.
- `.vmm(V: np.ndarray, noise: bool = False, mode: SimulationMode = "ideal") -> np.ndarray`: Execute vector-matrix multiplication $I = V \cdot (G^+ - G^-)$.
- `.reconstruct_weights() -> np.ndarray`: Extract effective stored weights.
- `.get_conductance() -> Tuple[np.ndarray, np.ndarray]`: Return raw $G^+$ and $G^-$ matrices.

### `TiledCrossbar`
- `TiledCrossbar(rows: int, cols: int, tile_rows: int, tile_cols: int, device: Device = None, mapping: MappingStrategy = None)`: Grid of physical crossbars for large matrices.
- `TiledCrossbar.from_matrix(W: np.ndarray, tile_shape: Tuple[int, int], ...)`: Factory method.
- `.vmm(V: np.ndarray, noise: bool = False, mode: SimulationMode = "ideal") -> np.ndarray`: Computes tiled VMM with automatic global $w_{\text{max}}$ scaling across tiles.

---

## 4. `analoglib.devices` (ReRAM & Memristive Devices)

### `Device` (Abstract Base Class)
- Abstract methods: `.g_range`, `.quantize(g)`, `.add_noise(g)`, `.to_dict()`, `from_dict(d)`.

### `ReRAM`
- `ReRAM(g_min: float = 1e-6, g_max: float = 100e-6, num_states: int = 256, read_noise_sigma: float = 0.0, device_variation_sigma: float = 0.0)`: Realistic ReRAM device model.

### `IdealDevice`
- `IdealDevice()`: Unquantized, noise-free baseline device ($G \in [0, 1]$ Siemens).

---

## 5. `analoglib.effects` (Physical Non-Idealities)

### `IRDrop`
- `IRDrop(r_wire: float = 1.0)`: Parasitic wire resistance IR drop effect along wordlines and bitlines.

### `Thermal`
- `Thermal(E_a: float = 0.1, T_ref: float = 300.0)`: Arrhenius temperature-dependent conductance scaling $G(T)$.

### `Drift`
- `Drift(nu: float = 0.05, t_0: float = 1.0)`: Power-law retention loss $G(t) = G_0 (t/t_0)^{-\nu}$.

---

## 6. `analoglib.mapping` (Weight-to-Conductance Mapping)

### `DifferentialMapping`
- `DifferentialMapping(w_max: float = None)`: Maps signed weights to differential pairs $(G^+, G^-)$.

### `OffsetMapping`
- `OffsetMapping(w_max: float = None)`: Maps signed weights to single conductance with reference zero-offset.

---

## 7. `analoglib.adc_dac` (Peripheral Converters)

### `ADC`
- `ADC(bits: int = 8, v_min: float = -500e-6, v_max: float = 500e-6)`: Uniform linear ADC quantization.

### `DAC`
- `DAC(bits: int = 8, v_min: float = 0.0, v_max: float = 1.0)`: Uniform linear DAC quantization.

---

## 8. `analoglib.simulation` (Execution Engine)

### `SimulationEngine`
- `SimulationEngine(crossbars: List[Crossbar], adc: ADC = None, dac: DAC = None)`: Multi-layer crossbar simulation pipeline.
- `.run(V: np.ndarray, mode: str = "ideal") -> np.ndarray`: Run multi-layer inference in `"ideal"`, `"device"`, or `"hardware"` mode.

---

## 9. `analoglib.analysis` (Profiler & Analytics)

### `AnalogProfiler`
- `AnalogProfiler(t_read: float = 10e-9, cell_feature_F: float = 10.0, V_supply: float = 1.0)`: Analyzes physical metrics.
- `.profile(crossbars: List[Crossbar], V_input: np.ndarray, adc: ADC = None, dac: DAC = None) -> AnalogReport`: Computes array power, read energy, ADC/DAC energy, area, latency, and TOPS/W.

### `AnalogReport`
- Data class containing performance metrics and `.summary()` / `.print()` reporting methods.

---

## 10. `analoglib.exporters` (SPICE Netlist Exporter)

### `SpiceExporter`
- `SpiceExporter(dialect: str = "ngspice", R_load: float = 1e3, V_dd: float = 1.0)`: Converts crossbar models to SPICE netlists.
- `.export(path: str, crossbars: List[Crossbar], V_inputs: np.ndarray = None) -> Path`: Writes `.cir` file.
- `.export_str(crossbars: List[Crossbar]) -> str`: Returns netlist string.

---

## 11. `analoglib.neural` (Model Converters)

- `from_numpy(weights: List[np.ndarray], activations: List[str] = None) -> AIRGraph`
- `from_torch(module: nn.Module) -> AIRGraph`

---

## 12. `analoglib.visualization` (Plotting Utilities)

- `plot_conductance_matrix(crossbar, save_path: str = None)`
- `plot_weight_error_histogram(W_orig, W_recon, save_path: str = None)`
- `plot_noise_sweep(sigmas, snr_db, save_path: str = None)`
- `plot_adc_precision_sweep(bits_list, sqnr_db, save_path: str = None)`

---

## 13. `analoglib.cli` (Command Line Interface)

- `analog info <file.analog>`
- `analog simulate <file.analog> [--mode ideal|device|hardware]`
- `analog profile <file.analog>`
- `analog export-spice <file.analog> [--out circuit.cir]`
