# AnalogLib API Reference (v0.1.0)

This section provides the complete reference for all public modules, classes, functions, and parameters in **AnalogLib v0.1.0**.

---

## Top-Level Exports (`analoglib`)

Import top-level exports directly from `analoglib`:

```python
import analoglib as al
```

### Globals & Configurations
- `al.__version__`: Package version string (`"0.1.0"`).
- `al.set_seed(seed: int)`: Sets the global pseudo-random number generator seed for deterministic simulation.
- `al.get_rng()`: Returns the global NumPy `Generator` instance.

---

## Module Index

### 1. Devices (`analoglib.devices` / `al.devices`)
- `al.Device`: Abstract base class for memristive device models.
- `al.IdealDevice(g_min=1e-6, g_max=100e-6)`: Continuous unquantized memristor model.
- `al.ReRAM(g_min, g_max, num_states, read_noise_sigma, programming_error_sigma, d2d_variation_sigma, stuck_at_fault_rate)`: Multi-state ReRAM model.

### 2. Weight Mapping (`analoglib.mapping` / `al.mapping`)
- `al.MappingStrategy`: Base mapping class.
- `al.DifferentialMapping(w_max=1.0)`: Maps signed weights to differential $G^+, G^-$ conductance pairs.
- `al.OffsetMapping(w_max=1.0)`: Maps signed weights to single conductance cell $G$ with static offset reference.

### 3. Crossbars & Tiling (`analoglib.crossbar` / `al.crossbar`)
- `al.Crossbar(rows, cols, device, mapping, differential)`: Single physical crossbar array engine.
- `al.TiledCrossbar(rows, cols, tile_rows, tile_cols, device, mapping)`: 2D tile grid for large weight matrices.
- `al.TiledCrossbar.from_matrix(W, tile_shape)`: Factory constructor.

### 4. Converters & Peripherals (`analoglib.adc_dac` / `al.adc_dac`)
- `al.ADC(bits=8, v_min=0.0, v_max=1.0)`: Analog-to-Digital Converter.
- `al.DAC(bits=8, v_min=0.0, v_max=1.0)`: Digital-to-Analog Converter.

### 5. Multi-Layer Engine (`analoglib.simulation` / `al.simulation`)
- `al.SimulationMode`: Enum (`IDEAL`, `DEVICE`, `HARDWARE`).
- `al.SimulationEngine(crossbars, adc, dac, activations)`: Multi-layer runtime simulator.

### 6. AIR & Model Abstraction (`analoglib.air` / `al.air`)
- `al.AnalogModel`: High-level wrapper for building, compiling, and running analog models.
- `al.air.AIRGraph`, `al.air.AIRLayer`, `al.air.lower()`: Intermediate Representation contract.

### 7. Neural Framework Converters (`analoglib.neural` / `al.neural`)
- `al.neural.from_numpy(weights, name, activations)`: Converts NumPy arrays to `AIRGraph`.
- `al.neural.from_torch(module, name, input_shape)`: Converts PyTorch `nn.Module` to `AIRGraph`.

### 8. Non-Ideality Physical Effects (`analoglib.effects` / `al.effects`)
- `al.effects.IRDrop(r_wire)`: Parasitic wire resistance voltage drop.
- `al.effects.Thermal(E_a, T_ref)`: Arrhenius temperature-dependent scaling.
- `al.effects.Drift(nu, t_0)`: Power-law retention loss drift.

### 9. File Format & Exporters
- `al.save(path, crossbars, model_name, description, extra_meta)`: Encrypted `.analog` save.
- `al.load(path)`: Encrypted `.analog` load.
- `al.exporters.SpiceExporter(dialect, R_load, V_dd)`: SPICE netlist generator.

### 10. Analytics & Profiling (`analoglib.analysis` / `al.analysis`)
- `al.analysis.AnalogProfiler(t_read, cell_feature_F, V_supply)`: Computes power, energy, area, and TOPS/W.
- `al.analysis.AnalogReport`: Dataclass container for hardware metrics.

### 11. Command-Line Interface (`analog.cli`)
- `analog info <file.analog>`
- `analog simulate <file.analog> [--mode ideal|device|hardware]`
- `analog export-spice <file.analog> [--out file.cir] [--dialect ngspice]`
- `analog profile <file.analog>`
