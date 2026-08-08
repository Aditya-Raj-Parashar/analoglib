# AnalogLib — API Reference

Complete reference for every public class, function, and parameter in `analoglib v0.1.0`.

---

## Table of Contents

- [Top-Level Functions](#top-level-functions)
- [Devices](#devices)
  - [Device (base)](#device-base-class)
  - [IdealDevice](#idealdevice)
  - [ReRAM](#reram)
- [Mapping Strategies](#mapping-strategies)
  - [MappingStrategy (base)](#mappingstrategy-base-class)
  - [DifferentialMapping](#differentialmapping)
  - [OffsetMapping](#offsetmapping)
- [Crossbar](#crossbar)
- [ADC / DAC](#adc--dac)
- [SimulationEngine](#simulationengine)
- [Serialization](#serialization)
- [Configuration](#configuration)
- [Enumerations](#enumerations)

---

## Top-Level Functions

These are available directly as `al.<function>` after `import analoglib as al`.

### `al.save(path, crossbars, *, model_name="", description="", extra_meta=None) → Path`

Save crossbar model(s) to an encrypted `.analog` file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Output file path. `.analog` extension added automatically if missing. |
| `crossbars` | `list[Crossbar]` | One or more crossbar layers with loaded weights. |
| `model_name` | `str` | Optional model name stored in metadata. |
| `description` | `str` | Optional description. |
| `extra_meta` | `dict` | Arbitrary user metadata (must be JSON-serializable). |

**Returns**: Absolute `Path` to the saved file.

### `al.load(path) → dict`

Load an encrypted `.analog` file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | Path to `.analog` file. |

**Returns**: Dictionary with keys:
- `"meta"` — metadata dict (model_name, description, timestamps, etc.)
- `"config"` — list of crossbar configuration dicts
- `"crossbars"` — list of reconstructed `Crossbar` objects with conductances restored

**Raises**: `ValueError` if file is not a valid `.analog` file.

### `al.set_seed(seed: int) → None`

Set the global random seed for reproducibility. Affects all noise and variation operations.

### `al.get_rng() → numpy.random.Generator`

Get a NumPy random generator honoring the global seed.

### `al.to_numpy(x) → numpy.ndarray`

Convert any array-like input to NumPy. Accepts: `numpy.ndarray`, `list`, `tuple`, `torch.Tensor`, `tf.Tensor`.

---

## Devices

### Device (base class)

`analoglib.devices.Device` — Abstract base for all analog memory devices.

```python
class Device(ABC):
    g_min: float          # Minimum conductance (Siemens)
    g_max: float          # Maximum conductance (Siemens)
    num_states: int       # Discrete programmable levels (0 = continuous)
```

#### Properties

| Property | Returns | Description |
|----------|---------|-------------|
| `g_range` | `float` | `g_max - g_min` |

#### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `quantize` | `(g: ndarray) → ndarray` | Snap conductances to nearest valid device state |
| `add_noise` | `(g: ndarray) → ndarray` | Inject read noise |
| `add_variation` | `(g: ndarray) → ndarray` | Apply device-to-device variation |
| `conductance_levels` | `() → ndarray` | Array of all valid conductance values |
| `to_dict` | `() → dict` | Serialize device configuration |
| `from_dict` | `(d: dict) → Device` | Reconstruct device from dict (class method) |

#### Registry

| Method | Description |
|--------|-------------|
| `Device.registry()` | Returns `dict` of all registered device subclasses |
| `Device.get(name)` | Look up device class by name string |

---

### IdealDevice

`analoglib.devices.IdealDevice` — Perfect device, no non-idealities.

```python
IdealDevice(g_min=0.0, g_max=1.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_min` | `0.0` | Minimum conductance |
| `g_max` | `1.0` | Maximum conductance |

Behavior: `quantize()` only clamps to bounds. `add_noise()` and `add_variation()` are identity operations.

---

### ReRAM

`analoglib.devices.ReRAM` — Resistive RAM with configurable non-idealities.

```python
ReRAM(
    g_min=1e-6,
    g_max=100e-6,
    num_states=256,
    read_noise_sigma=0.0,
    programming_error_sigma=0.0,
    d2d_variation_sigma=0.0,
    stuck_at_fault_rate=0.0,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `g_min` | `1e-6` | Minimum conductance (S) |
| `g_max` | `100e-6` | Maximum conductance (S) |
| `num_states` | `256` | Discrete levels (256 = 8-bit) |
| `read_noise_sigma` | `0.0` | Gaussian read noise σ (relative to window) |
| `programming_error_sigma` | `0.0` | Write error σ (applied after quantization) |
| `d2d_variation_sigma` | `0.0` | Device-to-device variation σ |
| `stuck_at_fault_rate` | `0.0` | Fraction of stuck devices (0.0 to 1.0) |

**All sigma values** are relative — e.g., `read_noise_sigma=0.03` means σ = 3% of `(g_max - g_min)`.

---

## Mapping Strategies

### MappingStrategy (base class)

`analoglib.mapping.MappingStrategy` — Abstract base for weight ↔ conductance mapping.

| Method | Signature | Description |
|--------|-----------|-------------|
| `weights_to_conductance` | `(W, device) → tuple[ndarray, ...]` | Forward mapping |
| `conductance_to_weights` | `(*G, device=) → ndarray` | Inverse mapping |

---

### DifferentialMapping

`analoglib.mapping.DifferentialMapping` — Maps weights to (G⁺, G⁻) pairs.

```python
DifferentialMapping(w_max=None)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_max` | `None` | Max absolute weight for normalization. `None` = auto-detect from data. |

**Forward**: `weights_to_conductance(W, device) → (G_pos, G_neg)`

**Inverse**: `conductance_to_weights(G_pos, G_neg, device=) → W`

---

### OffsetMapping

`analoglib.mapping.OffsetMapping` — Single-device mapping with conductance offset.

```python
OffsetMapping(w_max=None)
```

**Forward**: `weights_to_conductance(W, device) → (G,)` (single matrix)

**Inverse**: `conductance_to_weights(G, device=) → W`

---

## Crossbar

`analoglib.crossbar.Crossbar` — Resistive crossbar array for analog VMM.

```python
Crossbar(
    rows,
    cols,
    device=None,
    mapping=None,
    differential=True,
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rows` | — | Number of row lines (input dimension) |
| `cols` | — | Number of column lines (output dimension) |
| `device` | `IdealDevice()` | Analog memory device model |
| `mapping` | `DifferentialMapping()` | Weight-to-conductance mapping strategy |
| `differential` | `True` | Use G⁺/G⁻ differential representation |

### Methods

#### `load_weights(W, quantize=True, apply_variation=False)`

Convert and store weight matrix `W` (shape `(rows, cols)`) as conductances.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `W` | — | Weight matrix (numpy, torch, tf, or list) |
| `quantize` | `True` | Quantize conductances to device levels |
| `apply_variation` | `False` | Apply D2D variation after programming |

#### `vmm(V, noise=False, mode=SimulationMode.IDEAL) → ndarray`

Vector-matrix multiply. Accepts 1-D `(rows,)` or batch `(batch, rows)`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `V` | — | Input voltage vector |
| `noise` | `False` | Inject read noise during computation |
| `mode` | `IDEAL` | Simulation fidelity level |

**Returns**: Output current `(cols,)` or `(batch, cols)`.

#### `get_conductance() → tuple[ndarray, ...]`

Returns `(G_pos, G_neg)` for differential or `(G,)` for single-device.

#### `reconstruct_weights() → ndarray`

Inverse-map conductances back to weights (includes quantization error).

---

## ADC / DAC

### ADC

`analoglib.adc_dac.ADC` — Analog-to-digital converter.

```python
ADC(bits=8, v_min=0.0, v_max=1.0)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bits` | `8` | Resolution (n bits → 2ⁿ levels) |
| `v_min` | `0.0` | Minimum representable value |
| `v_max` | `1.0` | Maximum representable value |

#### `convert(x: ndarray) → ndarray`

Quantize continuous values. Clips to `[v_min, v_max]`, snaps to nearest level.

#### Properties

- `resolution` → voltage step per LSB
- `num_levels` → `2^bits`

### DAC

`analoglib.adc_dac.DAC` — Digital-to-analog converter. Same interface as ADC.

```python
DAC(bits=8, v_min=0.0, v_max=1.0)
```

---

## SimulationEngine

`analoglib.simulation.SimulationEngine` — Orchestrates multi-layer analog inference.

```python
SimulationEngine(crossbars=None, adc=None, dac=None)
```

### Methods

#### `run(x, mode="ideal") → ndarray`

Forward pass through all crossbars.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x` | — | Input vector/batch |
| `mode` | `"ideal"` | `"ideal"`, `"device"`, or `"hardware"` |

#### `run_comparison(x, modes=None) → dict[str, ndarray]`

Run in multiple modes, returns `{"ideal": ..., "device": ..., "hardware": ...}`.

#### `add_crossbar(xbar)`

Append a crossbar to the layer stack.

---

## Serialization

### `al.save(path, crossbars, **kwargs) → Path`

See [Top-Level Functions](#top-level-functions).

### `al.load(path) → dict`

See [Top-Level Functions](#top-level-functions).

### File format

Files use magic bytes `0xAE4C4942`, AES-256-GCM encryption, zlib compression. Not readable without AnalogLib.

---

## Configuration

### `al.CFG`

Global `AnalogConfig` instance:

| Field | Default | Description |
|-------|---------|-------------|
| `seed` | `None` | Random seed (None = non-deterministic) |
| `default_dtype` | `np.float64` | Default array dtype |
| `float_tolerance` | `1e-12` | Numerical comparison tolerance |

---

## Enumerations

### `al.SimulationMode`

```python
SimulationMode.IDEAL     # Mathematical only
SimulationMode.DEVICE    # + quantization, noise, variation
SimulationMode.HARDWARE  # + ADC/DAC
SimulationMode.SPICE     # Circuit-level (future)
```

### `al.NoiseType`

```python
NoiseType.NONE
NoiseType.GAUSSIAN
NoiseType.UNIFORM
NoiseType.THERMAL
NoiseType.SHOT
```
