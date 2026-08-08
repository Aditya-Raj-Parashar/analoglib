# Hardware & Model Compatibility Matrix (AnalogLib v0.1.0)

This matrix presents the **exact capability status of AnalogLib v0.1.0**, derived strictly from the implemented source code and unit tests.

> [!IMPORTANT]
> AnalogLib documentation clearly demarcates currently implemented features from future roadmap items. Do not assume unlisted features are available.

---

## 1. Machine Learning Framework Compatibility

| Framework | Status | Public API | Notes / Scope |
| :--- | :--- | :--- | :--- |
| **NumPy Weight Arrays** | ✅ **Supported** | `al.neural.from_numpy(weights)` <br> `al.AnalogModel.from_numpy(weights)` | Converts list of 2D weight arrays and optional activation names to `AIRGraph`. |
| **PyTorch (`nn.Module`)** | ✅ **Supported** | `al.neural.from_torch(module)` <br> `al.AnalogModel.from_torch(module)` | Supports `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax`, `nn.Sequential`, and `nn.Conv2d` (via im2col lowering). Requires `torch`. |
| **TensorFlow / Keras** | 🚧 **Planned** | *Not Implemented* | Currently listed in optional dependencies target (`[tensorflow]`), but converter is planned for v0.2.0. Extract weights to NumPy list as a workaround. |
| **ONNX Models** | 🚧 **Planned** | *Not Implemented* | Planned ONNX graph importer. Export ONNX weights to PyTorch or NumPy as a workaround. |

---

## 2. NVM Device Models

| Device Model | Status | Class | Implemented Non-Idealities & Parameters |
| :--- | :--- | :--- | :--- |
| **Ideal Continuous Device** | ✅ **Supported** | `al.IdealDevice` | Continuous conductance range $[g_{\text{min}}, g_{\text{max}}]$, zero noise, infinite state precision. |
| **ReRAM Device Model** | ✅ **Supported** | `al.ReRAM` | • Discrete conductance states ($N_{\text{states}}$)<br>• Gaussian read noise (`read_noise_sigma`) <br>• Programming error (`programming_error_sigma`)<br>• Device-to-device variation (`d2d_variation_sigma`)<br>• Stuck-at-0 / Stuck-at-1 faults (`stuck_at_fault_rate`) |

---

## 3. Weight-to-Conductance Mapping Strategies

| Strategy | Status | Class | Equations / Conductance Pair |
| :--- | :--- | :--- | :--- |
| **Differential Pair** | ✅ **Supported** | `al.DifferentialMapping` | $G^+ = G_{\text{mid}} + \frac{1}{2}\alpha W$, $G^- = G_{\text{mid}} - \frac{1}{2}\alpha W$<br>Net differential: $\Delta G = G^+ - G^- = \alpha W$ |
| **Offset Reference** | ✅ **Supported** | `al.OffsetMapping` | Single conductance $G = G_{\text{min}} + \alpha (W + w_{\text{max}})$ with static reference zero-offset. |

---

## 4. Crossbar & Tiling Engines

| Feature | Status | Class / Method | Description |
| :--- | :--- | :--- | :--- |
| **Single Crossbar Engine** | ✅ **Supported** | `al.Crossbar(rows, cols)` | Executes vector-matrix multiplication $I = V \cdot (G^+ - G^-)$. |
| **2D Tiled Crossbar Grid** | ✅ **Supported** | `al.TiledCrossbar.from_matrix(W, tile_shape)` | Automatically partitions large matrices across a 2D grid of physical tiles with shared global $w_{\text{max}}$ scale. |

---

## 5. Physical Non-Ideality Effects (`analoglib.effects`)

| Physical Effect | Status | Class | Model Equation / Parameter |
| :--- | :--- | :--- | :--- |
| **Parasitic Wire IR Drop** | ✅ **Supported** | `al.effects.IRDrop(r_wire)` | Spatial voltage drop model along wordlines/bitlines (`r_wire` in Ohms/cell). |
| **Arrhenius Thermal Scaling** | ✅ **Supported** | `al.effects.Thermal(E_a, T_ref)` | $G(T) = G_0 \exp\left(-\frac{E_a}{k_B} \left(\frac{1}{T} - \frac{1}{T_{\text{ref}}}\right)\right)$. |
| **Retention Loss Drift** | ✅ **Supported** | `al.effects.Drift(nu, t_0)` | Power-law conductance relaxation $G(t) = G_0 \left(\frac{\max(t, t_0)}{t_0}\right)^{-\nu}$. |

---

## 6. Peripheral Converters & Hardware Simulation

| Component | Status | Class | Functionality |
| :--- | :--- | :--- | :--- |
| **Digital-to-Analog (DAC)** | ✅ **Supported** | `al.DAC(bits, v_min, v_max)` | Quantizes digital inputs into discrete voltage steps. |
| **Analog-to-Digital (ADC)** | ✅ **Supported** | `al.ADC(bits, v_min, v_max)` | Hard-clips output currents and quantizes to discrete ADC output levels. |
| **Multi-Mode Engine** | ✅ **Supported** | `al.SimulationEngine` | Modes: `"ideal"` (math), `"device"` (quantized NVM + noise), `"hardware"` (DAC $\rightarrow$ Crossbar $\rightarrow$ ADC). |

---

## 7. Serialization, Exporters & Analytics

| Feature | Status | API | Details |
| :--- | :--- | :--- | :--- |
| **Encrypted `.analog` Format** | ✅ **Supported** | `al.save()` / `al.load()` | AES-256-GCM + MsgPack + zlib compressed binary format with magic header `0xAE4C4942`. |
| **SPICE Netlist Exporter** | ✅ **Supported** | `al.exporters.SpiceExporter` | Exports crossbars to ngspice and LTspice `.cir` netlists with subcircuits and resistor arrays. |
| **Analytics Profiler** | ✅ **Supported** | `al.analysis.AnalogProfiler` | Computes array power ($P = \sum V_i^2 G_{i,j}$), read energy, ADC/DAC energy, area, latency, TOPS/W. |
| **CLI Utility** | ✅ **Supported** | `analog <cmd>` | `analog info`, `analog simulate`, `analog profile`, `analog export-spice`. |
