# AnalogLib Roadmap & Release History

This document outlines the current version status and future development roadmap for AnalogLib.

---

## 1. Version 0.1.0 (Current Release — Stable)

### Implemented Features
- **Core Architecture**: Differential conductance mapping ($G^+, G^-$), offset reference mapping ($G_{\text{mid}}$), continuous ideal memristor device (`IdealDevice`), multi-state ReRAM device model (`ReRAM`).
- **Tiling & Scaling**: Single `Crossbar` array engine and automatic 2D matrix tiling (`TiledCrossbar`).
- **Peripherals & Modes**: Linear `ADC` / `DAC` models and multi-mode simulation (`"ideal"`, `"device"`, `"hardware"`).
- **Physical Non-Idealities**: Parasitic wire IR drop (`IRDrop`), Arrhenius temperature scaling (`Thermal`), and power-law retention drift (`Drift`).
- **Framework Converters**: NumPy arrays (`al.neural.from_numpy`) and PyTorch modules (`al.neural.from_torch`) with `Conv2d` im2col lowering.
- **Circuit Export & Serialization**: SPICE netlist export (`SpiceExporter`) for ngspice/LTspice and encrypted `.analog` binary format (AES-256-GCM + MsgPack + zlib).
- **Analytics & CLI**: Hardware profiler (`AnalogProfiler`), report generator (`AnalogReport`), and command line interface (`analog`).

---

## 2. Version 0.2.0 (Planned Next Release)

- **TensorFlow / Keras Converter**: Direct `from_keras(model)` converter.
- **Advanced Non-Idealities**: MNA (Modified Nodal Analysis) full IR-drop solver and 1T1R cell transistor select line dynamics.
- **Enhanced SPICE Export**: Subcircuit model support for non-linear memristor I-V curves (J4 / VTeam SPICE models).

---

## 3. Version 0.3.0 (Future Roadmap)

- **ONNX Model Import**: Direct ONNX graph parser (`from_onnx(path)`).
- **Hardware-In-The-Loop (HIL)**: Microcontroller / FPGA crossbar testbench interface via serial/gRPC.
- **Noise-Aware Training**: PyTorch custom autograd function for noise-in-the-loop training (`al.nn.AnalogLinear`).
