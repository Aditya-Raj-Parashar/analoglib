# Installation & Setup Guide

This guide covers environment prerequisites, package installation, optional dependencies, and installation verification for **AnalogLib v0.1.0**.

---

## 1. Requirements

- **Python**: `3.10`, `3.11`, `3.12`, or `3.13`
- **Operating System**: Linux, macOS, or Windows
- **Core Dependencies**:
  - `numpy >= 1.24`
  - `pycryptodome >= 3.19` (required for AES-256-GCM `.analog` format)
  - `msgpack >= 1.0.0` (preferred binary serializer)

---

## 2. Installation Methods

### Install from PyPI

```bash
pip install analoglib
```

### Install from Source (Development)

Clone the repository and perform an editable installation:

```bash
git clone https://github.com/Aditya-Raj-Parashar/analoglib.git
cd analoglib
pip install -e .
```

---

## 3. Optional Dependencies

AnalogLib modularizes optional features to keep base installation lightweight.

| Optional Extra | Package | Installed Command | Purpose |
| :--- | :--- | :--- | :--- |
| `[torch]` | `torch >= 2.0` | `pip install "analoglib[torch]"` | PyTorch model conversion (`al.neural.from_torch`) |
| `[viz]` | `matplotlib >= 3.7` | `pip install "analoglib[viz]"` | Plotting utilities (`al.visualization`) |
| `[dev]` | `pytest`, `build`, `twine` | `pip install "analoglib[dev]"` | Running tests and building packages |
| `[all]` | All extras | `pip install "analoglib[all]"` | Full installation |

---

## 4. Verification

After installation, verify that AnalogLib imports correctly and run a 1-line check:

```python
import analoglib as al

print(f"AnalogLib Version: {al.__version__}")
```

You can also run the pytest suite if installed from source:

```bash
pytest
```
