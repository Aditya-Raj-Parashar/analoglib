# The `.analog` Binary File Format Specification

The `.analog` file format is an encrypted, compressed binary format designed for storing physical crossbar models, device configurations, and mapped conductance matrices.

---

## 1. File Binary Layout

```text
┌──────────────────────────────────────────┐
│ Magic bytes   : 4 bytes  (0xAE4C4942)    │
│ Format version: 4 bytes  (uint32 LE = 1) │
│ Header length : 4 bytes  (uint32 LE)     │
├──────────────────────────────────────────┤
│ Encrypted payload (AES-256-GCM)          │
│   - 16 bytes nonce                       │
│   - 16 bytes tag                         │
│   - ciphertext (zlib-compressed msgpack) │
└──────────────────────────────────────────┘
```

Header Magic: `0xAE4C4942` (`b"\xAE\x4C\x49\x42"` $\rightarrow$ ASCII `"ALIB"`).

---

## 2. Security & Encryption Details

- **Key Derivation**: 32-byte AES key derived using `scrypt(internal_secret, salt="analog-format-v1", key_len=32, N=16384, r=8, p=1)`.
- **Cipher**: AES-256-GCM authenticated encryption.
- **Compression**: zlib level 6 compression.
- **Payload**: MessagePack dictionary containing `meta`, `config`, `tensors`, and `tensor_info`.

---

## 3. Python API Usage

```python
import analoglib as al
import numpy as np

xbar = al.Crossbar(32, 16, device=al.ReRAM(num_states=256))
xbar.load_weights(np.random.randn(32, 16))

# Save crossbar model
file_path = al.save("my_model.analog", [xbar], model_name="ResNetBlock", extra_meta={"author": "Aditya"})

# Load crossbar model
loaded_data = al.load("my_model.analog")
crossbars = loaded_data["crossbars"]
metadata = loaded_data["meta"]

print("Model Name:", metadata["model_name"])
print("Loaded Crossbar Shape:", crossbars[0].rows, "x", crossbars[0].cols)
```
