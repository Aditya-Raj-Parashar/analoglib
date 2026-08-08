"""Encrypted binary .analog file format.

Inspired by GGUF's design (magic number, versioned header, self-contained
metadata, aligned tensor data), with a critical requirement: the file
contents are NOT accessible without the analoglib library.

File layout
-----------
::

    ┌──────────────────────────────────────────┐
    │ Magic bytes   : 4 bytes  (0xAE4C4942)    │
    │ Format version: 4 bytes  (uint32 LE)     │
    │ Header length : 4 bytes  (uint32 LE)     │
    ├──────────────────────────────────────────┤
    │ Encrypted payload (AES-256-GCM)          │
    │   - 16 bytes nonce                       │
    │   - 16 bytes tag                         │
    │   - ciphertext (zlib-compressed msgpack) │
    └──────────────────────────────────────────┘

The encrypted payload, once decrypted, is a zlib-compressed MessagePack
blob containing:

.. code-block:: python

    {
        "meta": { ... },           # model metadata
        "config": { ... },         # device, crossbar, mapping, ADC/DAC config
        "tensors": {
            "layer_0_g_pos": <bytes>,  # numpy array as raw bytes
            "layer_0_g_neg": <bytes>,
            ...
        },
        "tensor_info": {
            "layer_0_g_pos": {"dtype": "float64", "shape": [128, 64]},
            ...
        }
    }

The encryption key is derived from an internal library secret combined
with the format version.  This makes .analog files opaque to external
tools — only ``analoglib.load()`` can read them.
"""

from __future__ import annotations

import io
import json
import struct
import zlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import msgpack
    _HAS_MSGPACK = True
except ImportError:
    _HAS_MSGPACK = False

from Crypto.Cipher import AES
from Crypto.Protocol.KDF import scrypt

from ..version import __version__, __format_version__
from ..crossbar.crossbar import Crossbar
from ..devices.base import Device
from ..mapping.base import MappingStrategy


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAGIC = b"\xAE\x4C\x49\x42"   # "ALIB" marker
FORMAT_VERSION = 1
HEADER_SIZE = 12               # magic(4) + version(4) + header_len(4)

# Internal secret — combined with format version to derive encryption key.
# This is NOT true security (key is embedded in source), but fulfils the
# requirement that .analog files are opaque without the library.
_INTERNAL_SECRET = b"analoglib-v1-proprietary-format-key-2026"


# ---------------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------------

def _derive_key(version: int = FORMAT_VERSION) -> bytes:
    """Derive a 32-byte AES key from the internal secret + version."""
    salt = f"analog-format-v{version}".encode()
    return scrypt(
        _INTERNAL_SECRET, salt, key_len=32, N=2**14, r=8, p=1
    )


# ---------------------------------------------------------------------------
# Encrypt / Decrypt
# ---------------------------------------------------------------------------

def _encrypt(data: bytes, key: bytes) -> bytes:
    """AES-256-GCM encrypt. Returns: nonce(16) + tag(16) + ciphertext."""
    cipher = AES.new(key, AES.MODE_GCM, nonce=None)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return cipher.nonce + tag + ciphertext


def _decrypt(blob: bytes, key: bytes) -> bytes:
    """AES-256-GCM decrypt from nonce(16) + tag(16) + ciphertext."""
    nonce = blob[:16]
    tag = blob[16:32]
    ciphertext = blob[32:]
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


# ---------------------------------------------------------------------------
# Internal serialization (msgpack or fallback to json)
# ---------------------------------------------------------------------------

def _pack(obj: dict) -> bytes:
    """Serialize dict to bytes (msgpack preferred, JSON fallback)."""
    if _HAS_MSGPACK:
        return msgpack.packb(obj, use_bin_type=True)
    else:
        # Fallback: JSON with base64-encoded binary values
        return json.dumps(obj, default=_json_default).encode("utf-8")


def _unpack(data: bytes) -> dict:
    """Deserialize bytes to dict."""
    if _HAS_MSGPACK:
        return msgpack.unpackb(data, raw=False)
    else:
        return json.loads(data.decode("utf-8"), object_hook=_json_hook)


def _json_default(obj):
    import base64
    if isinstance(obj, bytes):
        return {"__bytes__": base64.b64encode(obj).decode("ascii")}
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _json_hook(d):
    import base64
    if "__bytes__" in d:
        return base64.b64decode(d["__bytes__"])
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save(
    path: str | Path,
    crossbars: List[Crossbar],
    *,
    model_name: str = "",
    description: str = "",
    extra_meta: Dict[str, Any] | None = None,
) -> Path:
    """Save crossbar model(s) to an encrypted .analog file.

    Parameters
    ----------
    path : str or Path
        Output file path (should have .analog extension).
    crossbars : list of Crossbar
        Crossbar layer(s) with loaded weights.
    model_name, description : str
        Optional metadata fields.
    extra_meta : dict
        Additional user-defined metadata.

    Returns
    -------
    Path
        Absolute path to the saved file.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".analog")

    # -- Build payload --
    meta = {
        "format_version": __format_version__,
        "library_version": __version__,
        "model_name": model_name,
        "description": description,
        "created": datetime.now(timezone.utc).isoformat(),
        "num_layers": len(crossbars),
    }
    if extra_meta:
        meta.update(extra_meta)

    configs = []
    tensors = {}
    tensor_info = {}

    for i, xbar in enumerate(crossbars):
        configs.append(xbar.to_dict())

        # Store conductance arrays
        g_matrices = xbar.get_conductance()
        if xbar.differential:
            _store_tensor(tensors, tensor_info, f"layer_{i}_g_pos", g_matrices[0])
            _store_tensor(tensors, tensor_info, f"layer_{i}_g_neg", g_matrices[1])
        else:
            _store_tensor(tensors, tensor_info, f"layer_{i}_g", g_matrices[0])

    payload = {
        "meta": meta,
        "config": configs,
        "tensors": tensors,
        "tensor_info": tensor_info,
    }

    # -- Serialize, compress, encrypt --
    raw = _pack(payload)
    compressed = zlib.compress(raw, level=6)
    key = _derive_key(FORMAT_VERSION)
    encrypted = _encrypt(compressed, key)

    # -- Write file --
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", FORMAT_VERSION))
        f.write(struct.pack("<I", len(encrypted)))
        f.write(encrypted)

    return path.resolve()


def load(path: str | Path) -> Dict[str, Any]:
    """Load an encrypted .analog file.

    Parameters
    ----------
    path : str or Path
        Path to `.analog` file.

    Returns
    -------
    dict
        Deserialized model with keys:
        ``meta``, ``config``, ``crossbars`` (list of reconstructed Crossbar objects).

    Raises
    ------
    ValueError
        If the file is not a valid .analog file.
    """
    path = Path(path)

    with open(path, "rb") as f:
        # -- Read & validate header --
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(
                f"Not a valid .analog file (bad magic: {magic!r}). "
                "This file cannot be opened without analoglib."
            )

        version = struct.unpack("<I", f.read(4))[0]
        payload_len = struct.unpack("<I", f.read(4))[0]
        encrypted = f.read(payload_len)

    # -- Decrypt, decompress, deserialize --
    key = _derive_key(version)
    try:
        compressed = _decrypt(encrypted, key)
    except Exception as e:
        raise ValueError(
            f"Failed to decrypt .analog file (version {version}). "
            f"Possible version mismatch or corrupt file."
        ) from e

    raw = zlib.decompress(compressed)
    payload = _unpack(raw)

    # -- Reconstruct crossbars --
    crossbars = []
    configs = payload["config"]
    tensors = payload["tensors"]
    tensor_info = payload["tensor_info"]

    for i, cfg in enumerate(configs):
        device = Device.from_dict(cfg["device"])
        mapping = MappingStrategy.from_dict(cfg["mapping"])
        xbar = Crossbar(
            rows=cfg["rows"],
            cols=cfg["cols"],
            device=device,
            mapping=mapping,
            differential=cfg["differential"],
        )

        # Restore conductances directly
        if cfg["differential"]:
            g_pos = _load_tensor(tensors, tensor_info, f"layer_{i}_g_pos")
            g_neg = _load_tensor(tensors, tensor_info, f"layer_{i}_g_neg")
            xbar._g_pos = g_pos
            xbar._g_neg = g_neg
        else:
            g = _load_tensor(tensors, tensor_info, f"layer_{i}_g")
            xbar._g_pos = g
            xbar._g_neg = None

        crossbars.append(xbar)

    return {
        "meta": payload["meta"],
        "config": configs,
        "crossbars": crossbars,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_tensor(
    tensors: dict,
    tensor_info: dict,
    name: str,
    arr: np.ndarray,
) -> None:
    """Store a numpy array as raw bytes in the tensors dict."""
    tensors[name] = arr.tobytes()
    tensor_info[name] = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _load_tensor(
    tensors: dict,
    tensor_info: dict,
    name: str,
) -> np.ndarray:
    """Reconstruct a numpy array from raw bytes."""
    info = tensor_info[name]
    raw = tensors[name]
    if isinstance(raw, str):
        import base64
        raw = base64.b64decode(raw)
    return np.frombuffer(raw, dtype=np.dtype(info["dtype"])).reshape(info["shape"]).copy()
