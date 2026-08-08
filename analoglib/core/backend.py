"""Backend-agnostic array utilities.

The core library operates on NumPy arrays.  This module provides helper
functions for casting inputs from various sources (lists, PyTorch tensors,
TensorFlow tensors) into NumPy arrays, and for common array operations used
throughout the library.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import CFG


# ---------------------------------------------------------------------------
# Array conversion
# ---------------------------------------------------------------------------

def to_numpy(x: Any) -> np.ndarray:
    """Convert *x* to a NumPy ndarray.

    Accepted inputs
    ---------------
    * ``numpy.ndarray`` — returned as-is (no copy unless dtype differs).
    * ``list`` / ``tuple`` — converted via ``np.asarray``.
    * PyTorch ``Tensor`` — detached, moved to CPU, converted.
    * TensorFlow ``Tensor`` / ``EagerTensor`` — converted via ``.numpy()``.

    Raises
    ------
    TypeError
        If *x* cannot be converted.
    """
    if isinstance(x, np.ndarray):
        return np.asarray(x, dtype=CFG.default_dtype)

    # PyTorch tensor
    if _is_torch_tensor(x):
        return x.detach().cpu().numpy().astype(CFG.default_dtype)

    # TensorFlow tensor
    if _is_tf_tensor(x):
        return x.numpy().astype(CFG.default_dtype)

    # Plain Python sequences
    if isinstance(x, (list, tuple)):
        return np.asarray(x, dtype=CFG.default_dtype)

    raise TypeError(
        f"Cannot convert {type(x).__name__} to numpy array. "
        "Supported: numpy.ndarray, list, tuple, torch.Tensor, tf.Tensor."
    )


# ---------------------------------------------------------------------------
# Array helpers
# ---------------------------------------------------------------------------

def ensure_2d(x: np.ndarray) -> np.ndarray:
    """Ensure *x* is at least 2-D (promotes 1-D to column vector)."""
    x = np.atleast_1d(x)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Element-wise clamp to ``[lo, hi]``."""
    return np.clip(x, lo, hi)


# ---------------------------------------------------------------------------
# Private helpers — lazy framework detection
# ---------------------------------------------------------------------------

def _is_torch_tensor(x: Any) -> bool:
    try:
        import torch  # noqa: F811
        return isinstance(x, torch.Tensor)
    except ImportError:
        return False


def _is_tf_tensor(x: Any) -> bool:
    try:
        import tensorflow as tf  # noqa: F811
        return isinstance(x, tf.Tensor)
    except ImportError:
        return False
