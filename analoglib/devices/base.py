"""Abstract base class for all analog memory devices."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from ..core.config import get_rng
from ..core.registry import RegistryMixin


class Device(RegistryMixin, ABC, registry_name="devices"):
    """Base class for analog memory devices (ReRAM, PCM, etc.).

    Every device defines a conductance window ``[g_min, g_max]`` and
    a number of discrete conductance states.  Subclasses implement
    quantization, noise injection, and device-to-device variation.

    Parameters
    ----------
    g_min : float
        Minimum conductance in Siemens.
    g_max : float
        Maximum conductance in Siemens.
    num_states : int
        Number of discrete programmable conductance levels.
        Use ``0`` or a very large number for continuous (ideal) devices.
    """

    def __init__(
        self,
        g_min: float,
        g_max: float,
        num_states: int = 0,
    ) -> None:
        if g_min < 0:
            raise ValueError(f"g_min must be non-negative, got {g_min}")
        if g_max <= g_min:
            raise ValueError(
                f"g_max ({g_max}) must be greater than g_min ({g_min})"
            )
        if num_states < 0:
            raise ValueError(f"num_states must be non-negative, got {num_states}")

        self.g_min = g_min
        self.g_max = g_max
        self.num_states = num_states

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def g_range(self) -> float:
        """Conductance window width: ``g_max - g_min``."""
        return self.g_max - self.g_min

    def conductance_levels(self) -> np.ndarray:
        """Return array of valid discrete conductance values.

        If ``num_states == 0`` (continuous), returns an empty array.
        """
        if self.num_states == 0:
            return np.array([], dtype=np.float64)
        return np.linspace(self.g_min, self.g_max, self.num_states)

    # ------------------------------------------------------------------
    # Abstract interface — subclasses MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def quantize(self, g: np.ndarray) -> np.ndarray:
        """Quantize conductances to the nearest valid device state.

        Parameters
        ----------
        g : ndarray
            Conductance values (Siemens).

        Returns
        -------
        ndarray
            Quantized conductances within ``[g_min, g_max]``.
        """

    @abstractmethod
    def add_noise(self, g: np.ndarray) -> np.ndarray:
        """Inject read noise onto conductance values.

        Parameters
        ----------
        g : ndarray
            Conductance values.

        Returns
        -------
        ndarray
            Noisy conductances (still clamped to ``[g_min, g_max]``).
        """

    @abstractmethod
    def add_variation(self, g: np.ndarray) -> np.ndarray:
        """Apply device-to-device (D2D) variation.

        Parameters
        ----------
        g : ndarray
            Conductance values.

        Returns
        -------
        ndarray
            Conductances with variation applied.
        """

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize device configuration to a plain dict."""
        return {
            "type": self.__class__.__name__,
            "g_min": self.g_min,
            "g_max": self.g_max,
            "num_states": self.num_states,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Device":
        """Reconstruct a device from a dict produced by :meth:`to_dict`.

        This uses the class registry to resolve the concrete subclass.
        """
        device_cls = cls.get(d["type"])
        # Forward all keys except 'type' to the constructor
        kwargs = {k: v for k, v in d.items() if k != "type"}
        return device_cls(**kwargs)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"g_min={self.g_min:.2e}, g_max={self.g_max:.2e}, "
            f"num_states={self.num_states})"
        )
