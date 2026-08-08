"""Abstract base for weight ↔ conductance mapping strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

from ..core.registry import RegistryMixin
from ..devices.base import Device


class MappingStrategy(RegistryMixin, ABC, registry_name="mappings"):
    """Base class for weight-to-conductance mapping.

    A mapping translates neural-network weights (which can be positive
    or negative floats) into physical conductance values constrained to
    ``[g_min, g_max]``.  The reverse mapping reconstructs weights from
    conductances.

    Subclasses define:
    * ``weights_to_conductance`` — forward mapping
    * ``conductance_to_weights`` — inverse
    """

    @abstractmethod
    def weights_to_conductance(
        self,
        W: np.ndarray,
        device: Device,
    ) -> Tuple[np.ndarray, ...]:
        """Map weight matrix to one or more conductance matrices.

        Parameters
        ----------
        W : ndarray
            Weight matrix (may contain negative values).
        device : Device
            Target device (defines ``g_min``, ``g_max``, ``num_states``).

        Returns
        -------
        tuple of ndarray
            Conductance matrices.  For differential mapping this is
            ``(G_pos, G_neg)``; for single-device it is ``(G,)``.
        """

    @abstractmethod
    def conductance_to_weights(
        self,
        *G: np.ndarray,
        device: Device,
    ) -> np.ndarray:
        """Reconstruct weights from conductance matrices.

        Parameters
        ----------
        *G : ndarray
            Conductance matrices (same tuple structure as returned by
            ``weights_to_conductance``).
        device : Device
            Device that was used for the forward mapping.

        Returns
        -------
        ndarray
            Reconstructed weight matrix.
        """

    def to_dict(self) -> Dict[str, Any]:
        """Serialize mapping configuration."""
        return {"type": self.__class__.__name__}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MappingStrategy":
        """Reconstruct mapping from dict."""
        mapping_cls = cls.get(d["type"])
        kwargs = {k: v for k, v in d.items() if k != "type"}
        return mapping_cls(**kwargs)
