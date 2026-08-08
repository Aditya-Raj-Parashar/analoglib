"""Crossbar array — the central analog computation engine.

Physical model
--------------
A crossbar array of size ``(rows, cols)`` stores conductances ``G[i, j]``
at each junction.  An input voltage vector ``V`` is applied to the rows,
and the output current at each column is:

    I_j = Σ_i  G[i, j] · V[i]

In matrix form:  ``I = V @ G``  (vector-matrix multiply).

For **differential** operation, two conductance matrices (G⁺, G⁻) are
used, and the effective output is:

    I_j = Σ_i (G⁺[i,j] - G⁻[i,j]) · V[i]
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core.backend import to_numpy
from ..core.types import SimulationMode
from ..devices.base import Device
from ..devices.ideal import IdealDevice
from ..mapping.base import MappingStrategy
from ..mapping.differential import DifferentialMapping


class Crossbar:
    """Resistive crossbar array for analog vector-matrix multiply.

    Parameters
    ----------
    rows : int
        Number of rows (input dimension).
    cols : int
        Number of columns (output dimension).
    device : Device
        Analog memory device model.
    mapping : MappingStrategy or None
        Weight-to-conductance mapping.  Defaults to ``DifferentialMapping``.
    differential : bool
        If ``True`` (default), use differential (G⁺ - G⁻) representation.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        device: Device | None = None,
        mapping: MappingStrategy | None = None,
        differential: bool = True,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Crossbar dimensions must be positive, got ({rows}, {cols})")

        self.rows = rows
        self.cols = cols
        self.device = device or IdealDevice()
        self.mapping = mapping or DifferentialMapping()
        self.differential = differential

        # Conductance storage
        self._g_pos: Optional[np.ndarray] = None  # (rows, cols)
        self._g_neg: Optional[np.ndarray] = None  # (rows, cols) — only if differential
        self._w_max: Optional[float] = None        # Stored for reconstruction

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(
        self,
        W: Any,
        quantize: bool = True,
        apply_variation: bool = False,
    ) -> None:
        """Convert and store weights as conductances.

        Parameters
        ----------
        W : array-like
            Weight matrix of shape ``(rows, cols)``.
        quantize : bool
            If ``True``, quantize conductances to device levels.
        apply_variation : bool
            If ``True``, apply device-to-device variation after programming.
        """
        W = to_numpy(W)
        if W.shape != (self.rows, self.cols):
            raise ValueError(
                f"Weight shape {W.shape} doesn't match crossbar ({self.rows}, {self.cols})"
            )

        # Store w_max for reconstruction
        self._w_max = float(np.max(np.abs(W)))

        # Map weights to conductances
        conductances = self.mapping.weights_to_conductance(W, self.device)

        if self.differential:
            if len(conductances) != 2:
                raise ValueError(
                    f"Differential crossbar requires mapping that produces 2 matrices, "
                    f"got {len(conductances)} from {type(self.mapping).__name__}"
                )
            self._g_pos, self._g_neg = conductances
        else:
            if len(conductances) != 1:
                raise ValueError(
                    f"Non-differential crossbar requires mapping that produces 1 matrix, "
                    f"got {len(conductances)} from {type(self.mapping).__name__}"
                )
            self._g_pos = conductances[0]
            self._g_neg = None

        # Apply device transformations
        if quantize:
            self._g_pos = self.device.quantize(self._g_pos)
            if self._g_neg is not None:
                self._g_neg = self.device.quantize(self._g_neg)

        if apply_variation:
            self._g_pos = self.device.add_variation(self._g_pos)
            if self._g_neg is not None:
                self._g_neg = self.device.add_variation(self._g_neg)

    # ------------------------------------------------------------------
    # Vector-Matrix Multiply
    # ------------------------------------------------------------------

    def vmm(
        self,
        V: Any,
        noise: bool = False,
        mode: SimulationMode = SimulationMode.IDEAL,
    ) -> np.ndarray:
        """Perform analog vector-matrix multiply.

        Parameters
        ----------
        V : array-like
            Input voltage vector of shape ``(rows,)`` or batch ``(batch, rows)``.
        noise : bool
            If ``True``, inject read noise during computation.
        mode : SimulationMode
            Simulation fidelity level.

        Returns
        -------
        ndarray
            Output current vector of shape ``(cols,)`` or ``(batch, cols)``.
        """
        self._check_loaded()
        V = to_numpy(V)

        # Handle 1-D input
        squeezed = False
        if V.ndim == 1:
            V = V.reshape(1, -1)
            squeezed = True

        if V.shape[-1] != self.rows:
            raise ValueError(
                f"Input dimension {V.shape[-1]} doesn't match crossbar rows {self.rows}"
            )

        # Get conductance matrices (optionally with noise)
        g_pos = self._get_conductance(self._g_pos, noise, mode)
        if self.differential:
            g_neg = self._get_conductance(self._g_neg, noise, mode)
            # I = V @ (G+ - G-)
            I_out = V @ (g_pos - g_neg)
        else:
            I_out = V @ g_pos

        if squeezed:
            I_out = I_out.squeeze(0)

        return I_out

    def _get_conductance(
        self,
        g: np.ndarray,
        noise: bool,
        mode: SimulationMode,
    ) -> np.ndarray:
        """Optionally apply noise to conductance for a single read."""
        if mode == SimulationMode.IDEAL or not noise:
            return g
        # Device-aware and above: apply read noise
        return self.device.add_noise(g)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_conductance(self) -> Tuple[np.ndarray, ...]:
        """Return stored conductance matrices.

        Returns
        -------
        tuple of ndarray
            ``(G_pos, G_neg)`` for differential, ``(G,)`` otherwise.
        """
        self._check_loaded()
        if self.differential:
            return (self._g_pos.copy(), self._g_neg.copy())
        return (self._g_pos.copy(),)

    def reconstruct_weights(self) -> np.ndarray:
        """Reconstruct weights from stored conductances.

        Note: if quantization was applied, the reconstructed weights
        will differ from the originals by the quantization error.
        """
        self._check_loaded()
        if self.differential:
            return self.mapping.conductance_to_weights(
                self._g_pos, self._g_neg, device=self.device
            )
        return self.mapping.conductance_to_weights(
            self._g_pos, device=self.device
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _check_loaded(self) -> None:
        if self._g_pos is None:
            raise RuntimeError(
                "Crossbar has no weights loaded. Call load_weights() first."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize crossbar configuration (not conductance data)."""
        return {
            "rows": self.rows,
            "cols": self.cols,
            "differential": self.differential,
            "device": self.device.to_dict(),
            "mapping": self.mapping.to_dict(),
        }

    def __repr__(self) -> str:
        loaded = "loaded" if self._g_pos is not None else "empty"
        mode = "differential" if self.differential else "single"
        return (
            f"Crossbar({self.rows}×{self.cols}, {mode}, "
            f"device={self.device.__class__.__name__}, {loaded})"
        )
