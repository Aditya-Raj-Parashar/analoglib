"""IR Drop effect — parasitic wire resistance reduces effective cell voltage.

Physical model
--------------
Wordlines (rows) and bitlines (columns) have finite resistance R_wire
per cell-pitch.  As current flows through the array, voltage drops
accumulate along the wires:

    V_ij = V_i - I_ij * sum_of_parasitic_drops

This reduces the effective voltage seen by each cell from the ideal
applied voltage.  We model this with a linear approximation:

    V_eff[i, j] ≈ V_i  × (1 - alpha × j)    (column position j adds drop)
    + column-sum current feedback on the row side

The first-order approximation is:
    V_eff[i, j] = V_i × (1 - r_wire × sum_{j'<j} G[i, j'] × (n_cols - j))

For simplicity we implement a mean-field approximation that captures
the spatial gradient effects without full MNA matrix solving.

Reference: Hu et al., "Memristor-based analog computation and neural network
classification with a dot product engine", Advanced Materials 2018.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .base import Effect, EffectContext


class IRDrop(Effect):
    """Parasitic IR drop along crossbar word/bitlines.

    Parameters
    ----------
    r_wire : float
        Parasitic wire resistance per cell in Ohms.
        Typical values: 0.5–10 Ω per cell segment.
    """

    def __init__(self, r_wire: float = 1.0) -> None:
        if r_wire < 0:
            raise ValueError(f"r_wire must be >= 0, got {r_wire}")
        self.r_wire = r_wire

    def apply(self, g: np.ndarray, context: EffectContext) -> np.ndarray:
        """Apply linear IR drop approximation to conductance matrix.

        The effective conductance is reduced by a spatial factor that
        depends on the cumulative column position and row current.

        Parameters
        ----------
        g : ndarray  (rows×cols)
            Nominal conductance matrix.
        context : EffectContext
            Contains row voltages V_row.

        Returns
        -------
        ndarray
            Effective conductance after IR drop.
        """
        if self.r_wire == 0.0:
            return g.copy()

        rows, cols = g.shape
        V = context.V_row  # shape (rows,)

        # Column-position IR drop factor:
        # Each column j farther from the driver sees additional voltage drop
        # from the accumulated current path through r_wire per preceding column.
        # Mean-field: sum of G along each row ≈ average load current source
        G_row_sum = g.sum(axis=1, keepdims=True)  # (rows, 1)
        V_col     = np.arange(cols) / max(cols - 1, 1)  # normalized 0..1

        # Row voltage drop: delta_V[i] = V[i] * r_wire * G_row_sum[i] * col_fraction
        # Shape broadcasting: (rows,1) * (1,cols) → (rows, cols)
        drop_factor = 1.0 - self.r_wire * (V[:, np.newaxis] * G_row_sum * V_col)
        drop_factor = np.clip(drop_factor, 0.0, 1.0)

        return g * drop_factor

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "IRDrop", "r_wire": self.r_wire}

    @classmethod
    def _from_dict_impl(cls, d: Dict[str, Any]) -> "IRDrop":
        return cls(r_wire=d["r_wire"])

    def __repr__(self) -> str:
        return f"IRDrop(r_wire={self.r_wire})"
