"""TiledCrossbar — large matrix-vector multiply via automatic weight tiling.

When a weight matrix is larger than a physical crossbar tile, it must be
partitioned across multiple tiles.  TiledCrossbar handles this automatically.

Architecture
------------
Given weight W (M×N) and tile_shape (T_r × T_c):

    Tiles grid:  ceil(M/T_r) rows  ×  ceil(N/T_c) cols

                ┌────┬────┬───┐
                │ T₀₀│ T₀₁│...│   ← column tiles (split output dim)
                ├────┼────┼───┤
                │ T₁₀│ T₁₁│...│   ← row tiles (split input dim)
                └────┴────┴───┘

VMM computation:
  For each column tile group (same output cols):
      sum partial currents from all row tiles
  Concatenate column tile groups → final output

Invariant
---------
In ideal mode (IdealDevice, no noise):

    TiledCrossbar(W, tile_shape).vmm(x) == Crossbar(W).vmm(x)

This is the primary correctness test.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import numpy as np

from ..core.backend import to_numpy
from ..core.types import SimulationMode
from ..devices.base import Device
from ..devices.ideal import IdealDevice
from ..mapping.base import MappingStrategy
from ..mapping.differential import DifferentialMapping
from .crossbar import Crossbar


class TiledCrossbar:
    """Tiled analog crossbar supporting arbitrary weight matrix sizes.

    Parameters
    ----------
    rows : int
        Full matrix row count (input dimension).
    cols : int
        Full matrix column count (output dimension).
    tile_rows : int
        Maximum rows per physical tile.
    tile_cols : int
        Maximum columns per physical tile.
    device : Device
        Device model shared across all tiles.
    mapping : MappingStrategy
        Weight-to-conductance mapping shared across all tiles.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        tile_rows: int,
        tile_cols: int,
        device: Device | None = None,
        mapping: MappingStrategy | None = None,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Matrix dimensions must be positive, got ({rows}, {cols})")
        if tile_rows <= 0 or tile_cols <= 0:
            raise ValueError(f"Tile dimensions must be positive, got ({tile_rows}, {tile_cols})")

        self.rows      = rows
        self.cols      = cols
        self.tile_rows = tile_rows
        self.tile_cols = tile_cols
        self.device    = device or IdealDevice()
        self.mapping   = mapping or DifferentialMapping()

        # Compute grid dimensions
        self.n_tile_rows = math.ceil(rows / tile_rows)
        self.n_tile_cols = math.ceil(cols / tile_cols)

        # 2-D grid of Crossbar tiles: _tiles[r_tile][c_tile]
        self._tiles: list[list[Optional[Crossbar]]] = [
            [None] * self.n_tile_cols for _ in range(self.n_tile_rows)
        ]
        self._weights: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(
        cls,
        W: Any,
        tile_shape: Tuple[int, int],
        device: Device | None = None,
        mapping: MappingStrategy | None = None,
        quantize: bool = True,
    ) -> "TiledCrossbar":
        """Build and load a TiledCrossbar from a weight matrix.

        Parameters
        ----------
        W : array-like
            2-D weight matrix (rows, cols).
        tile_shape : (tile_rows, tile_cols)
            Size of each physical crossbar tile.
        device : Device, optional
            Device model (default: IdealDevice).
        mapping : MappingStrategy, optional
            Mapping strategy (default: DifferentialMapping).
        quantize : bool
            Quantize conductances to device levels.

        Returns
        -------
        TiledCrossbar
        """
        W = to_numpy(W)
        if W.ndim != 2:
            raise ValueError(f"Weight matrix must be 2-D, got shape {W.shape}")
        tile_r, tile_c = tile_shape
        tc = cls(W.shape[0], W.shape[1], tile_r, tile_c, device=device, mapping=mapping)
        tc.load_weights(W, quantize=quantize)
        return tc

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, W: Any, quantize: bool = True) -> None:
        """Partition W into tiles and load each Crossbar.

        Uses global w_max (from full matrix) so all tiles share the same
        conductance scale — this guarantees the key invariant:
            TiledCrossbar.vmm(x) == Crossbar(W).vmm(x)  (in ideal mode)

        Parameters
        ----------
        W : array-like
            Shape (rows, cols).
        quantize : bool
            Quantize conductances.
        """
        W = to_numpy(W)
        if W.shape != (self.rows, self.cols):
            raise ValueError(
                f"Weight shape {W.shape} doesn't match TiledCrossbar ({self.rows}, {self.cols})"
            )
        self._weights = W.copy()

        # Compute global w_max from the full matrix (ensures identical scale across tiles)
        global_w_max = float(np.max(np.abs(W)))
        if global_w_max == 0.0:
            global_w_max = 1.0

        for ri in range(self.n_tile_rows):
            r_start = ri * self.tile_rows
            r_end   = min(r_start + self.tile_rows, self.rows)

            for ci in range(self.n_tile_cols):
                c_start = ci * self.tile_cols
                c_end   = min(c_start + self.tile_cols, self.cols)

                W_tile = W[r_start:r_end, c_start:c_end]
                actual_rows = r_end - r_start
                actual_cols = c_end - c_start

                # Build a tile-specific mapping with fixed global w_max
                from ..mapping.differential import DifferentialMapping
                from ..mapping.offset import OffsetMapping
                if isinstance(self.mapping, DifferentialMapping):
                    tile_mapping = DifferentialMapping(w_max=global_w_max)
                elif isinstance(self.mapping, OffsetMapping):
                    tile_mapping = OffsetMapping(w_max=global_w_max)
                else:
                    tile_mapping = self.mapping  # custom mapping: user must handle scale

                xbar = Crossbar(
                    rows=actual_rows,
                    cols=actual_cols,
                    device=self.device,
                    mapping=tile_mapping,
                    differential=True,
                )
                xbar.load_weights(W_tile, quantize=quantize)
                self._tiles[ri][ci] = xbar


    # ------------------------------------------------------------------
    # VMM
    # ------------------------------------------------------------------

    def vmm(
        self,
        V: Any,
        noise: bool = False,
        mode: SimulationMode = SimulationMode.IDEAL,
    ) -> np.ndarray:
        """Tiled vector-matrix multiply.

        Splits the input vector into row-tile segments, computes each tile's
        partial output, accumulates within each column group, and concatenates.

        Parameters
        ----------
        V : array-like
            Input vector (rows,) or batch (batch, rows).
        noise : bool
            Apply device read noise.
        mode : SimulationMode
            Simulation fidelity.

        Returns
        -------
        ndarray
            Output (cols,) or (batch, cols).
        """
        self._check_loaded()
        V = to_numpy(V)

        squeezed = False
        if V.ndim == 1:
            V = V[np.newaxis, :]  # (1, rows)
            squeezed = True

        batch = V.shape[0]

        # Build output by accumulating column-tile groups
        # I_out shape: (batch, cols)
        I_out = np.zeros((batch, self.cols), dtype=np.float64)

        for ri in range(self.n_tile_rows):
            r_start = ri * self.tile_rows
            r_end   = min(r_start + self.tile_rows, self.rows)
            V_slice = V[:, r_start:r_end]   # (batch, tile_rows)

            for ci in range(self.n_tile_cols):
                c_start = ci * self.tile_cols
                c_end   = min(c_start + self.tile_cols, self.cols)

                xbar = self._tiles[ri][ci]
                I_tile = xbar.vmm(V_slice, noise=noise, mode=mode)  # (batch, tile_cols)
                I_out[:, c_start:c_end] += I_tile

        if squeezed:
            I_out = I_out.squeeze(0)
        return I_out

    # ------------------------------------------------------------------
    # Reconstruction & accessors
    # ------------------------------------------------------------------

    def reconstruct_weights(self) -> np.ndarray:
        """Reconstruct full weight matrix from all tile conductances."""
        self._check_loaded()
        W_recon = np.zeros((self.rows, self.cols), dtype=np.float64)
        for ri in range(self.n_tile_rows):
            r_start = ri * self.tile_rows
            r_end   = min(r_start + self.tile_rows, self.rows)
            for ci in range(self.n_tile_cols):
                c_start = ci * self.tile_cols
                c_end   = min(c_start + self.tile_cols, self.cols)
                W_recon[r_start:r_end, c_start:c_end] = self._tiles[ri][ci].reconstruct_weights()
        return W_recon

    @property
    def n_tiles(self) -> int:
        """Total number of physical tiles."""
        return self.n_tile_rows * self.n_tile_cols

    def tile_grid_info(self) -> dict:
        """Return tile grid summary information."""
        return {
            "matrix_shape": (self.rows, self.cols),
            "tile_shape": (self.tile_rows, self.tile_cols),
            "n_tile_rows": self.n_tile_rows,
            "n_tile_cols": self.n_tile_cols,
            "n_tiles": self.n_tiles,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_loaded(self) -> None:
        if self._weights is None:
            raise RuntimeError("TiledCrossbar has no weights. Call load_weights() first.")

    def __repr__(self) -> str:
        status = "loaded" if self._weights is not None else "empty"
        return (
            f"TiledCrossbar({self.rows}x{self.cols}, "
            f"tile={self.tile_rows}x{self.tile_cols}, "
            f"grid={self.n_tile_rows}x{self.n_tile_cols}, "
            f"{self.device.__class__.__name__}, {status})"
        )
