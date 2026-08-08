"""Matplotlib-based visualization plots for AnalogLib.

All functions produce publication-quality matplotlib figures.
Matplotlib is an optional dependency — import errors are raised
with an actionable message.

Functions
---------
plot_conductance_matrix      — heatmap of G+ and G- arrays
plot_weight_error_histogram  — quantization error distribution
plot_noise_sweep             — SNR vs read noise sigma
plot_adc_precision_sweep     — SQNR vs ADC bit-width
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_conductance_matrix(
    crossbar,
    *,
    title: str = "Crossbar Conductance",
    figsize: Tuple[int, int] = (10, 4),
    cmap: str = "viridis",
    save_path: Optional[str] = None,
):
    """Plot G+ and G- conductance heatmaps side by side.

    Parameters
    ----------
    crossbar : Crossbar
        Loaded crossbar with conductances.
    title : str
        Figure title.
    figsize : tuple
        Figure width, height in inches.
    cmap : str
        Matplotlib colormap name.
    save_path : str, optional
        If given, save figure to this path instead of displaying.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    G_pos, G_neg = crossbar.get_conductance()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for ax, G, label in zip(axes, [G_pos, G_neg], ["G+ (Siemens)", "G- (Siemens)"]):
        im = ax.imshow(G.T, aspect="auto", cmap=cmap)
        ax.set_xlabel("Row index")
        ax.set_ylabel("Column index")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, format="%.2e")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_weight_error_histogram(
    W_original: np.ndarray,
    W_reconstructed: np.ndarray,
    *,
    title: str = "Weight Quantization Error",
    bins: int = 50,
    figsize: Tuple[int, int] = (8, 4),
    save_path: Optional[str] = None,
):
    """Plot histogram of absolute and relative weight quantization errors.

    Parameters
    ----------
    W_original : ndarray
        Original weight matrix.
    W_reconstructed : ndarray
        Reconstructed weight matrix (after quantization roundtrip).
    title : str
        Figure title.
    bins : int
        Number of histogram bins.
    figsize : tuple
        Figure dimensions.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    abs_err = np.abs(W_original - W_reconstructed).ravel()
    rel_err = (abs_err / (np.abs(W_original).ravel() + 1e-12)) * 100

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(title, fontsize=13, fontweight="bold")

    axes[0].hist(abs_err, bins=bins, color="#4a90d9", edgecolor="white", linewidth=0.5)
    axes[0].set_xlabel("Absolute Error")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"Absolute Error (max={abs_err.max():.4f})")
    axes[0].axvline(abs_err.mean(), color="red", linestyle="--", label=f"mean={abs_err.mean():.4f}")
    axes[0].legend()

    axes[1].hist(rel_err, bins=bins, color="#e74c3c", edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Relative Error (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Relative Error (mean={rel_err.mean():.2f}%)")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_noise_sweep(
    sigmas: List[float],
    snr_db: List[float],
    *,
    title: str = "SNR vs Read Noise",
    figsize: Tuple[int, int] = (7, 5),
    save_path: Optional[str] = None,
):
    """Plot SNR (dB) as a function of read noise sigma.

    Parameters
    ----------
    sigmas : list of float
        Read noise sigma values (relative fraction).
    snr_db : list of float
        Corresponding SNR values in dB.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(sigmas, snr_db, "o-", color="#2ecc71", linewidth=2, markersize=6)
    ax.set_xlabel("Read Noise Sigma (relative)")
    ax.set_ylabel("SNR (dB)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_adc_precision_sweep(
    bits_list: List[int],
    sqnr_db: List[float],
    *,
    title: str = "SQNR vs ADC Bit-Width",
    figsize: Tuple[int, int] = (7, 5),
    save_path: Optional[str] = None,
):
    """Plot SQNR (dB) vs ADC bit-width.

    Parameters
    ----------
    bits_list : list of int
        ADC resolutions tested.
    sqnr_db : list of float
        SQNR for each resolution in dB.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    save_path : str, optional
        Save path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(bits_list, sqnr_db, "s-", color="#9b59b6", linewidth=2, markersize=7)
    ax.set_xlabel("ADC Bit-Width")
    ax.set_ylabel("SQNR (dB)")
    ax.set_title(title, fontweight="bold")
    ax.set_xticks(bits_list)
    ax.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
