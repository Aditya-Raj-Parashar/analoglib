"""Regression tests for physical simulations."""

import numpy as np
import pytest

import analoglib as al


def test_noise_scaling_regression():
    """Verify that increasing Noise Sigma statistically increases the output variance and reduces SNR."""
    al.set_seed(42)
    rng = np.random.default_rng(42)

    rows, cols = 32, 16
    W = rng.uniform(-1, 1, (rows, cols))
    V = rng.uniform(0, 1, rows)

    # 1. Establish strict physical baseline using 0-noise ReRAM.
    dev_ref = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.0)
    xbar_ref = al.Crossbar(rows, cols, device=dev_ref, mapping=al.DifferentialMapping(w_max=1.0))
    xbar_ref.load_weights(W, quantize=True)
    out_ref = xbar_ref.vmm(V, mode=al.SimulationMode.DEVICE)

    prev_mean_err = -1.0

    for sigma in [0.01, 0.05, 0.1]:
        dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=sigma)
        xbar = al.Crossbar(rows, cols, device=dev, mapping=al.DifferentialMapping(w_max=1.0))
        xbar.load_weights(W, quantize=True)

        errors = []
        for _ in range(20):
            out = xbar.vmm(V, noise=True, mode=al.SimulationMode.DEVICE)
            errors.append(np.abs(out_ref - out))

        mean_err = np.mean(errors)
        # Verify that as noise sigma increases, the output variance firmly increases
        assert mean_err > prev_mean_err
        prev_mean_err = mean_err


def test_hardware_stage_scaling_regression():
    """Verify that ADC current clipping correctly maps to unscaled physical quantities."""
    al.set_seed(42)
    rng = np.random.default_rng(42)
    
    g_min, g_max = 1e-6, 100e-6
    dev = al.ReRAM(g_min=g_min, g_max=g_max, num_states=64)
    xbar = al.Crossbar(8, 4, device=dev, mapping=al.DifferentialMapping(w_max=1.0))
    xbar.load_weights(rng.uniform(-1, 1, (8, 4)))

    # Set ADC to proper ampere bounds
    adc = al.ADC(bits=6, v_min=-500e-6, v_max=500e-6)
    dac = al.DAC(bits=6, v_min=0.0, v_max=1.0)
    
    engine = al.SimulationEngine(crossbars=[xbar], adc=adc, dac=dac)
    V = rng.uniform(0, 1, 8)
    
    results = engine.run_comparison(V, modes=["ideal", "hardware"])
    math_expected = results["ideal"]
    
    scale_factor = (g_max - g_min) / 1.0
    hardware_reconstructed = results["hardware"] / scale_factor
    
    # Verify the absolute error between Math and Hardware is reasonable (quantization bounded)
    max_err = np.max(np.abs(math_expected - hardware_reconstructed))
    assert max_err < 5.0
    
    # Ensure it hasn't clipped to zero (saturation bug)
    assert np.any(np.abs(hardware_reconstructed) > 0.01)
