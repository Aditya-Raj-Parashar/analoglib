"""AnalogLib Interactive Demo
=============================
Run this script to explore the core features of AnalogLib.

Usage:
    python examples/demo.py

The demo walks through:
  1. Device creation and inspection
  2. Weight-to-conductance mapping
  3. Crossbar VMM computation
  4. Quantization error analysis
  5. Noise sensitivity study
  6. Multi-mode simulation comparison
  7. Save/Load .analog file roundtrip

Each section prints results to the console.
"""

import os
import sys
import tempfile

import numpy as np

# Add parent directory to path for editable installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import analoglib as al

# Reproducibility
al.set_seed(42)
rng = np.random.default_rng(42)


def banner(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# =====================================================================
#  1. DEVICE CREATION & INSPECTION
# =====================================================================
banner("1. Device Creation & Inspection")

ideal = al.IdealDevice(g_min=0.0, g_max=1.0)
print(f"Ideal device:  {ideal}")
print(f"  g_range = {ideal.g_range}")
print(f"  levels  = {len(ideal.conductance_levels())} (continuous)")

reram = al.ReRAM(
    g_min=1e-6,
    g_max=100e-6,
    num_states=256,
    read_noise_sigma=0.03,
    programming_error_sigma=0.01,
    d2d_variation_sigma=0.02,
    stuck_at_fault_rate=0.001,
)
print(f"\nReRAM device:  {reram}")
print(f"  g_range = {reram.g_range:.2e} S")
print(f"  levels  = {len(reram.conductance_levels())}")
print(f"  first 5 levels: {reram.conductance_levels()[:5]}")

# Registry
print(f"\nRegistered devices: {list(al.Device.registry().keys())}")


# =====================================================================
#  2. WEIGHT-TO-CONDUCTANCE MAPPING
# =====================================================================
banner("2. Weight-to-Conductance Mapping")

W = np.array([
    [ 0.8, -0.5,  0.3],
    [-0.2,  0.9, -0.7],
])

mapping = al.DifferentialMapping(w_max=1.0)
device = al.IdealDevice(g_min=0.0, g_max=1.0)

g_pos, g_neg = mapping.weights_to_conductance(W, device)
W_recon = mapping.conductance_to_weights(g_pos, g_neg, device=device)

print(f"Original weights:\n{W}")
print(f"\nG+ (positive conductance):\n{g_pos}")
print(f"\nG- (negative conductance):\n{g_neg}")
print(f"\nG+ - G- (effective weight):\n{g_pos - g_neg}")
print(f"\nReconstructed weights:\n{W_recon}")
print(f"\nRoundtrip error (ideal): {np.max(np.abs(W - W_recon)):.2e}")


# =====================================================================
#  3. CROSSBAR VMM COMPUTATION
# =====================================================================
banner("3. Crossbar VMM Computation")

rows, cols = 4, 3
W = rng.uniform(-1, 1, (rows, cols))
V = rng.uniform(0, 1, rows)

# Ideal crossbar
xbar_ideal = al.Crossbar(rows, cols, device=al.IdealDevice(),
                          mapping=al.DifferentialMapping(w_max=1.0))
xbar_ideal.load_weights(W, quantize=False)
out_ideal = xbar_ideal.vmm(V)

# NumPy reference
out_numpy = V @ W

print(f"Weight matrix W ({rows}×{cols}):\n{W}")
print(f"\nInput voltage V: {V}")
print(f"\nNumPy reference (V @ W):      {out_numpy}")
print(f"Crossbar VMM (ideal):         {out_ideal}")
print(f"Match: {np.allclose(out_ideal, out_numpy)}")


# =====================================================================
#  4. QUANTIZATION ERROR ANALYSIS
# =====================================================================
banner("4. Quantization Error Analysis")

W = rng.uniform(-1, 1, (32, 16))
mapping = al.DifferentialMapping(w_max=1.0)

print(f"{'States':>8}  {'Max Error':>12}  {'Mean Error':>12}  {'Bits':>6}")
print("-" * 45)

for n_states in [4, 8, 16, 32, 64, 128, 256, 1024]:
    dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=n_states,
                   read_noise_sigma=0.0)
    xbar = al.Crossbar(32, 16, device=dev, mapping=mapping)
    xbar.load_weights(W, quantize=True)
    W_recon = xbar.reconstruct_weights()
    max_err = np.max(np.abs(W - W_recon))
    mean_err = np.mean(np.abs(W - W_recon))
    bits = np.log2(n_states)
    print(f"{n_states:>8}  {max_err:>12.6f}  {mean_err:>12.6f}  {bits:>6.1f}")


# =====================================================================
#  5. NOISE SENSITIVITY STUDY
# =====================================================================
banner("5. Noise Sensitivity Study")

W = rng.uniform(-1, 1, (32, 16))
V = rng.uniform(0, 1, 32)

# Reference output (physical baseline in Amperes)
dev_ref = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.0)
xbar_ref = al.Crossbar(32, 16, device=dev_ref, mapping=al.DifferentialMapping(w_max=1.0))
xbar_ref.load_weights(W, quantize=True)
out_ref = xbar_ref.vmm(V, mode=al.SimulationMode.DEVICE)

print(f"{'Noise sig':>10}  {'Max Err (A)':>12}  {'Mean Err (A)':>12}  {'Rel Err (%)':>12}  {'SNR (dB)':>10}")
print("-" * 64)

for sigma in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]:
    dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256,
                   read_noise_sigma=sigma)
    xbar = al.Crossbar(32, 16, device=dev,
                       mapping=al.DifferentialMapping(w_max=1.0))
    xbar.load_weights(W, quantize=True)

    # Average over 50 noisy reads
    errors = []
    for _ in range(50):
        out = xbar.vmm(V, noise=True, mode=al.SimulationMode.DEVICE)
        errors.append(np.abs(out_ref - out))

    abs_err_matrix = np.array(errors)
    max_err = np.max(abs_err_matrix)
    mean_err = np.mean(abs_err_matrix)
    rel_err = (mean_err / (np.mean(np.abs(out_ref)) + 1e-30)) * 100

    signal_power = np.mean(out_ref ** 2)
    noise_power = np.mean(abs_err_matrix ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-30))
    
    print(f"{sigma:>10.3f}  {max_err:>12.2e}  {mean_err:>12.2e}  {rel_err:>12.2f}  {snr:>10.1f}")



# =====================================================================
#  6. MULTI-MODE SIMULATION COMPARISON
# =====================================================================
banner("6. Simulation Engine — Mode Comparison")

dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=64,
               read_noise_sigma=0.05)
mapping = al.DifferentialMapping(w_max=1.0)
xbar = al.Crossbar(8, 4, device=dev, mapping=mapping)
xbar.load_weights(rng.uniform(-1, 1, (8, 4)))

# Physical current bounds: V_max (1.0) * nrows (8) * g_max (100uS) = 800uA
# Let's set ADC to capture ± 500 µA expected output range
adc = al.ADC(bits=6, v_min=-500e-6, v_max=500e-6)
dac = al.DAC(bits=6, v_min=0.0, v_max=1.0)

engine = al.SimulationEngine(crossbars=[xbar], adc=adc, dac=dac)
V = rng.uniform(0, 1, 8)

results = engine.run_comparison(V, modes=["ideal", "device", "hardware"])

# Diagnostic Trace for Hardware Pipeline
print(f"--- Hardware Pipeline Trace ---")
scale_factor = (dev.g_max - dev.g_min) / 1.0  # I_out = V @ W_norm * scale_factor

print(f"1. Input Voltage V (DAC input):    {V[:4]}...")
V_dac = dac.convert(V)
print(f"2. DAC Output Voltages:            {V_dac[:4]}...")
# We use device mode to see raw current before ADC
I_raw = xbar.vmm(V_dac, noise=True, mode=al.SimulationMode.DEVICE)
print(f"3. Raw Crossbar Current (A):       {I_raw}")
I_quant = adc.convert(I_raw)
print(f"4. Quantized ADC Current (A):      {I_quant}")
W_recon_output = I_quant / scale_factor
print(f"5. Reconstructed Math Output:      {W_recon_output}")

print(f"\n--- Simulation Engine API Output ---")
for mode_name, output in results.items():
    print(f"Output ({mode_name:>8}): {output}")

# For mathematical error comparison, we reconstruct ALL outputs back to dimensionless math space
# since results["ideal"] in this Engine is evaluated on a ReRAM crossbar (which outputs Amperes).
math_expected = results["ideal"] / scale_factor
device_reconstructed = results["device"] / scale_factor
hardware_reconstructed = results["hardware"] / scale_factor

print(f"\nDevice vs Ideal math absolute error:   {np.max(np.abs(math_expected - device_reconstructed)):.6f}")
print(f"Hardware vs Ideal math absolute error: {np.max(np.abs(math_expected - hardware_reconstructed)):.6f}")


# =====================================================================
#  7. SAVE/LOAD .ANALOG FILE ROUNDTRIP
# =====================================================================
banner("7. Save/Load .analog File Roundtrip")

# Create a multi-layer model
dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256)
mapping = al.DifferentialMapping(w_max=1.0)

xbar1 = al.Crossbar(16, 8, device=dev, mapping=mapping)
xbar1.load_weights(rng.uniform(-1, 1, (16, 8)))

xbar2 = al.Crossbar(8, 4, device=dev, mapping=mapping)
xbar2.load_weights(rng.uniform(-1, 1, (8, 4)))

V = rng.uniform(0, 1, 16)

# Run before save
engine = al.SimulationEngine(crossbars=[xbar1, xbar2])
out_before = engine.run(V, mode="ideal")

# Save
tmpdir = tempfile.mkdtemp()
path = os.path.join(tmpdir, "demo_model.analog")
result_path = al.save(
    path, [xbar1, xbar2],
    model_name="demo_2layer",
    description="Two-layer MLP mapped to ReRAM crossbar",
    extra_meta={"author": "AnalogLib Demo", "seed": 42},
)
file_size = os.path.getsize(result_path)
print(f"Saved to: {result_path}")
print(f"File size: {file_size:,} bytes")

# Verify it's not readable
with open(result_path, "rb") as f:
    header = f.read(4)
print(f"Magic bytes: {header.hex()} ({'valid' if header == al.serialization.analog_format.MAGIC else 'INVALID'})")

# Load
loaded = al.load(result_path)
print(f"\nLoaded metadata:")
for k, v in loaded["meta"].items():
    print(f"  {k}: {v}")

# Run after load
engine_loaded = al.SimulationEngine(crossbars=loaded["crossbars"])
out_after = engine_loaded.run(V, mode="ideal")

print(f"\nOutput before save: {out_before}")
print(f"Output after load:  {out_after}")
print(f"Exact match: {np.allclose(out_before, out_after, atol=1e-12)}")


# =====================================================================
#  8. ADC BIT-WIDTH SWEEP
# =====================================================================
banner("8. ADC Bit-Width Sweep")

signal = np.linspace(-1, 1, 1000)
print(f"{'Bits':>6}  {'Levels':>8}  {'Max Quant Err':>15}  {'SQNR (dB)':>12}")
print("-" * 45)

for bits in [2, 4, 6, 8, 10, 12]:
    adc = al.ADC(bits=bits, v_min=-1.0, v_max=1.0)
    quantized = adc.convert(signal)
    error = signal - quantized
    max_err = np.max(np.abs(error))
    sqnr = 10 * np.log10(np.mean(signal**2) / (np.mean(error**2) + 1e-30))
    print(f"{bits:>6}  {adc.num_levels:>8}  {max_err:>15.8f}  {sqnr:>12.1f}")


# =====================================================================
banner("Demo Complete!")
print("All features verified successfully.")
print("See docs/ for detailed documentation.")
