import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import analoglib as al

al.set_seed(42)
rng = np.random.default_rng(42)

def test_noise():
    print("--- Noise Test ---")
    W = rng.uniform(-1, 1, (32, 16))
    V = rng.uniform(0, 1, 32)
    
    # Reference (ideal device, mathematical output)
    out_ref_math = V @ W
    
    # We should use a baseline ReRAM device without noise as the physical reference!
    dev_baseline = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=0.0)
    xbar_baseline = al.Crossbar(32, 16, device=dev_baseline, mapping=al.DifferentialMapping(w_max=1.0))
    xbar_baseline.load_weights(W, quantize=False)
    out_ref_phys = xbar_baseline.vmm(V)
    
    # Observe unit scaling
    scale = (100e-6 - 1e-6) / 1.0
    print(f"out_ref_math (norm): {out_ref_math[:3]}")
    print(f"out_ref_phys (A):    {out_ref_phys[:3]}")
    print(f"out_ref_phys/scale:  {out_ref_phys[:3] / scale}")
    
    for sigma in [0.0, 0.05, 0.20]:
        dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=256, read_noise_sigma=sigma)
        xbar = al.Crossbar(32, 16, device=dev, mapping=al.DifferentialMapping(w_max=1.0))
        xbar.load_weights(W, quantize=True)
        
        out = xbar.vmm(V, noise=True, mode=al.SimulationMode.DEVICE)
        
        err = np.abs(out_ref_phys - out)
        print(f"sigma={sigma}, max err(A)={np.max(err):.2e}, mean err(A)={np.mean(err):.2e}")

def test_hardware():
    print("\n--- Hardware Saturation Test ---")
    dev = al.ReRAM(g_min=1e-6, g_max=100e-6, num_states=64, read_noise_sigma=0.0)
    mapping = al.DifferentialMapping(w_max=1.0)
    xbar = al.Crossbar(8, 4, device=dev, mapping=mapping)
    xbar.load_weights(rng.uniform(-1, 1, (8, 4)))
    
    # Look at ADC settings in demo
    adc = al.ADC(bits=6, v_min=-5e-3, v_max=5e-3)
    dac = al.DAC(bits=6, v_min=0.0, v_max=1.0)
    
    engine = al.SimulationEngine(crossbars=[xbar], adc=adc, dac=dac)
    V_env = rng.uniform(0, 1, 8)
    
    # Run hardware mode
    V_dac = dac.convert(V_env)
    I_out = xbar.vmm(V_dac, noise=True, mode=al.SimulationMode.HARDWARE)
    ADC_out = adc.convert(I_out)
    
    print(f"I_out before ADC: {I_out}")
    print(f"ADC output:       {ADC_out}")
    print(f"Why saturation at 7.9365e-5? Let's check ADC step size: {adc.resolution}")
    print(f"Ah! ADC is configured with v_min=-5e-3, v_max=5e-3 (Volts, or large Amperes?)")
    print(f"Wait, I_out is in Amperes (~ 10^-5 A) but ADC limits are 5e-3.")
    # Actually wait. If I_out is ~ 4e-5, and ADC v_max=5e-3...
    print(f"ADC.convert: clips to [-5e-3, 5e-3], then snaps to steps.")
    print(f"step = (5e-3 - -5e-3) / (2**6 - 1) = 10e-3 / 63 = 1.58e-4")
    print(f"Wait, if step is 1.58e-4, and I_out is 4e-5, then 4e-5 / 1.58e-4 ≈ 0.25.")
    print(f"It snaps to level 0 (which is 0.0) or level 1? ADC formula: step * round((x - v_min)/step) + v_min")

if __name__ == "__main__":
    test_noise()
    test_hardware()
