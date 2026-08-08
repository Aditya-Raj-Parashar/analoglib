# Hardware Simulation Mode (`SimulationMode.HARDWARE`)

Hardware mode provides full end-to-end physical simulation across peripherals and array non-idealities:

$$\text{Input Activation} \xrightarrow{\text{DAC}} V_{\text{in}} \xrightarrow{\text{Crossbar VMM}} I_{\text{out}} \xrightarrow{\text{ADC}} \text{Digital Output}$$

---

## Solved Non-Idealities

1. **DAC Quantization**: Input activations clipped and quantized to DAC voltage resolution.
2. **Parasitic IR Drop**: Spatial voltage drop along rows and columns based on `r_wire`.
3. **Arrhenius Thermal Scaling**: Conductance modification based on temperature $T$.
4. **Power-law Drift**: Retention loss relaxation over elapsed time $t$.
5. **Read Noise & Faults**: Cell conductance readout noise and stuck-at defects.
6. **ADC Quantization & Clipping**: Output current clipping to $[v_{\text{min}}, v_{\text{max}}]$ range and uniform ADC quantization.

---

## Code Example

```python
import analoglib as al
import numpy as np

device = al.ReRAM(num_states=256, read_noise_sigma=0.01)
xbar = al.Crossbar(64, 32, device=device)
xbar.load_weights(np.random.randn(64, 32))

g_pos, _ = xbar.get_conductance()

# Attach physical effects
context = al.effects.EffectContext(V_row=np.ones(64) * 0.5, G=g_pos, T_kelvin=320.0, t_seconds=3600.0)
ir_drop = al.effects.IRDrop(r_wire=1.0)
thermal = al.effects.Thermal(E_a=0.1)
drift = al.effects.Drift(nu=0.05)



engine = al.SimulationEngine(
    crossbars=[xbar],
    adc=al.ADC(bits=8),
    dac=al.DAC(bits=8),
)

V_in = np.random.uniform(0, 1, 64)
out_hardware = engine.run(V_in, mode="hardware")
print("Hardware mode output shape:", out_hardware.shape)
```
