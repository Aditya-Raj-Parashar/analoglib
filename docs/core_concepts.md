# Core Concepts & Mathematical Formulation

**AnalogLib** models physical analog in-memory computing (IMC) architectures using first-principles physics and circuit theory. This document provides the mathematical derivations, physical equations, and architectural concepts behind AnalogLib.

---

## 1. Conductance Mapping Strategies

Weights in a digital neural network $W \in [-w_{\text{max}}, w_{\text{max}}]$ are mapped onto physical device conductances $G \in [G_{\text{min}}, G_{\text{max}}]$.

### Differential Conductance Mapping (`DifferentialMapping`)
To represent positive and negative weights, each logical matrix entry $W_{i,j}$ uses two physical memristive cells: a positive cell $G^+_{i,j}$ and a negative cell $G^-_{i,j}$.

Given weight scaling factor:
$$\alpha = \frac{G_{\text{max}} - G_{\text{min}}}{2 \cdot w_{\text{max}}}$$
And midpoint conductance:
$$G_{\text{mid}} = \frac{G_{\text{max}} + G_{\text{min}}}{2}$$

The mapped conductances are defined as:
$$G^+_{i,j} = G_{\text{mid}} + \frac{1}{2} \alpha W_{i,j}$$
$$G^-_{i,j} = G_{\text{mid}} - \frac{1}{2} \alpha W_{i,j}$$

#### Net Differential Conductance & Current:
$$\Delta G_{i,j} = G^+_{i,j} - G^-_{i,j} = \alpha W_{i,j}$$
$$I_j = \sum_i V_i (G^+_{i,j} - G^-_{i,j}) = \sum_i V_i \Delta G_{i,j} = \alpha \sum_i V_i W_{i,j}$$

---

## 2. Device Non-Idealities & Conductance Quantization

### Discrete Conductance States (`ReRAM`)
Physical NVM cells (ReRAM/PCM) possess a finite number of programmable conductance levels $N_{\text{states}}$.

Level step size:
$$\Delta G = \frac{G_{\text{max}} - G_{\text{min}}}{N_{\text{states}} - 1}$$

Quantization function:
$$Q(G) = G_{\text{min}} + \text{round}\left( \frac{G - G_{\text{min}}}{\Delta G} \right) \cdot \Delta G$$

### Conductance Read Noise
During readout, thermal and random telegraph noise (RTN) perturb the cell conductance:
$$\tilde{G} = Q(G) + \delta G, \quad \delta G \sim \mathcal{N}\left(0, \sigma_{\text{read}}^2 (G_{\text{max}} - G_{\text{min}})^2\right)$$

---

## 3. Hardware Non-Ideality Effects (`analoglib.effects`)

### 1. Parasitic Wire IR Drop (`IRDrop`)
Wordlines and bitlines have finite parasitic wire resistance $r_{\text{wire}}$ per cell segment. The effective row voltage decreases along the array:
$$V_{\text{eff}}[i, j] = V_i \cdot \left(1 - r_{\text{wire}} \sum_{j'=0}^{j-1} G_{i, j'} \cdot (N_{\text{cols}} - j') \right)$$

### 2. Arrhenius Thermal Scaling (`Thermal`)
ReRAM conductance follows Arrhenius temperature dependence:
$$G(T) = G_0 \cdot \exp\left( -\frac{E_a}{k_B} \left( \frac{1}{T} - \frac{1}{T_{\text{ref}}} \right) \right)$$
where $E_a$ is the activation energy (eV), $k_B = 8.6173 \times 10^{-5}\text{ eV/K}$, and $T_{\text{ref}} = 300\text{ K}$.

### 3. Power-Law Retention Drift (`Drift`)
State relaxation over time follows power-law decay:
$$G(t) = G_0 \cdot \left( \frac{\max(t, t_0)}{t_0} \right)^{-\nu}$$
where $\nu$ is the drift exponent ($\nu \approx 0.02 - 0.12$) and $t_0$ is the reference time (1 sec).

---

## 4. Peripheral Converters & Hardware Metrics

### Uniform ADC / DAC Quantization (`ADC`, `DAC`)
The DAC converts input activations $x \in [0, 1]$ into voltages $V \in [V_{\text{min}}, V_{\text{max}}]$ with $B_{\text{dac}}$ bits.
The ADC converts column currents $I \in [I_{\text{min}}, I_{\text{max}}]$ into digital values with $B_{\text{adc}}$ bits.

### Power & Energy Profiling (`AnalogProfiler`)
- Array Read Power:
  $$P_{\text{array}} = \sum_{i,j} V_i^2 \cdot G_{i,j}$$
- Read Energy:
  $$E_{\text{read}} = P_{\text{array}} \cdot t_{\text{read}}$$
- Energy Efficiency (TOPS/W):
  $$\text{TOPS/W} = \frac{2 \cdot M \cdot N}{(E_{\text{read}} + E_{\text{adc}} + E_{\text{dac}}) \times 10^{12}}$$
