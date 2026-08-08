# Project Prompt: Build an Open-Source Analog Computing Library

## Role

You are a senior software architect, analog-computing researcher, ML engineer, and scientific-computing developer.

Your task is to **design and develop a serious, extensible software library for analog computing**, with an initial focus on **analog neural-network computation, resistive crossbar architectures, ReRAM/memristive devices, and hardware-aware simulation**.

Do not treat this as a simple Python utility or a toy simulator.

The long-term goal is to create an ecosystem where a researcher can go from:

**Neural Network → Analog Mapping → Device Model → Crossbar → Circuit-Level Model → Simulation → Hardware Export**

The library should be designed so that it can eventually support both **research experimentation and real analog/neuromorphic hardware development**.

---

# 1. Core Concept

The library should provide a software abstraction layer for analog computing.

A user should be able to describe an analog computational system using high-level Python APIs instead of manually implementing every physical operation.

For example:

```python
import analoglib as al

device = al.devices.ReRAM(
    g_min=1e-6,
    g_max=100e-6,
    levels=256
)

crossbar = al.Crossbar(
    rows=128,
    cols=64,
    device=device,
    differential=True
)

crossbar.load_weights(weights)

output = crossbar.vmm(input_vector)
```

The library should internally translate mathematical operations into analog hardware concepts:

```text
Weight
   ↓
Weight-to-Conductance Mapping
   ↓
Conductance Matrix
   ↓
Physical Crossbar
   ↓
Input Voltage
   ↓
Ohmic Current
   ↓
Column Current Summation
   ↓
Differential Readout
   ↓
ADC / Analog Output
```

The user should not need to manually calculate every intermediate physical quantity unless they want low-level access.

---

# 2. Main Design Philosophy

Follow these principles:

### Hardware-aware

Do not assume analog computation is mathematically ideal.

Support real-world effects such as:

* device conductance limits
* finite conductance states
* quantization
* device variation
* programming error
* read noise
* thermal noise
* stuck-at faults
* asymmetric positive/negative conductances
* nonlinear device behavior
* ADC/DAC quantization
* IR drop
* sneak-path effects where applicable
* limited voltage/current ranges
* saturation
* crossbar size limitations
* write/read errors

### Modular

Every major physical component should be replaceable.

For example:

```text
Device
Crossbar
Mapping
DAC
ADC
Amplifier
Neuron
Activation
Noise Model
Variation Model
Simulator
Exporter
```

should be independent modules.

A researcher should be able to replace one without rewriting the entire system.

### Framework-independent where possible

The core library should not depend entirely on PyTorch.

It should support:

* NumPy
* PyTorch
* potentially JAX later
* Python-native arrays

PyTorch integration can be an additional module.

---

# 3. Analog Model Representation

One of the most important features should be a **persistent analog model format**.

Create a dedicated file format, initially something like:

```text
.analog
```

The exact extension can be changed if you discover a better naming convention.

The file should represent an analog computational model rather than simply serializing Python objects.

For example:

```text
network.analog
```

should contain information such as:

```text
Model metadata
Device technology
Crossbar architecture
Layer configuration
Weight/conductance representation
Precision
Mapping strategy
ADC configuration
DAC configuration
Noise models
Variation models
Simulation parameters
Hardware constraints
```

The format should ideally be:

* human-readable where practical
* versioned
* portable
* deterministic
* extensible
* independent of Python implementation details

Consider using a structured format internally such as JSON/YAML/TOML, or a binary container if necessary.

Do not blindly choose one.

Evaluate the trade-offs first.

The library should provide:

```python
model.save("network.analog")
```

and:

```python
model = al.load("network.analog")
```

The saved model should contain enough information to reproduce the analog simulation.

---

# 4. Neural Network → Analog Conversion

A major goal is converting trained neural networks into analog hardware representations.

Example:

```python
model = MyNeuralNetwork()

analog_model = al.convert(
    model,
    target="ReRAM-crossbar"
)
```

The conversion pipeline should potentially perform:

```text
PyTorch / NumPy Model
        ↓
Extract Weights
        ↓
Quantization
        ↓
Weight Normalization
        ↓
Positive/Negative Mapping
        ↓
Conductance Mapping
        ↓
Crossbar Allocation
        ↓
Hardware Constraints
        ↓
Analog Model
```

Support differential representation such as:

[
G^+ = G_{min} + f(W^+)
]

and

[
G^- = G_{min} + f(W^-)
]

where positive and negative weights are represented through separate conductance paths.

Do not hard-code this single mapping.

Provide pluggable mapping strategies.

For example:

```python
al.mapping.DifferentialMapping()
al.mapping.SingleDeviceMapping()
al.mapping.OffsetMapping()
al.mapping.CustomMapping()
```

---

# 5. Crossbar Engine

The crossbar should be one of the core abstractions.

Example:

```python
crossbar = al.Crossbar(
    rows=128,
    cols=64,
    device=device
)
```

It should support:

* matrix dimensions
* conductance matrices
* voltage inputs
* current outputs
* VMM
* differential operation
* tiled crossbars
* multi-crossbar neural networks
* device-level constraints
* noise
* variation
* quantization

At minimum, support:

```python
I = crossbar.vmm(V)
```

with the physical interpretation:

[
I_j = \sum_i G_{ij}V_i
]

But the implementation should also allow more sophisticated physical models later.

---

# 6. Device Abstraction

Create a common device interface.

For example:

```python
device = al.devices.ReRAM(...)
```

Potential initial devices:

* Ideal conductance
* ReRAM
* Memristor
* PCM
* SRAM-based analog memory
* Generic programmable resistor

Each device should expose properties such as:

```text
G_min
G_max
number_of_states
programming_error
read_noise
nonlinearity
retention
variability
```

Do not pretend all devices behave identically.

The architecture should allow future device models to be added without modifying the crossbar engine.

---

# 7. Circuit-Level Abstraction

Eventually the library should move beyond matrix-level simulation.

Introduce abstractions for:

```text
Voltage Source
Current Source
Resistor
Memristive Device
Op-Amp
Transimpedance Amplifier
Differential Amplifier
ADC
DAC
Neuron
Activation Circuit
```

For example:

```python
amp = al.circuits.DifferentialAmplifier(...)
adc = al.circuits.ADC(bits=8)
```

The library should be capable of representing an analog computation at multiple abstraction levels.

Possible levels:

```text
L0 — Mathematical
L1 — Conductance / VMM
L2 — Device-aware
L3 — Circuit-aware
L4 — SPICE-level
```

The user should be able to choose the required fidelity.

---

# 8. SPICE / Circuit Export

A major feature should be exporting analog models into circuit simulators.

For example:

```python
model.export_spice("network.cir")
```

Potential targets:

* generic SPICE
* LTspice
* ngspice
* PSpice-compatible netlists where practical
* Verilog-A later

Do not assume every simulator uses identical syntax.

Create separate exporters:

```text
export/
    spice.py
    ltspice.py
    ngspice.py
    verilog_a.py
```

The generated netlist should correspond to the actual analog model rather than being a fake representation.

---

# 9. Simulation Engine

Provide multiple simulation modes.

### Ideal simulation

Fast mathematical simulation.

```python
model.simulate(mode="ideal")
```

### Device-aware simulation

Include:

* conductance quantization
* variation
* noise
* nonlinearities

```python
model.simulate(mode="device")
```

### Hardware-aware simulation

Include:

* ADC/DAC
* voltage limits
* current limits
* IR drop
* amplifier behavior
* crossbar constraints

```python
model.simulate(mode="hardware")
```

### Circuit simulation

Generate or interface with SPICE.

```python
model.simulate(mode="spice")
```

Do not implement everything immediately.

Design the architecture so these modes can evolve independently.

---

# 10. Analysis Tools

The library should provide tools for analyzing an analog model.

Examples:

```python
model.power()
model.energy()
model.latency()
model.error()
model.accuracy()
model.area()
```

Potential outputs:

```text
Energy per MAC
Energy per inference
Power consumption
Latency
Effective precision
Quantization error
Analog computation error
Device variation error
ADC energy
DAC energy
Crossbar utilization
```

Where exact physical values cannot be reliably calculated, clearly identify them as estimates.

Never present an approximation as a physically validated result.

---

# 11. Hardware-Aware Neural Network Evaluation

A user should be able to compare:

```text
Digital Model Accuracy
vs
Ideal Analog Accuracy
vs
Device-Aware Analog Accuracy
vs
Hardware-Constrained Accuracy
```

For example:

```python
report = analog_model.evaluate(
    test_data,
    modes=[
        "digital",
        "ideal",
        "device",
        "hardware"
    ]
)
```

Generate useful reports showing where accuracy is lost.

---

# 12. Visualization

Provide visualization utilities.

Examples:

```python
crossbar.plot()
crossbar.plot_conductance()
model.plot_error()
model.plot_device_variation()
model.plot_current()
```

Possible visualizations:

* conductance heatmap
* weight heatmap
* positive/negative conductance matrices
* crossbar architecture
* current flow
* layer mapping
* quantization levels
* error distribution
* device variation
* ADC quantization
* power/accuracy trade-off

Keep visualization as a separate module.

---

# 13. Automatic Hardware Mapping

A higher-level API should eventually allow:

```python
analog_model = al.compile(
    neural_network,
    hardware={
        "device": "ReRAM",
        "crossbar_size": (128, 128),
        "adc_bits": 8,
        "dac_bits": 8
    }
)
```

The compiler should determine:

* number of crossbars
* weight partitioning
* tiling
* conductance mapping
* required ADCs
* required DACs
* layer connectivity
* hardware constraints

This should behave conceptually like a compiler:

```text
High-Level Neural Network
          ↓
Analog Intermediate Representation
          ↓
Hardware Mapping
          ↓
Physical Analog Model
          ↓
Simulation / Export
```

Consider creating an **Analog Intermediate Representation (AIR)**.

This could become one of the most important architectural components of the project.

---

# 14. Analog Intermediate Representation

Design an intermediate representation that sits between ML frameworks and hardware.

For example:

```text
Neural Network
      ↓
     AIR
      ↓
Device Mapping
      ↓
Crossbar Graph
      ↓
Circuit Representation
```

The AIR should describe concepts such as:

```text
AnalogTensor
AnalogLayer
ConductanceMatrix
Crossbar
Device
Signal
ADC
DAC
Amplifier
Connection
```

This will allow the library to support different backends later.

---

# 15. Extensibility / Plugin System

Design the system so researchers can add their own:

* devices
* mapping algorithms
* noise models
* circuit models
* exporters
* optimization algorithms
* hardware backends

without modifying the core library.

For example:

```python
@al.register_device
class MyExperimentalDevice(al.Device):
    ...
```

---

# 16. Command-Line Interface

Eventually provide a CLI.

Example:

```bash
analog compile model.pt --target reram
```

```bash
analog simulate network.analog
```

```bash
analog inspect network.analog
```

```bash
analog export network.analog --format spice
```

```bash
analog analyze network.analog
```

---

# 17. Reproducibility

Scientific reproducibility is extremely important.

Every `.analog` model should ideally record:

* library version
* model version
* device configuration
* mapping method
* random seed
* simulation parameters
* precision
* hardware assumptions
* backend
* timestamp
* optional experiment metadata

A researcher should be able to load an old `.analog` file years later and understand what it represents.

---

# 18. Testing

Do not build the library without extensive tests.

Create tests for:

### Mathematical correctness

Verify:

[
I = GV
]

and matrix VMM behavior.

### Mapping correctness

Verify:

```text
weight → conductance → reconstructed weight
```

### Quantization

Verify expected precision and error.

### Differential operation

Verify:

```text
I_positive - I_negative
```

corresponds correctly to signed weights.

### Device models

Test boundary conditions.

### Serialization

Verify:

```python
model.save()
model = load()
```

produces equivalent behavior.

### Export

Verify generated SPICE/netlists are structurally valid.

### Reproducibility

Same seed + same model + same parameters should produce reproducible results.

---

# 19. Documentation

Create proper documentation from the beginning.

Include:

```text
Getting Started
Installation
Core Concepts
Analog Computing Basics
Device Models
Crossbars
Weight Mapping
Neural Network Conversion
Simulation
Noise & Variation
Circuit Models
SPICE Export
Analog File Format
AIR
CLI
Developer Guide
Adding New Devices
Adding New Backends
Research Examples
```

Include examples rather than only API references.

---

# 20. Project Structure

Propose a clean architecture before implementation.

A possible starting point:

```text
analoglib/
│
├── core/
├── devices/
├── mapping/
├── crossbar/
├── circuits/
├── analog_ir/
├── simulation/
├── neural/
├── analysis/
├── visualization/
├── serialization/
├── exporters/
├── backends/
├── cli/
└── tests/
```

But **do not blindly follow this structure**.

Analyze it first and modify it if you find a better architecture.

---

# 21. Important: Tinker and Improve the Concept

Do NOT simply implement exactly what this prompt says.

You are explicitly encouraged to:

* question architectural decisions
* identify missing concepts
* identify unrealistic assumptions
* research better approaches
* propose alternative designs
* experiment with small prototypes
* benchmark competing approaches
* simplify unnecessarily complex components
* add useful features that logically belong in the ecosystem
* remove features that are premature or poorly designed

If you discover that a different architecture would be significantly better, explain why and use it.

The objective is to build the **best practical analog-computing software architecture**, not to mechanically follow this document.

---

# 22. Research / Exploration Mode

Before committing to major architectural decisions, investigate relevant existing approaches and concepts.

Look into areas such as:

* analog neural-network simulators
* memristor simulation frameworks
* neuromorphic computing frameworks
* SPICE integration
* PyTorch hardware-aware simulation
* MLIR/compiler-style intermediate representations
* analog in-memory computing
* ReRAM crossbar simulation
* circuit netlist generation
* model serialization formats

Do not copy existing projects.

Study their architecture, identify strengths and weaknesses, and use those observations to improve this project.

Clearly distinguish:

```text
Known / established
vs
Proposed by us
vs
Experimental
```

---

# 23. MOST IMPORTANT: PLAN BEFORE ACTION

Before writing substantial code, **STOP and prepare a detailed implementation plan.**

Do not immediately start generating the entire repository.

First provide:

## Phase 0 — Architecture Analysis

Explain:

* what the library is
* what problem it solves
* target users
* abstraction levels
* major components
* data flow
* design philosophy

## Phase 1 — Architecture

Provide:

* package structure
* class hierarchy
* module responsibilities
* dependency graph
* data structures
* API design
* Analog Intermediate Representation

## Phase 2 — File Format

Define:

* `.analog` format
* schema
* versioning
* serialization
* compatibility strategy

## Phase 3 — MVP

Identify the smallest useful implementation.

Prefer something like:

```text
Device
   ↓
Weight Mapping
   ↓
Crossbar
   ↓
Analog VMM
   ↓
Simulation
   ↓
Save/Load .analog
```

Do not attempt to implement every advanced feature immediately.

## Phase 4 — Development Roadmap

Create milestones such as:

```text
M0 — Architecture
M1 — Core abstractions
M2 — Ideal crossbar
M3 — Device models
M4 — Analog file format
M5 — Neural-network conversion
M6 — Hardware-aware simulation
M7 — SPICE export
M8 — Analysis tools
M9 — CLI
M10 — Documentation
```

Modify these milestones if your analysis suggests something better.

---

# 24. After the Plan

Once the plan is presented, identify:

### Critical risks

What could make the architecture fail?

### Technical uncertainties

What needs experimentation?

### MVP boundaries

What should NOT be implemented initially?

### Performance risks

Where could simulation become computationally expensive?

### Scientific validity risks

Which results might be misleading without circuit-level validation?

### Compatibility risks

Which external tools/frameworks may create problems?

---

# 25. Then Tinker

After presenting the plan, do a small amount of exploratory implementation.

Build tiny prototypes for the most uncertain architectural decisions.

For example:

```text
Prototype 1:
Weight → Conductance → Weight reconstruction

Prototype 2:
Conductance matrix → VMM

Prototype 3:
Differential crossbar

Prototype 4:
Analog model serialization

Prototype 5:
SPICE netlist generation
```

Use these prototypes to validate the architecture.

If a prototype reveals a problem, **change the architecture rather than forcing the prototype to fit the original design.**

---

# 26. Development Rules

Follow these rules throughout development:

1. Do not create unnecessary abstractions.
2. Do not hard-code ReRAM everywhere.
3. Keep device models independent from crossbars.
4. Keep simulation independent from visualization.
5. Keep serialization independent from Python class internals.
6. Avoid circular dependencies.
7. Prefer clear APIs over clever APIs.
8. Write tests alongside implementation.
9. Document scientific assumptions.
10. Never claim circuit-level accuracy from an ideal mathematical model.
11. Clearly separate exact calculations from approximations.
12. Design for future hardware technologies.
13. Maintain backward compatibility for `.analog` files.
14. Optimize only after profiling.
15. Keep the MVP usable.

---

# 27. Desired End-State

Ultimately I want this ecosystem to allow a researcher to do something conceptually like:

```python
import analoglib as al

# Load trained neural network
nn = al.load_neural_network("model.pt")

# Compile it for analog hardware
analog = al.compile(
    nn,
    device="ReRAM",
    crossbar_size=(128, 128),
    adc_bits=8,
    dac_bits=8
)

# Inspect mapping
analog.inspect()

# Simulate
result = analog.simulate(
    mode="hardware"
)

# Analyze
report = analog.analyze()

# Save portable analog model
analog.save("model.analog")

# Export circuit
analog.export("model.cir", format="spice")
```

And another researcher should be able to simply do:

```python
analog = al.load("model.analog")

analog.simulate()
analog.inspect()
analog.analyze()
```

without needing the original Python training environment.

---

# 28. Your First Response

**Do NOT start by writing the complete codebase.**

Your first response to this prompt must contain only the planning/analysis stage.

Give me:

1. Your understanding of the project
2. Proposed architecture
3. Core concepts
4. Analog Intermediate Representation proposal
5. `.analog` file-format proposal
6. Core class/API proposal
7. Data flow
8. MVP definition
9. Development phases
10. Risks and technical challenges
11. What you would change/improve from this specification
12. A list of experiments/prototypes you want to run
13. Recommended technology stack
14. Proposed repository structure
15. A concrete step-by-step implementation roadmap

**Do not implement the full project yet.**

Think like a researcher and architect first.

Challenge my assumptions.

Tinker with the architecture mentally and, where appropriate, through small isolated experiments.

Once the plan is complete and internally consistent, wait for approval before beginning the main implementation.
