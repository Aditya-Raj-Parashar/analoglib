# Converting PyTorch Models (`al.neural.from_torch`)

AnalogLib provides direct conversion of PyTorch `nn.Module` objects into `AIRGraph` and `AnalogModel`.

---

## 1. Supported PyTorch Layers

The PyTorch converter inspects the PyTorch computational graph and lowers supported modules:

- `nn.Linear` $\rightarrow$ Mapped to 2D Crossbar VMM layer.
- `nn.Conv2d` $\rightarrow$ Lowered via **im2col** sliding window unrolling into equivalent 2D crossbar VMM.
- `nn.ReLU`, `nn.Sigmoid`, `nn.Tanh`, `nn.Softmax` $\rightarrow$ Mapped to activation function nodes.
- `nn.Sequential` $\rightarrow$ Recursively unpacked into sequence.

---

## 2. Code Example: PyTorch MLP & Conv2d

```python
# Note: Requires optional dependency torch (`pip install "analoglib[torch]"`)
try:
    import torch
    import torch.nn as nn
    import analoglib as al

    # Define PyTorch model
    class SimpleNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(784, 128)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    pytorch_model = SimpleNet()

    # Convert PyTorch module to AnalogModel
    model = al.AnalogModel.from_torch(pytorch_model)

    # Compile and simulate
    model.compile(device=al.ReRAM(num_states=256))
    x_in = torch.randn(784).detach().numpy()
    out = model.simulate(x_in, mode="ideal")
    print("Inference completed! Output shape:", out.output.shape)

except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
```
