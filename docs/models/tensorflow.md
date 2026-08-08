# TensorFlow / Keras Framework Compatibility

> [!WARNING]
> **Status: 🚧 Planned / Not Implemented in AnalogLib v0.1.0**
> Direct conversion of TensorFlow or Keras model objects (`al.neural.from_keras` or `al.neural.from_tf`) is **not implemented** in v0.1.0. This feature is scheduled for the v0.2.0 release.

---

## Migration Workaround for TensorFlow / Keras Users

To simulate Keras / TensorFlow models in AnalogLib v0.1.0, extract the layer weights as NumPy arrays using `.get_weights()` and load them using `al.neural.from_numpy()` or `AnalogModel.from_numpy()`:

```python
import analoglib as al
import numpy as np

# 1. Extract weights from your TensorFlow / Keras model
# keras_model = tf.keras.Sequential(...)
# W1, b1 = keras_model.layers[0].get_weights()
# W2, b2 = keras_model.layers[1].get_weights()

# Simulated extracted NumPy weight matrices
W1 = np.random.randn(784, 128)
W2 = np.random.randn(128, 10)

# 2. Build AnalogModel from NumPy weight arrays
model = al.AnalogModel.from_numpy(
    weights=[W1, W2],
    activations=["relu", "softmax"],
    name="keras_migrated_model",
)

# 3. Compile and simulate
model.compile(device=al.ReRAM(num_states=256), adc_bits=8, dac_bits=8)
x_in = np.random.uniform(0, 1, 784)
result = model.simulate(x_in, mode="hardware")

print("Migration simulation successful! Output:", result.output.shape)
```
