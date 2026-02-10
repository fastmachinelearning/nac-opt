# Rule4ML Package Documentation

## 1. Overview

**Rule4ML** is a Python package designed to estimate FPGA resource utilization (LUT, FF, DSP, BRAM) and latency (Clock Cycles) for machine learning models **pre-synthesis**. Instead of running a full High-Level Synthesis (HLS) and FPGA implementation flow (which can take hours), Rule4ML uses pre-trained regression models (MLPs) to predict these metrics in milliseconds.

It acts as a wrapper around the `hls4ml` workflow, effectively replacing the "Synthesis" step with a "Prediction" step during design space exploration.

---

## 2. Core API & Workflow

The primary entry point for prediction is the `MultiModelWrapper` class.

### Class: `MultiModelWrapper`

This class manages the loading of pre-trained surrogate models (MLPs) and orchestrates the feature extraction and prediction pipeline.

#### Usage Pattern

```python
from rule4ml.wrappers import MultiModelWrapper

# 1. Initialize and load default predictors
estimator = MultiModelWrapper()
estimator.load_default_models()

# 2. Predict (accepts Keras Model, PyTorch Module, or file path)
# Returns a Pandas DataFrame
predictions_df = estimator.predict(model_object, **config_overrides)
```

---

## 3. Data Structures & Inputs

### A. Model Inputs

The package accepts model objects directly from memory. It does not require serializing them to disk first, although it supports file paths.

- **Type**: `keras.Model`, `torch.nn.Module`, or `str` (path to `.h5`/`.onnx`)
- **Constraint**: The models must be compatible with `hls4ml` parsing (generally supported layers: Dense, Conv2D, BatchNormalization, ReLU, Softmax, Activation)

### B. Configuration (kwargs)

Configuration is passed as keyword arguments or a dictionary. These parameters mimic `hls4ml` configuration keys.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `Board` | `str` | `'pynq-z2'` | Target FPGA board (e.g., `pynq-z2`, `zcu102`, `alveo-u200`) |
| `Part` | `str` | Board specific | Specific FPGA part number |
| `ClockPeriod` | `int`/`float` | `5` | Target clock period in ns |
| `IOType` | `str` | `'io_parallel'` | Dataflow architecture: `'io_parallel'` (latency-optimized) or `'io_stream'` (throughput-optimized) |
| `Precision` | `str` | `'ap_fixed<16,6>'` | Default precision for the model. Format: `ap_fixed<total_bits, int_bits>` |
| `ReuseFactor` | `int` | `1` | Controls parallelism. `1` = fully parallel. Higher = more serial (saves resources) |
| `Strategy` | `str` | `'Latency'` | Optimization strategy (`'Latency'` or `'Resource'`) |

### C. Output Data

The output is always a **Pandas DataFrame**.

**Columns**:
- **Resource Metrics**: `BRAM`, `DSP`, `FF`, `LUT` (Integer values representing count)
- **Latency Metrics**: `Cycles` (Inference latency in clock cycles), `Interval` (Initiation Interval)
- **Percentages (Optional)**: Some methods return `%` columns based on the selected board's total capacity

---

## 4. Internal Feature Extraction

To make predictions, Rule4ML does not feed the raw model topology to its predictors. Instead, it extracts a **"Feature Vector"** that summarizes the model architecture.

### The Feature Vector (Input to ML Predictors)

The extraction logic traverses the model graph and aggregates counts. The internal MLPs expect a numerical vector containing features such as:

- **Layer Counts**: Number of Dense, Conv2D, BatchNormalization, ReLU, etc.
- **Neuron/Parameter Stats**: Total weights, total biases, total neurons
- **Operation Counts**: Estimated number of multiply-accumulate (MAC) operations
- **Configuration Features**: Numerical representation of `ReuseFactor`, `ClockPeriod`, and `Precision` (bit-width)

> **Note**: The extractors flatten the complex graph structure into a fixed-size tabular format suitable for the internal MLP regressors.

---

## 5. Critical Internal Logic

### A. Reuse Factor Validation (`_validate_reuse_factor`)

A specific logic exists to ensure the requested `ReuseFactor` is mathematically valid for the layer dimensions. Synthesis fails if this condition is not met.

#### Logic (Python equivalent)

```python
def _validate_reuse_factor(n_in, n_out, rf):
    multfactor = min(n_in, rf)
    # Calculate limit of multipliers
    multiplier_limit = int(math.ceil((n_in * n_out) / float(multfactor)))
    
    # Check divisibility constraints specific to HLS4ML implementation
    valid = ((multiplier_limit % n_out) == 0) or (rf >= n_in)
    valid = valid and (((rf % n_in) == 0) or (rf < n_in))
    valid = valid and (((n_in * n_out) % rf) == 0)
    
    return valid
```

**Parameters**:
- `n_in`: Input size
- `n_out`: Output size
- `rf`: Reuse Factor

**Output**: `Boolean`

### B. Helper: `get_closest_reuse_factor`

Because users might request an invalid Reuse Factor, the system includes a helper to find the nearest valid factor:

- Iterates `rf` from `1` to `n_in * n_out`
- Filters by `_validate_reuse_factor`
- Returns the valid `rf` closest to the requested value

---

## 6. Synthesis & Training (Data Generation Side)

> **Note**: This section applies if you are generating new training data, not just using the tool for prediction.

### Synthesizer Class

Handles the interaction with `hls4ml`.

**Input**: Takes a generator configuration (ranges for layers, neurons)

**Process**:
1. Generates a random Keras model
2. Calls `hls4ml.converters.convert_from_keras_model`
3. Calls `hls_model.build()` (triggers Vivado/Quartus synthesis)
4. Parses the resulting XML/JSON report files to extract ground truth `BRAM`, `DSP`, etc.