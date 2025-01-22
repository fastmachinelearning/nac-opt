# Neural Architecture Codesign for Fast Physics Applications
This repository contains the implementation of Neural Architecture Codesign (NAC), a framework for optimizing neural network architectures for physics applications with hardware efficiency in mind. NAC employs a two-stage optimization process to discover models that balance task performance with hardware constraints.

<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14618350.svg)](https://doi.org/10.5281/zenodo.14618350) -->

## Overview

NAC automates the design of deep learning models for physics applications while considering hardware constraints. The framework uses neural architecture search and network compression in a two-stage approach:

1. Global Search Stage: Explores diverse architectures while considering hardware constraints
2. Local Search Stage: Fine-tunes and compresses promising candidates
3. FPGA Synthesis (*optional*): Converts optimized models to FPGA-deployable code

The framework is demonstrated through two case studies:
- BraggNN: Fast X-ray Bragg peak analysis for materials science
- Jet Classification: Deep Sets architecture for particle physics

## Case Studies

The framework is demonstrated through two case studies:

### BraggNN
- Fast X-ray Bragg peak analysis for materials science
- Convolutional architecture with attention mechanisms
- Optimizes for peak position prediction accuracy and inference speed

### Deep Sets for Jet Classification 
- Particle physics classification using permutation-invariant architectures
- Optimizes classification accuracy and hardware efficiency


## Installation

1. Create a conda environment:
```bash
conda create --name NAC_env python=3.10.10
conda activate NAC_env
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download datasets:
- For BraggNN:
```bash
python data/get_dataset.py
```
- For Deep Sets: Download `normalized_data3.zip` and extract to `/data/normalized_data3/`

## Usage

### Global Search

Run architecture search for either BraggNN or Deep Sets:

```bash
python global_search.py
```

The script will output results to `global_search.txt`. For the Deep Sets model, results will be in `Results/global_search.txt`.

### Local Search

Run model compression and optimization:

```bash
python local_search.py
```

Results will be saved in `Results/deepsets_search_results.txt` or `Results/bragg_search_results.txt`.

## Directory Structure

```
.
├── data/                         # Dataset handling
├── examples/                     # Example configs and search spaces
│   ├── BraggNN/
│   └── DeepSets/
├── models/                       # Model architectures
├── utils/                        # Utility functions
├── global_search.py             # Global architecture search
├── local_search.py              # Local optimization
└── requirements.txt
```

## Architecture Search Methodology

### Global Search Stage
The global search explores a wide range of model architectures to find promising candidates that balance performance and hardware efficiency. This stage:

1. **Example Model Starting Points**: 
   - Uses pre-defined model configurations in `*_model_example_configs.yaml` as initial reference points
   - For BraggNN: includes baseline architectures like OpenHLS and original BraggNN
   - For Deep Sets: includes baseline architectures of varying sizes (tiny to large)

2. **Explores Architecture Space**:
   - Search space defined in `*_search_space.yaml` specifies possible model variations
   - For BraggNN: explores combinations of convolutional, attention, and MLP blocks
   - For Deep Sets: varies network widths, aggregation functions, and MLP architectures

3. **Multi-Objective Optimization**:
   - Uses NSGA-II algorithm to optimize both task performance and hardware efficiency
   - Evaluates models based on accuracy/mean distance and bit operations (BOPs)
   - Maintains diverse population of candidate architectures

Run global search with:
```bash
python global_search.py
```

### Local Search Stage
The local search takes promising architectures from the global search and optimizes them further through:

1. **Training Optimization**:
   - Fine-tunes hyperparameters using tree-structured Parzen estimation
   - Optimizes learning rates, batch sizes, and regularization

2. **Model Compression**:
   - Quantization-aware training (4-32 bits)
   - Iterative magnitude pruning (20 iterations, removing 20% parameters each time)
   - Evaluates trade-offs between model size, accuracy, and hardware efficiency

3. **Architecture Selection**:
   - Identifies best models across different operating points
   - Balances accuracy, latency, and resource utilization

Run local search with:
```bash
python local_search.py
```
## Results

The framework achieves:

### BraggNN
- 0.5% improved accuracy with 5.9× fewer BOPs (large model)
- 3% accuracy decrease for 39.2× fewer BOPs (small model)
- 4.92 μs latency with <10% FPGA resource utilization

### Jet Classification
- 1.06% improved accuracy with 7.2× fewer BOPs (medium model)
- 2.8% accuracy decrease for 30.25× fewer BOPs (tiny model)
- 70 ns latency with <3% FPGA resource utilization

