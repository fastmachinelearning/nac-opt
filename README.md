# SNAC-Pack: Surrogate Neural Architecture Codesign Package

SNAC-Pack is a TensorFlow-based framework for hardware-aware neural architecture search (NAS) targeting FPGA deployment. It automates the discovery of neural network architectures that balance task performance against hardware resource constraints.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14618350.svg)](https://doi.org/10.5281/zenodo.14618350)

## Overview

SNAC-Pack performs two-stage optimization to find Pareto-optimal models:

1. **Global Search**: NSGA-II multi-objective search (via Optuna) over a configurable architecture space. Objectives include accuracy, bit operations (BOPs), and optionally FPGA resource utilization and clock cycles estimated by `rule4ml`.
2. **Local Search**: Quantization-aware training (QAT via QKeras) combined with iterative magnitude pruning (via TF Model Optimization), further compressing the best candidates from global search.

The result is a set of deployable, quantized, pruned models with known FPGA resource footprints, ready for downstream HLS synthesis with `hls4ml`.

## Installation

Create and activate a conda environment:

```bash
conda create -n snac-pack python=3.10
conda activate snac-pack
```

Install the core TensorFlow stack (pinned for compatibility with QKeras and rule4ml):

```bash
pip install \
  "tensorflow==2.15.1" \
  "keras==2.15.0" \
  "tensorboard==2.15.2" \
  "protobuf==4.25.8" \
  "ml-dtypes==0.3.2" \
  "tensorflow-estimator==2.15.0" \
  "tensorflow-io-gcs-filesystem==0.37.1" \
  "tensorflow-model-optimization==0.7.5"
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Tutorials

Three self-contained tutorials are provided, each with a Python script, Jupyter notebook, and YAML config. All parameters (search space, dataset, epochs, local search settings) are controlled through the config file — no code changes needed for common experiments.

Run from the repo root:

```bash
python tutorials/tutorial_1_mlp/tutorial_1_mlp.py     # MLP search on MNIST
python tutorials/tutorial_2_block/tutorial_2_block.py  # Block-based search on Fashion-MNIST
python tutorials/tutorial_3_qubit/tutorial_3_qubit.py  # Qubit readout classification
```

| Tutorial | Dataset | Model Types | Hardware Metrics |
|----------|---------|-------------|-----------------|
| 1 — MLP | MNIST | MLP | Yes |
| 2 — Block | Fashion-MNIST | Conv, ConvAttn, MLP | No |
| 3 — Qubit | Superconducting qubit IQ data | MLP | Yes |

### Config-Driven Workflow

Each tutorial reads all parameters from its config YAML. For example, to run more trials or enable hardware metrics in Tutorial 2, edit `tutorials/tutorial_2_block/t2_config.yaml`:

```yaml
search:
  n_trials: 50
  epochs: 20
  use_hardware_metrics: true
  objective_names: [performance_metric, bops, avg_resource, clock_cycles]
  maximize_flags: [true, false, false, false]
```

Results (CSV, Pareto front plots, per-trial YAML configs, best model config) are written to the directory specified by `output.results_dir` in the config.

### Multi-Node Search (Tutorial 3)

Tutorial 3 supports distributed search across multiple nodes via SLURM:

```bash
cd tutorials/tutorial_3_qubit

# Single node via CLI
python run_global_search.py --n_trials 1000 --epochs 20

# With shared Optuna storage (SQLite or PostgreSQL)
python run_global_search.py \
    --n_trials 1000 \
    --optuna_storage "sqlite:///./optuna.db" \
    --optuna_study_name "qubit_search"

# Multi-node via SLURM
sbatch run_global_search_slurm.sh
```

Each worker writes its own CSV (`*_rank{N}.csv`) to avoid race conditions; merge them after the job completes.

## Repository Structure

```
tutorials/
  tutorial_1_mlp/        t1_config.yaml, tutorial_1_mlp.py, tutorial_1_mlp.ipynb
  tutorial_2_block/      t2_config.yaml, tutorial_2_block.py, tutorial_2_block.ipynb
  tutorial_3_qubit/      t3_config.yaml, tutorial_3_qubit.py, tutorial_3_qubit.ipynb
                         run_global_search.py, run_global_search_slurm.sh
extras/
  animated_search_viz.py          Live Optuna search visualization callback
  create_3d_animation.py          3D Pareto front animation
  plot_global_search_csv.py       Post-hoc Pareto front plots from CSV
  plot_global_search_from_csv.ipynb
  qubit_iq_visualization.ipynb
utils/
  tf_global_search.py             GlobalSearchTF, Optuna objectives, rule4ml integration
  tf_local_search_combined.py     Combined QAT + iterative pruning with k-fold CV
  tf_local_search_separated.py    Separate QAT/pruning phases; load_model_from_yaml()
  tf_model_builder.py             MLP and DeepSets model builders
  tf_blocks.py                    Block-based architecture primitives
  tf_bops.py                      BOPs calculations for Dense/Conv/attention layers
  tf_data_preprocessing.py        Dataset loaders; load_generic_dataset() dispatcher
  tf_processor.py                 train_model(), evaluate_model() with early stopping
  tf_visualization.py             Pareto front plotting utilities
data/
  qubit_dataset.py                Qubit IQ data loader
  qubit/                          Raw qubit .npy data files (X/y train+test)
pytorch_NAC/                      Archived PyTorch NAC code (not used by TF pipeline)
```

## Architecture Search Details

### Search Space

The search space is defined in each tutorial's config YAML under the `search_space` key. The block-based search (Tutorials 2 & 3) supports four block types that can be combined in any order:

- **MLP** — fully connected layers with configurable width, activation, and normalization
- **Conv** — 2D convolution blocks
- **ConvAttn** — convolutional attention blocks
- **None** — skip connection (identity)

### Hardware Estimation

When `use_hardware_metrics: true`, SNAC-Pack uses `rule4ml` to estimate FPGA resource utilization (LUT%, FF%, DSP%, BRAM%) and clock cycles before synthesis. Models are automatically flattened and activations normalized to ReLU for compatibility with the estimator. These estimates are used as additional Optuna objectives, enabling the search to find models that fit within target resource budgets.

### Local Search

The local search takes the best architecture from global search and applies:

1. **Quantization-aware training** — sweeps over configurable `(total_bits, int_bits)` precision pairs using QKeras
2. **Iterative magnitude pruning** — applies structured sparsity over multiple rounds, rewinding weights via Lottery Ticket Hypothesis (combined mode)


## Citation

If you use SNAC-Pack in your research, please cite:

```bibtex
@misc{weitz2025neuralarchitecturecodesignfast,
      title={Neural Architecture Codesign for Fast Physics Applications},
      author={Jason Weitz and Dmitri Demler and Luke McDermott and Nhan Tran and Javier Duarte},
      year={2025},
      eprint={2501.05515},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2501.05515},
}
```
