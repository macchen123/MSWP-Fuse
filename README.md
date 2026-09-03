# MSWP-Fuse

PyTorch implementation accompanying the paper:

**MSWP-Fuse: Station- and Regime-Adaptive Multi-scale Weather–Power Fusion for Day-ahead Wind and Solar Forecasting**

The repository provides training and inference code for MSWP-Fuse and the comparison methods used in the paper.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

## Data

The competition dataset is not distributed with this repository. The authors do not publish or redistribute it. Researchers seeking access should contact the competition organizers.

If you have authorized access to the dataset, pass the directory containing your local copy to `--data-root`. Do not commit dataset files to this repository.

## Usage

```bash
mswpfuse --help
mswpfuse train-parent --help
mswpfuse train-selected --help
mswpfuse predict --help
```

Paper configurations are provided under `configs/`.

Some comparison methods use pinned upstream implementations. Download them when needed with:

```bash
bash scripts/setup_upstreams.sh
```
