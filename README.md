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

The anonymized dataset is available from [Google Drive](https://drive.google.com/file/d/1CofstEZ8K54DXUOB8Q1A4p2lBurbC5gS/view?usp=sharing).

After extraction, pass the directory containing the dataset to `--data-root`.

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
