# MSWP-Fuse

PyTorch implementation accompanying the paper:

**MSWP-Fuse: Station- and Regime-Adaptive Multi-scale Weather–Power Fusion for Day-ahead Wind and Solar Forecasting**

The repository provides the model, preprocessing, training, validation selection, and inference code for MSWP-Fuse and its comparison methods. It includes the six candidate configurations, reported module controls, weather-condition comparisons, and six-instance training-stability summary described in the paper and Supporting Information.

## Installation

Python 3.10 or newer is required. From a checkout of this repository:

```bash
python -m pip install -e .
mswpfuse --help
```

## Data

The competition dataset is not distributed with this repository. The authors do not publish or redistribute it. Researchers seeking access should contact the competition organizers.

If you have authorized access, pass the directory containing your local copy to `--data-root`. The loader expects the competition layout, including `train/fact_data_train/` and `train/nwp_data_train/`. Dataset files, fitted checkpoints, scalers, and prediction archives must remain local and must not be committed or uploaded to this repository.

The loader constructs seven days of quarter-hourly power history and a 96-step forecast. NWP standardization is fitted on the current window's training partition only. Training commands construct training and validation samples; evaluation samples are opened by the prediction commands.

## Train the reported configuration

The reference fit keeps the existing seed-42 setting and three successive stages:

1. Fit the complete history initializer using training data, selecting its checkpoint on the same window's validation month.
2. Transfer its history encoder into the backbone and fit the backbone for 18 epochs. Transferred history weights are frozen for the first three epochs.
3. Adapt the temporal residual and output head, selecting the adaptation checkpoint on validation data.

All stages run locally on the device selected with `--device`. CUDA reproductions use deterministic algorithms and `CUBLAS_WORKSPACE_CONFIG=:4096:8`, which the CLI sets before importing PyTorch.

```bash
DATA_ROOT=/path/to/authorized/raw_data

mswpfuse train-initializer --data-root "$DATA_ROOT" --fold summer \
  --output runs/summer/initializer.pt

mswpfuse train-parent --data-root "$DATA_ROOT" --fold summer \
  --initializer-checkpoint runs/summer/initializer.pt \
  --output runs/summer/parent.pt

mswpfuse train-selected --data-root "$DATA_ROOT" --fold summer \
  --parent-checkpoint runs/summer/parent.pt \
  --output runs/summer/selected.pt

mswpfuse predict --data-root "$DATA_ROOT" --fold summer --split test \
  --checkpoint runs/summer/selected.pt --output results/full/summer.npz
```

Repeat for `autumn` and `early_winter`, using separate outputs. Without `--configuration`, `train-selected` uses the configuration reported for that window:

| Window | Validation month | Evaluation months | Configuration | Weather-GRU width per direction |
| --- | --- | --- | --- | --- |
| Summer | June 2024 | July–August 2024 | 4 | 32 |
| Autumn | August 2024 | September–October 2024 | 1 | 24 |
| Early winter | October 2024 | November–December 2024 | 1 | 24 |

All architecture, optimizer, loss-share, and schedule settings are in `configs/paper.yaml`. A checkpoint records its configuration, fold, scaler, and trainable-parameter policy, so inference does not silently rebuild every model with one default width. Outputs are create-only. Checkpoints from the superseded model export are not compatible with the current, source-aligned topology.

## Repeat the six-configuration selection

To reproduce configuration selection rather than fit the already reported choice, run all six configurations against the same fold's initialized backbone:

```bash
for configuration in 1 2 3 4 5 6; do
  mswpfuse train-selected --data-root "$DATA_ROOT" --fold summer \
    --configuration "$configuration" --parent-checkpoint runs/summer/parent.pt \
    --output "runs/summer/configuration_${configuration}.pt"
done

mswpfuse select-fold --fold summer \
  --checkpoints runs/summer/configuration_{1,2,3,4,5,6}.pt \
  --output runs/summer/validation_selected.pt
```

Within each window, checkpoints within `1e-4` of the maximum validation C are eligible. Lower RMSE, lower MAE, earlier epoch, and configuration order resolve ties. The selection command reads only validation checkpoint metadata; it does not load evaluation data.

## Reported module controls

The main article compares Full with three module variants; Supporting Information S4 separately compares Full with Head-only. **Full** retains SRATR plus output-head adaptation, not whole-network fine-tuning.

| Variant | Implementation and training |
| --- | --- |
| `without_history_multiscale` | Single-scale linear history mapping; fit its own backbone and correction stages. |
| `without_source_aware_nwp_fusion` | Availability-masked mean NWP projection and equal branch weights; fit its own backbone and correction stages. |
| w/o SRATR | Evaluate the Full first-stage checkpoint using `predict --backbone`; no additional training. |
| `head_only` | Start from the same Full backbone and update only its original output head, bypassing SRATR and its auxiliary losses. |

For either representation replacement, supply its name to **both** `train-parent --variant` and `train-selected --variant`, with separate checkpoint outputs. Use the same-window initializer and the reported per-window configuration. The matched Head-only and backbone commands are:

```bash
mswpfuse train-selected --data-root "$DATA_ROOT" --fold summer --variant head_only \
  --parent-checkpoint runs/summer/parent.pt --output runs/summer/head_only.pt
mswpfuse predict --data-root "$DATA_ROOT" --fold summer \
  --checkpoint runs/summer/head_only.pt --output results/head_only/summer.npz
mswpfuse predict --data-root "$DATA_ROOT" --fold summer --backbone \
  --checkpoint runs/summer/parent.pt --output results/stage1_only/summer.npz
```

Repeat for the other two windows. All controls retain the applicable training budgets, validation rule, output constraints and recorded-target masks.

## Weather and training-stability summaries

Each prediction directory below contains `summer.npz`, `autumn.npz` and `early_winter.npz`, produced by the current prediction commands. Analysis checks station-date alignment, target masks and checkpoint identities before combining forecasts.

```bash
mswpfuse analyze-weather --data-root "$DATA_ROOT" \
  --full results/full --head-only results/head_only --output results/weather.json
```

This implements Supporting Information S5: training-only tertiles of physical wind speed, daytime POAI and standardized source disagreement, with all 27 Full-minus-Head-only comparisons. Metrics use each station-day's full valid target grid, including PV nights.

For the six-instance stability analysis in S6, run the complete three-stage pipeline separately for each instance and window. Supply your seed with `train-initializer --seed "$SEED"`; the backbone and correction stages inherit and verify it. Use the final per-window configuration and separate outputs. This is distinct from the six-configuration validation comparison above.

```bash
mswpfuse summarize-stability \
  --runs results/instance_a results/instance_b results/instance_c \
         results/instance_d results/instance_e results/instance_f \
  --reference results/nwp_gru --output results/stability.json
```

The fixed reference directory contains predictions from `predict-baseline --model NWPSeq2Seq`. The summary checks the paper's pooled-C criterion for the six supplied final instances and reports mean and sample standard deviation (`ddof=1`) by portfolio, resource and window. Seed values are caller-supplied; no additional seed search or run-selection procedure is introduced. Older prediction archives without stage/seed identity must be regenerated from identity-bearing checkpoints for this analysis.

## Comparison methods

Configurations are in `configs/baselines.yaml`. Some methods require pinned upstream implementations:

```bash
bash scripts/setup_upstreams.sh

mswpfuse train-baseline --data-root "$DATA_ROOT" --fold summer --model TimeMixer \
  --output runs/summer/timemixer.pt
mswpfuse predict-baseline --data-root "$DATA_ROOT" --fold summer --model TimeMixer \
  --checkpoint runs/summer/timemixer.pt --output results/timemixer_summer.npz
```

`--tslib-root` and `--pytorch-forecasting-root` can point to existing checkouts at the pinned commits. The comparison methods retain their source-specific objectives and checkpoint criteria rather than sharing a single training recipe.

The `TimeMixer` option is the tuned reference reported in the paper: width 128, feed-forward width 512, two layers, and **no future-NWP adapter**. It uses power history and calendar inputs. Its configuration was selected on the November 8–December 31 development interval and transferred to the three windows; this overlaps evaluation dates and is not an independently forward-selected reference. Its fitting and validation MAE use the original forward-filled, nonnegative targets; final evaluation retains the common recorded-target mask. These distinctions are preserved rather than relabeling it as the fixed-preset TimeMixer-NWP model.

## Local tests

The regression tests use synthetic inputs and do not distribute or require competition data:

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src python -m pytest tests -q
```

`SHA256SUMS` lists the released code and configuration files. These tests check the implementation and checkpoint interfaces; running them does not retrain or regenerate the paper's results.
