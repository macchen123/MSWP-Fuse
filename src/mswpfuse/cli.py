"""Command-line entry points for the paper's training and prediction analyses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .baselines import baseline_names
from .checkpoints import _require_new_output
from .data import default_data_root, fold_names
from .training_commands import (
    run_baseline_prediction, run_baseline_training, run_initializer, run_parent,
    run_prediction, run_selected, run_selection,
)


def _common_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="Authorized local dataset root; defaults to data/raw_data")
    parser.add_argument("--fold", choices=fold_names(), required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")


def build_parser() -> argparse.ArgumentParser:
    from .ablations import BACKBONE_VARIANTS, SELECTED_VARIANTS
    from .analysis_cli import add_analysis_parsers

    parser = argparse.ArgumentParser(description="MSWP-Fuse training and paper analyses")
    commands = parser.add_subparsers(dest="command", required=True)

    initializer = commands.add_parser("train-initializer", help="Fit the same-fold history initializer")
    _common_data_argument(initializer)
    initializer.add_argument("--seed", type=int, help="User-supplied training seed; defaults to the existing reference setting")
    initializer.add_argument("--output", type=Path, required=True)

    parent = commands.add_parser("train-parent", help="Fit the initialized backbone for 18 epochs")
    _common_data_argument(parent)
    parent.add_argument("--initializer-checkpoint", type=Path, required=True)
    parent.add_argument("--variant", choices=BACKBONE_VARIANTS, default="full")
    parent.add_argument("--seed", type=int, help="Must match the initializer; inherited when omitted")
    parent.add_argument("--output", type=Path, required=True)

    selected = commands.add_parser("train-selected", help="Fit one adaptation configuration and select its validation checkpoint")
    _common_data_argument(selected)
    selected.add_argument("--configuration", type=int, choices=range(1, 7), help="SI configuration number; defaults to the paper's choice for this fold")
    selected.add_argument("--parent-checkpoint", type=Path, required=True)
    selected.add_argument("--variant", choices=SELECTED_VARIANTS, default="full")
    selected.add_argument("--seed", type=int, help="Must match both preceding stages; inherited when omitted")
    selected.add_argument("--output", type=Path, required=True)

    selection = commands.add_parser("select-fold", help="Choose among all six configurations using validation scores only")
    selection.add_argument("--fold", choices=fold_names(), required=True)
    selection.add_argument("--checkpoints", type=Path, nargs=6, required=True)
    selection.add_argument("--output", type=Path, required=True)

    prediction = commands.add_parser("predict", help="Run MSWP-Fuse or its reported control inference")
    _common_data_argument(prediction)
    prediction.add_argument("--split", choices=("train", "validation", "test"), default="test")
    prediction.add_argument("--backbone", action="store_true", help="Evaluate the Full first-stage backbone (w/o SRATR), not a correction checkpoint")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--output", type=Path, required=True)

    baseline_train = commands.add_parser("train-baseline", help="Train one paper comparison model")
    _common_data_argument(baseline_train)
    baseline_train.add_argument("--model", choices=baseline_names(), required=True)
    baseline_train.add_argument("--seed", type=int, default=42)
    baseline_train.add_argument("--output", type=Path, required=True)
    baseline_train.add_argument("--tslib-root", type=Path)
    baseline_train.add_argument("--pytorch-forecasting-root", type=Path)

    baseline_predict = commands.add_parser("predict-baseline", help="Run one paper comparison model")
    _common_data_argument(baseline_predict)
    baseline_predict.add_argument("--model", choices=baseline_names(), required=True)
    baseline_predict.add_argument("--split", choices=("train", "validation", "test"), default="test")
    baseline_predict.add_argument("--checkpoint", type=Path, required=True)
    baseline_predict.add_argument("--output", type=Path, required=True)
    baseline_predict.add_argument("--tslib-root", type=Path)
    baseline_predict.add_argument("--pytorch-forecasting-root", type=Path)
    add_analysis_parsers(commands)
    return parser


def main(argv: list[str] | None = None) -> int:
    from .analysis_cli import run_analysis

    args = build_parser().parse_args(argv)
    _require_new_output(args.output)
    runners = {
        "train-initializer": run_initializer, "train-parent": run_parent,
        "train-selected": run_selected, "select-fold": run_selection,
        "predict": run_prediction, "train-baseline": run_baseline_training,
        "predict-baseline": run_baseline_prediction,
    }
    result = runners[args.command](args) if args.command in runners else run_analysis(args)
    print(json.dumps(result, sort_keys=True, default=str, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
