"""Command-line entry points for training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import (
    baseline_names,
    collect_tree_predictions,
    load_baseline,
    save_baseline,
    train_baseline,
)
from .data import NWPScaler, build_fold_datasets, build_split_dataset, default_data_root, fold_names, load_paper_settings
from .model import MSWPFuse
from .training import collect_predictions, seed_everything, select_device, train_parent, train_selected


def _write_model_checkpoint(
    path: Path,
    stage: str,
    fold: str,
    model: MSWPFuse,
    scaler: NWPScaler,
    summary: dict[str, object],
) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": "MSWPFuse",
            "stage": stage,
            "fold": fold,
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "scaler": scaler.to_dict(),
            "training": summary,
        },
        path,
    )


def _load_model_checkpoint(
    path: Path,
    device: torch.device,
    *,
    expected_stage: str,
    expected_fold: str,
) -> tuple[MSWPFuse, NWPScaler, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("model") != "MSWPFuse":
        raise ValueError("Checkpoint is not an MSWP-Fuse checkpoint")
    if payload.get("stage") != expected_stage:
        raise ValueError(f"Checkpoint stage is not {expected_stage!r}")
    if payload.get("fold") != expected_fold:
        raise ValueError(f"Checkpoint fold is not {expected_fold!r}")
    state = payload.get("state_dict")
    if not isinstance(state, dict) or "nwp_mean" not in state or "nwp_std" not in state:
        raise ValueError("Checkpoint lacks NWP scaler buffers")
    settings = load_paper_settings()
    model = MSWPFuse(settings["model"], nwp_mean=state["nwp_mean"], nwp_std=state["nwp_std"])
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    scaler_value = payload.get("scaler")
    scaler = NWPScaler.from_dict(scaler_value) if isinstance(scaler_value, dict) else NWPScaler(
        mean=state["nwp_mean"].detach().cpu().numpy().astype(np.float32),
        std=state["nwp_std"].detach().cpu().numpy().astype(np.float32),
    )
    return model, scaler, payload


def _save_predictions(path: Path, bundle: Any) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle.npz_payload())


def _common_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="Dataset root; defaults to data/raw_data")
    parser.add_argument("--fold", choices=fold_names(), required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSWP-Fuse training and inference")
    commands = parser.add_subparsers(dest="command", required=True)

    parent = commands.add_parser("train-parent", help="Train the public parent path")
    _common_data_argument(parent)
    parent.add_argument("--output", type=Path, required=True)

    selected = commands.add_parser("train-selected", help="Train the selected residual-and-head stage")
    _common_data_argument(selected)
    selected.add_argument("--output", type=Path, required=True)
    selected.add_argument("--parent-checkpoint", type=Path, help="Optional output from train-parent")

    prediction = commands.add_parser("predict", help="Run MSWP-Fuse inference")
    _common_data_argument(prediction)
    prediction.add_argument("--split", choices=("train", "validation", "test"), default="test")
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
    return parser


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    settings = load_paper_settings()
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    seed_everything(int(settings["selected_training"]["seed"]))
    model = MSWPFuse(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std)
    outcome = train_parent(model, datasets["train"], settings["selected_training"], settings["parent_stage"], device)
    _write_model_checkpoint(args.output, "parent", args.fold, model, scaler, outcome.summary())
    return {"command": "train-parent", "output": str(args.output), "device": str(device), **outcome.summary()}


def _run_selected(args: argparse.Namespace) -> dict[str, object]:
    settings = load_paper_settings()
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    seed_everything(int(settings["selected_training"]["seed"]))
    model = MSWPFuse(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std)
    parent_summary: object
    if args.parent_checkpoint is not None:
        parent, parent_scaler, parent_payload = _load_model_checkpoint(
            args.parent_checkpoint,
            device,
            expected_stage="parent",
            expected_fold=args.fold,
        )
        if not np.array_equal(parent_scaler.mean, scaler.mean) or not np.array_equal(parent_scaler.std, scaler.std):
            raise ValueError("Parent checkpoint scaler differs from the current training split")
        model.load_state_dict(parent.state_dict(), strict=True)
        initialization = "parent-checkpoint"
        parent_summary = parent_payload.get("training")
    else:
        parent_outcome = train_parent(
            model,
            datasets["train"],
            settings["selected_training"],
            settings["parent_stage"],
            device,
        )
        initialization = "freshly-trained-parent"
        parent_summary = parent_outcome.summary()
    outcome = train_selected(model, datasets["train"], datasets["validation"], settings["selected_training"], device)
    summary = {
        **outcome.summary(),
        "initialization": initialization,
        "parent_training": parent_summary,
    }
    _write_model_checkpoint(args.output, "selected", args.fold, model, scaler, summary)
    return {"command": "train-selected", "output": str(args.output), "device": str(device), **summary}


def _run_prediction(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    model, scaler, _ = _load_model_checkpoint(
        args.checkpoint,
        device,
        expected_stage="selected",
        expected_fold=args.fold,
    )
    dataset = build_split_dataset(args.data_root, args.fold, args.split, scaler)
    bundle = collect_predictions(model, dataset, device)
    _save_predictions(args.output, bundle)
    return {"command": "predict", "output": str(args.output), "device": str(device), "metrics": bundle.metrics()}


def _run_baseline_training(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    run = train_baseline(
        args.model,
        datasets["train"],
        device,
        args.seed,
        tslib_root=args.tslib_root,
        pytorch_forecasting_root=args.pytorch_forecasting_root,
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_baseline(args.output, run, scaler)
    return {"command": "train-baseline", "model": args.model, "output": str(args.output), "device": str(device), "epochs": len(run.history)}


def _run_baseline_prediction(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    model, scaler, kind = load_baseline(
        args.checkpoint,
        args.model,
        device,
        tslib_root=args.tslib_root,
        pytorch_forecasting_root=args.pytorch_forecasting_root,
    )
    dataset = build_split_dataset(args.data_root, args.fold, args.split, scaler)
    bundle = collect_tree_predictions(model, dataset) if kind == "tree" else collect_predictions(model, dataset, device)
    _save_predictions(args.output, bundle)
    return {"command": "predict-baseline", "model": args.model, "output": str(args.output), "device": str(device), "metrics": bundle.metrics()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train-parent":
        result = _run_parent(args)
    elif args.command == "train-selected":
        result = _run_selected(args)
    elif args.command == "predict":
        result = _run_prediction(args)
    elif args.command == "train-baseline":
        result = _run_baseline_training(args)
    elif args.command == "predict-baseline":
        result = _run_baseline_prediction(args)
    else:
        raise RuntimeError(f"Unknown command: {args.command}")
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
