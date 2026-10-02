"""Command-line entry points for training, validation selection, and inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from .baselines import baseline_names, collect_tree_predictions, load_baseline, save_baseline, train_baseline
from .data import NWPScaler, build_fold_datasets, build_split_dataset, default_data_root, fold_dates, fold_names, load_paper_settings
from .initializer import HistoryInitializer
from .model import AMENDMENT_PREFIXES, MSWPFuse, MSWPFuseBackbone
from .training import (
    collect_predictions,
    seed_everything,
    seed_mswpfuse,
    select_device,
    select_validation_row,
    train_initializer,
    train_parent,
    train_selected,
)

CHECKPOINT_VERSION = 2


def _require_new_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")


def _write_model_checkpoint(
    path: Path,
    stage: str,
    fold: str,
    model: nn.Module,
    scaler: NWPScaler,
    summary: dict[str, object],
    settings: dict[str, Any],
) -> None:
    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": "MSWPFuse",
            "stage": stage,
            "fold": fold,
            "configuration": settings["configuration"] if stage == "selected" else None,
            "model_config": settings["model"] if stage != "initializer" else {},
            "split_dates": {name: [value.isoformat() for value in fold_dates(fold, name)] for name in ("train", "validation", "test")},
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "requires_grad": {name: value.requires_grad for name, value in model.named_parameters()},
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
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
) -> tuple[nn.Module, NWPScaler, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("model") != "MSWPFuse":
        raise ValueError("Checkpoint is not an MSWP-Fuse checkpoint")
    if payload.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("This checkpoint uses the superseded model export; use a checkpoint from the current training commands")
    if payload.get("stage") != expected_stage or payload.get("fold") != expected_fold:
        raise ValueError(f"Checkpoint must be for {expected_stage}/{expected_fold}")
    expected_dates = {name: [value.isoformat() for value in fold_dates(expected_fold, name)] for name in ("train", "validation", "test")}
    if payload.get("split_dates") != expected_dates:
        raise ValueError("Checkpoint split dates differ from the current configuration")
    scaler = NWPScaler.from_dict(payload["scaler"])
    config = payload["model_config"]
    if expected_stage == "initializer":
        model: nn.Module = HistoryInitializer()
    elif expected_stage == "parent":
        model = MSWPFuseBackbone(config, nwp_mean=scaler.mean, nwp_std=scaler.std)
    elif expected_stage == "selected":
        model = MSWPFuse(config, nwp_mean=scaler.mean, nwp_std=scaler.std)
    else:
        raise ValueError(f"Unknown checkpoint stage: {expected_stage}")
    model.load_state_dict(payload["state_dict"], strict=True)
    if expected_stage != "initializer":
        if not np.array_equal(model.nwp_mean.cpu().numpy(), scaler.mean) or not np.array_equal(model.nwp_std.cpu().numpy(), scaler.std):
            raise ValueError("Checkpoint model buffers disagree with its scaler")
    parameters = dict(model.named_parameters())
    policy = payload["requires_grad"]
    if set(policy) != set(parameters) or any(not isinstance(value, bool) for value in policy.values()):
        raise ValueError("Checkpoint trainable-parameter policy is incomplete")
    for name, parameter in parameters.items():
        parameter.requires_grad_(policy[name])
    if device.type == "cuda":
        torch.set_float32_matmul_precision(payload["float32_matmul_precision"])
    model.to(device).eval()
    return model, scaler, payload


def _save_predictions(path: Path, bundle: Any) -> None:
    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bundle.npz_payload())


def _common_data_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-root", type=Path, default=default_data_root(), help="Authorized local dataset root; defaults to data/raw_data")
    parser.add_argument("--fold", choices=fold_names(), required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSWP-Fuse training and inference")
    commands = parser.add_subparsers(dest="command", required=True)

    initializer = commands.add_parser("train-initializer", help="Fit the same-fold history initializer from seed 42")
    _common_data_argument(initializer)
    initializer.add_argument("--output", type=Path, required=True)

    parent = commands.add_parser("train-parent", help="Fit the initialized backbone for 18 epochs")
    _common_data_argument(parent)
    parent.add_argument("--initializer-checkpoint", type=Path, required=True)
    parent.add_argument("--output", type=Path, required=True)

    selected = commands.add_parser("train-selected", help="Fit one adaptation configuration and select its validation checkpoint")
    _common_data_argument(selected)
    selected.add_argument("--configuration", type=int, choices=range(1, 7), help="SI configuration number; defaults to the paper's choice for this fold")
    selected.add_argument("--parent-checkpoint", type=Path, required=True)
    selected.add_argument("--output", type=Path, required=True)

    selection = commands.add_parser("select-fold", help="Choose among all six configurations using validation scores only")
    selection.add_argument("--fold", choices=fold_names(), required=True)
    selection.add_argument("--checkpoints", type=Path, nargs=6, required=True)
    selection.add_argument("--output", type=Path, required=True)

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


def _same_scaler(first: NWPScaler, second: NWPScaler) -> None:
    if not np.array_equal(first.mean, second.mean) or not np.array_equal(first.std, second.std):
        raise ValueError("Checkpoint scaler differs from the current training partition")


def _checkpoint_hash(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_initializer(args: argparse.Namespace) -> dict[str, object]:
    settings = load_paper_settings(args.fold)
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    seed_everything(int(settings["initializer_training"]["seed"]))
    model = HistoryInitializer()
    outcome = train_initializer(model, datasets["train"], datasets["validation"], settings["initializer_training"], device)
    summary = {**outcome.summary(), "settings": settings["initializer_training"]}
    _write_model_checkpoint(args.output, "initializer", args.fold, model, scaler, summary, settings)
    return {"command": args.command, "output": str(args.output), "device": str(device), **summary}


def _run_parent(args: argparse.Namespace) -> dict[str, object]:
    settings = load_paper_settings(args.fold)
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    initializer, initializer_scaler, _ = _load_model_checkpoint(args.initializer_checkpoint, torch.device("cpu"), expected_stage="initializer", expected_fold=args.fold)
    _same_scaler(scaler, initializer_scaler)
    seed_mswpfuse(int(settings["selected_training"]["seed"]), device)
    model = MSWPFuseBackbone(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std, history_state_dict=initializer.state_dict())
    outcome = train_parent(model, datasets["train"], settings["selected_training"], settings["parent_stage"], device)
    summary = {**outcome.summary(), "settings": settings["parent_stage"], "initializer_sha256": _checkpoint_hash(args.initializer_checkpoint)}
    _write_model_checkpoint(args.output, "parent", args.fold, model, scaler, summary, settings)
    return {"command": args.command, "output": str(args.output), "device": str(device), **summary}


def _run_selected(args: argparse.Namespace) -> dict[str, object]:
    settings = load_paper_settings(args.fold, args.configuration)
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    parent, parent_scaler, parent_payload = _load_model_checkpoint(args.parent_checkpoint, torch.device("cpu"), expected_stage="parent", expected_fold=args.fold)
    _same_scaler(scaler, parent_scaler)
    seed_mswpfuse(int(settings["selected_training"]["seed"]), device)
    model = MSWPFuse(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std)
    incompatible = model.load_state_dict(parent.state_dict(), strict=False)
    if incompatible.unexpected_keys or any(not key.startswith(AMENDMENT_PREFIXES) for key in incompatible.missing_keys):
        raise ValueError("Parent checkpoint does not match the adaptation backbone")
    outcome = train_selected(model, datasets["train"], datasets["validation"], settings["selected_training"], device)
    summary = {
        **outcome.summary(), "configuration": settings["configuration"],
        "settings": settings["selected_training"], "parent_sha256": _checkpoint_hash(args.parent_checkpoint),
        "initializer_sha256": parent_payload["training"]["initializer_sha256"],
    }
    _write_model_checkpoint(args.output, "selected", args.fold, model, scaler, summary, settings)
    return {"command": args.command, "output": str(args.output), "device": str(device), **summary}


def _run_selection(args: argparse.Namespace) -> dict[str, object]:
    candidates = []
    seen = set()
    parent_hash = None
    for path in args.checkpoints:
        _, _, payload = _load_model_checkpoint(path, torch.device("cpu"), expected_stage="selected", expected_fold=args.fold)
        configuration = payload["configuration"]
        if configuration not in range(1, 7) or configuration in seen:
            raise ValueError("Supply exactly one checkpoint for each of the six configurations")
        seen.add(configuration)
        summary = payload["training"]
        expected = load_paper_settings(args.fold, configuration)
        if payload["model_config"] != expected["model"] or summary["settings"] != expected["selected_training"]:
            raise ValueError("Checkpoint differs from its declared candidate configuration")
        if parent_hash is None:
            parent_hash = summary["parent_sha256"]
        elif parent_hash != summary["parent_sha256"]:
            raise ValueError("The six configurations must share the same initialized backbone checkpoint")
        candidates.append({"configuration": configuration, "epoch": summary["best_epoch"], "c": summary["best_c"], "rmse": summary["best_rmse"], "mae": summary["best_mae"], "path": path})
    chosen = select_validation_row(candidates)
    payload = torch.load(chosen["path"], map_location="cpu", weights_only=True)
    payload["selection"] = {"criterion": "same-fold validation C within 1e-4 of maximum, then RMSE, MAE, epoch, configuration", "candidates": [{key: value for key, value in row.items() if key != "path"} for row in candidates]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return {"command": args.command, "output": str(args.output), "configuration": chosen["configuration"], "validation_c": chosen["c"]}


def _run_prediction(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    model, scaler, _ = _load_model_checkpoint(args.checkpoint, device, expected_stage="selected", expected_fold=args.fold)
    dataset = build_split_dataset(args.data_root, args.fold, args.split, scaler)
    bundle = collect_predictions(model, dataset, device, batch_size=40)
    _save_predictions(args.output, bundle)
    return {"command": args.command, "output": str(args.output), "device": str(device), "metrics": bundle.metrics()}


def _run_baseline_training(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold, processed_targets=args.model == "TimeMixer")
    run = train_baseline(args.model, datasets["train"], device, args.seed, validation_dataset=datasets["validation"], tslib_root=args.tslib_root, pytorch_forecasting_root=args.pytorch_forecasting_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_baseline(args.output, run, scaler)
    return {"command": args.command, "model": args.model, "output": str(args.output), "device": str(device), "epochs": len(run.history)}


def _run_baseline_prediction(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    model, scaler, kind = load_baseline(args.checkpoint, args.model, device, tslib_root=args.tslib_root, pytorch_forecasting_root=args.pytorch_forecasting_root)
    dataset = build_split_dataset(args.data_root, args.fold, args.split, scaler)
    bundle = collect_tree_predictions(model, dataset) if kind == "tree" else collect_predictions(model, dataset, device)
    _save_predictions(args.output, bundle)
    return {"command": args.command, "model": args.model, "output": str(args.output), "device": str(device), "metrics": bundle.metrics()}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _require_new_output(args.output)
    runners = {
        "train-initializer": _run_initializer, "train-parent": _run_parent,
        "train-selected": _run_selected, "select-fold": _run_selection,
        "predict": _run_prediction, "train-baseline": _run_baseline_training,
        "predict-baseline": _run_baseline_prediction,
    }
    result = runners[args.command](args)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
