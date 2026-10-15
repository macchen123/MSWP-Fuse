"""Application operations for the paper's training and prediction commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import collect_tree_predictions, load_baseline, save_baseline, train_baseline
from .checkpoints import (
    _checkpoint_hash, _load_model_checkpoint, _require_new_output,
    _write_model_checkpoint, checked_seed, checkpoint_seed, checkpoint_stage_seeds,
)
from .data import NWPScaler, build_fold_datasets, build_split_dataset, load_paper_settings
from .initializer import HistoryInitializer
from .model import AMENDMENT_PREFIXES
from .training import (
    collect_predictions, seed_everything, seed_mswpfuse, select_device,
    select_validation_row, train_initializer, train_parent, train_selected,
)


def _same_scaler(first: NWPScaler, second: NWPScaler) -> None:
    if not np.array_equal(first.mean, second.mean) or not np.array_equal(first.std, second.std):
        raise ValueError("Checkpoint scaler differs from the current training partition")


def _training_seed(requested: int | None, parent: dict[str, Any], required_stages: set[str]) -> int:
    inherited = checkpoint_seed(parent)
    lineage = checkpoint_stage_seeds(parent)
    if inherited is None or not required_stages.issubset(lineage):
        raise ValueError("Checkpoint lacks complete stage seed identity; rebuild it with the current training commands")
    if requested is not None and checked_seed(requested) != inherited:
        raise ValueError("Use the same seed for the initializer, backbone, and correction stages")
    return inherited


def _apply_seed(settings: dict[str, Any], seed: int) -> None:
    settings["initializer_training"]["seed"] = seed
    settings["selected_training"]["seed"] = seed


def _check_reported_training(settings: dict[str, Any], fold: str, variant: str, seed: int) -> None:
    reference = load_paper_settings(fold)
    if variant != "full":
        if seed != reference["selected_training"]["seed"] or settings["configuration"] != reference["configuration"]:
            raise ValueError("Reported module controls use the fixed reference seed and final per-window configuration")
    elif seed != reference["selected_training"]["seed"] and settings["configuration"] != reference["configuration"]:
        raise ValueError("Multi-seed training uses the final per-window configuration, not another configuration search")


def _save_predictions(path: Path, bundle: Any, metadata: dict[str, Any]) -> None:
    _require_new_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **bundle.npz_payload(), metadata=np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)))


def run_initializer(args: argparse.Namespace) -> dict[str, object]:
    settings = load_paper_settings(args.fold)
    seed = checked_seed(settings["initializer_training"]["seed"] if args.seed is None else args.seed)
    _apply_seed(settings, seed)
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    seed_everything(seed)
    model = HistoryInitializer()
    outcome = train_initializer(model, datasets["train"], datasets["validation"], settings["initializer_training"], device)
    summary = {**outcome.summary(), "settings": settings["initializer_training"], "stage_seeds": {"initializer": seed}}
    _write_model_checkpoint(args.output, "initializer", args.fold, model, scaler, summary, settings)
    return {"command": args.command, "output": str(args.output), "device": str(device), "seed": seed, **summary}


def run_parent(args: argparse.Namespace) -> dict[str, object]:
    from .ablations import build_backbone

    settings = load_paper_settings(args.fold)
    device = select_device(args.device)
    initializer, initializer_scaler, initializer_payload = _load_model_checkpoint(
        args.initializer_checkpoint, torch.device("cpu"), expected_stage="initializer", expected_fold=args.fold,
    )
    seed = _training_seed(args.seed, initializer_payload, {"initializer"})
    _check_reported_training(settings, args.fold, args.variant, seed)
    _apply_seed(settings, seed)
    settings["variant"] = args.variant
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    _same_scaler(scaler, initializer_scaler)
    seed_mswpfuse(seed, device)
    model = build_backbone(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std,
                           history_state_dict=initializer.state_dict(), variant=args.variant)
    outcome = train_parent(model, datasets["train"], settings["selected_training"], settings["parent_stage"], device)
    summary = {
        **outcome.summary(), "settings": settings["parent_stage"],
        "initializer_sha256": _checkpoint_hash(args.initializer_checkpoint),
        "stage_seeds": {**checkpoint_stage_seeds(initializer_payload), "parent": seed},
    }
    _write_model_checkpoint(args.output, "parent", args.fold, model, scaler, summary, settings)
    return {"command": args.command, "output": str(args.output), "device": str(device), "variant": args.variant, "seed": seed, **summary}


def run_selected(args: argparse.Namespace) -> dict[str, object]:
    from .ablations import build_selected

    settings = load_paper_settings(args.fold, args.configuration)
    device = select_device(args.device)
    parent, parent_scaler, parent_payload = _load_model_checkpoint(
        args.parent_checkpoint, torch.device("cpu"), expected_stage="parent", expected_fold=args.fold,
    )
    expected_parent = "full" if args.variant == "head_only" else args.variant
    if parent_payload.get("variant", "full") != expected_parent:
        raise ValueError("The correction stage must consume its own variant's backbone; Head-only shares the Full backbone")
    seed = _training_seed(args.seed, parent_payload, {"initializer", "parent"})
    _check_reported_training(settings, args.fold, args.variant, seed)
    _apply_seed(settings, seed)
    settings["variant"] = args.variant
    if args.variant == "head_only":
        settings["selected_training"]["trainable_scope"] = "head_only"
    datasets, scaler = build_fold_datasets(args.data_root, args.fold)
    _same_scaler(scaler, parent_scaler)
    seed_mswpfuse(seed, device)
    model = build_selected(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std, variant=args.variant)
    incompatible = model.load_state_dict(parent.state_dict(), strict=False)
    if incompatible.unexpected_keys or any(not key.startswith(AMENDMENT_PREFIXES) for key in incompatible.missing_keys):
        raise ValueError("Parent checkpoint does not match the adaptation backbone")
    outcome = train_selected(model, datasets["train"], datasets["validation"], settings["selected_training"], device)
    summary = {
        **outcome.summary(), "configuration": settings["configuration"],
        "settings": settings["selected_training"], "parent_sha256": _checkpoint_hash(args.parent_checkpoint),
        "initializer_sha256": parent_payload["training"]["initializer_sha256"],
        "stage_seeds": {**checkpoint_stage_seeds(parent_payload), "selected": seed},
    }
    _write_model_checkpoint(args.output, "selected", args.fold, model, scaler, summary, settings)
    return {"command": args.command, "output": str(args.output), "device": str(device), "variant": args.variant, "seed": seed, **summary}


def run_selection(args: argparse.Namespace) -> dict[str, object]:
    candidates = []
    seen = set()
    parent_hash = None
    for path in args.checkpoints:
        _, _, payload = _load_model_checkpoint(path, torch.device("cpu"), expected_stage="selected", expected_fold=args.fold)
        if payload.get("variant", "full") != "full":
            raise ValueError("The six-configuration comparison applies to the Full model only")
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
    _require_new_output(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    return {"command": args.command, "output": str(args.output), "configuration": chosen["configuration"], "validation_c": chosen["c"]}


def run_prediction(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    stage = "parent" if args.backbone else "selected"
    model, scaler, payload = _load_model_checkpoint(args.checkpoint, device, expected_stage=stage, expected_fold=args.fold)
    variant = payload.get("variant", "full")
    if args.backbone:
        reference_seed = load_paper_settings(args.fold)["selected_training"]["seed"]
        if variant != "full" or checkpoint_seed(payload) != reference_seed:
            raise ValueError("The reported w/o SRATR prediction uses the fixed-reference Full first-stage backbone")
    dataset = build_split_dataset(args.data_root, args.fold, args.split, scaler)
    bundle = collect_predictions(model, dataset, device, batch_size=40)
    training = payload["training"]
    metadata = {
        "model": "MSWP-Fuse", "fold": args.fold, "split": args.split,
        "variant": "stage1_only" if args.backbone else variant,
        "seed": checkpoint_seed(payload), "stage_seeds": checkpoint_stage_seeds(payload),
        "configuration": payload.get("configuration"),
        "initializer_sha256": training.get("initializer_sha256"),
        "parent_sha256": _checkpoint_hash(args.checkpoint) if args.backbone else training.get("parent_sha256"),
        "checkpoint_sha256": _checkpoint_hash(args.checkpoint),
    }
    _save_predictions(args.output, bundle, metadata)
    return {"command": args.command, "output": str(args.output), "device": str(device), "variant": metadata["variant"], "metrics": bundle.metrics()}


def run_baseline_training(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    datasets, scaler = build_fold_datasets(args.data_root, args.fold, processed_targets=args.model == "TimeMixer")
    run = train_baseline(args.model, datasets["train"], device, args.seed, validation_dataset=datasets["validation"], tslib_root=args.tslib_root, pytorch_forecasting_root=args.pytorch_forecasting_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_baseline(args.output, run, scaler)
    return {"command": args.command, "model": args.model, "output": str(args.output), "device": str(device), "epochs": len(run.history)}


def run_baseline_prediction(args: argparse.Namespace) -> dict[str, object]:
    device = select_device(args.device)
    model, scaler, kind = load_baseline(args.checkpoint, args.model, device, tslib_root=args.tslib_root, pytorch_forecasting_root=args.pytorch_forecasting_root)
    dataset = build_split_dataset(args.data_root, args.fold, args.split, scaler)
    bundle = collect_tree_predictions(model, dataset) if kind == "tree" else collect_predictions(model, dataset, device)
    _save_predictions(args.output, bundle, {
        "model": args.model, "fold": args.fold, "split": args.split,
        "checkpoint_sha256": _checkpoint_hash(args.checkpoint),
    })
    return {"command": args.command, "model": args.model, "output": str(args.output), "device": str(device), "metrics": bundle.metrics()}
