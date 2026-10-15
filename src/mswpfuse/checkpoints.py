"""Model checkpoint persistence and training-lineage identity."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import NWPScaler, fold_dates
from .initializer import HistoryInitializer

CHECKPOINT_VERSION = 2


def _require_new_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {path}")


def checked_seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    return value


def checkpoint_seed(payload: Mapping[str, Any]) -> int | None:
    value = payload.get("seed")
    configured = payload.get("training", {}).get("settings", {}).get("seed")
    if value is not None and configured is not None and checked_seed(value) != checked_seed(configured):
        raise ValueError("Checkpoint seed disagrees with its training settings")
    return None if value is None and configured is None else checked_seed(configured if value is None else value)


def checkpoint_stage_seeds(payload: Mapping[str, Any]) -> dict[str, int]:
    stage = str(payload["stage"])
    seed = checkpoint_seed(payload)
    values = payload.get("stage_seeds", {})
    if not isinstance(values, dict):
        raise ValueError("Checkpoint stage_seeds must be a mapping")
    allowed = {
        "initializer": {"initializer"},
        "parent": {"initializer", "parent"},
        "selected": {"initializer", "parent", "selected"},
    }
    if stage not in allowed or set(values) - allowed[stage]:
        raise ValueError("Checkpoint contains an unknown or future training stage")
    result = {name: checked_seed(value) for name, value in values.items()}
    if seed is None and result:
        raise ValueError("Checkpoint stage seeds lack their current stage seed")
    if seed is not None:
        if any(value != seed for value in result.values()):
            raise ValueError("Checkpoint stages use inconsistent random seeds")
        result[stage] = seed
    return result


def _checkpoint_hash(path: Path) -> str:
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    stage_settings = settings["initializer_training" if stage == "initializer" else "selected_training"]
    seed = checked_seed(stage_settings["seed"])
    stage_seeds = dict(summary.get("stage_seeds", {}))
    stage_seeds[stage] = seed
    checkpoint_stage_seeds({"stage": stage, "seed": seed, "stage_seeds": stage_seeds})
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model": "MSWPFuse",
            "stage": stage,
            "fold": fold,
            "variant": settings.get("variant", "full"),
            "seed": seed,
            "stage_seeds": stage_seeds,
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
    from .ablations import build_backbone, build_selected

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
    checkpoint_stage_seeds(payload)
    scaler = NWPScaler.from_dict(payload["scaler"])
    config = payload["model_config"]
    variant = payload.get("variant", "full")
    if expected_stage == "initializer":
        if variant != "full":
            raise ValueError("History initialization does not have ablation variants")
        model: nn.Module = HistoryInitializer()
    elif expected_stage == "parent":
        model = build_backbone(config, nwp_mean=scaler.mean, nwp_std=scaler.std, variant=variant)
    elif expected_stage == "selected":
        model = build_selected(config, nwp_mean=scaler.mean, nwp_std=scaler.std, variant=variant)
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
