"""Training, evaluation, and checkpoint-selection helpers."""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

from .losses import CompositeForecastLoss, GradientBalance, calibrate_gradient_weights, masked_mean, valid_mask
from .metrics import compute_metrics
from .model import AMENDMENT_PREFIXES, MSWPFuse


@dataclass(frozen=True)
class PredictionBundle:
    predictions: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    valid_target_mask: np.ndarray
    station_ids: np.ndarray
    resource_ids: np.ndarray
    target_ordinals: np.ndarray
    sample_keys: tuple[str, ...]

    def metrics(self) -> dict[str, object]:
        return compute_metrics(
            self.predictions,
            self.raw_targets,
            self.valid_target_mask,
            self.station_ids,
            self.resource_ids,
            self.target_ordinals,
            self.sample_keys,
        )

    def npz_payload(self) -> dict[str, np.ndarray]:
        return {
            "predictions": self.predictions.astype(np.float32),
            "targets": self.targets.astype(np.float32),
            "raw_targets": self.raw_targets.astype(np.float32),
            "valid_target_mask": self.valid_target_mask.astype(bool),
            "station_ids": self.station_ids.astype(np.int16),
            "resource_ids": self.resource_ids.astype(np.int8),
            "target_ordinals": self.target_ordinals.astype(np.int32),
            "sample_keys": np.asarray(self.sample_keys),
        }


@dataclass(frozen=True)
class TrainingOutcome:
    best_epoch: int
    best_c: float | None
    best_mae: float | None
    best_rmse: float | None
    elapsed_seconds: float
    history: tuple[dict[str, float | int], ...]
    state_dict: dict[str, Tensor]
    gradient_balance: GradientBalance | None = None

    def summary(self) -> dict[str, object]:
        balance = None
        if self.gradient_balance is not None:
            balance = {
                "weights": dict(self.gradient_balance.weights),
                "gradient_rms": dict(self.gradient_balance.gradient_rms),
            }
        return {
            "best_epoch": self.best_epoch,
            "best_c": self.best_c,
            "best_mae": self.best_mae,
            "best_rmse": self.best_rmse,
            "elapsed_seconds": self.elapsed_seconds,
            "history": list(self.history),
            "gradient_balance": balance,
        }


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        name: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def forecast_inputs(batch: Mapping[str, object]) -> dict[str, Tensor]:
    names = ("hist_power", "hist_time", "future_time", "nwp", "nwp_availability", "station_id", "resource_id")
    values = {name: batch[name] for name in names}
    if not all(isinstance(value, Tensor) for value in values.values()):
        raise TypeError("Model inputs must be tensors")
    return values  # type: ignore[return-value]


class CompleteDateBatchSampler(Sampler[list[int]]):
    """Keeps all ten stations together for each target date."""

    def __init__(self, dataset: Dataset[Any], dates_per_batch: int, seed: int, shuffle: bool):
        refs = getattr(dataset, "refs", None)
        if refs is None:
            raise TypeError("Complete-date batches require a dataset with refs")
        if dates_per_batch <= 0:
            raise ValueError("dates_per_batch must be positive")
        by_date: dict[int, dict[int, int]] = defaultdict(dict)
        for index, ref in enumerate(refs):
            ordinal = ref.target_date.toordinal()
            station = int(ref.station_id)
            if station in by_date[ordinal]:
                raise ValueError(f"Duplicate station-date row: {station}/{ordinal}")
            by_date[ordinal][station] = index
        expected = set(range(1, 11))
        self.groups: tuple[tuple[int, ...], ...] = tuple(
            tuple(members[station] for station in range(1, 11))
            for _, members in sorted(by_date.items())
            if _check_complete_date(members, expected)
        )
        if not self.groups:
            raise ValueError("No complete dates are available")
        self.dates_per_batch = int(dates_per_batch)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        order = list(range(len(self.groups)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(order)
        for start in range(0, len(order), self.dates_per_batch):
            yield [index for group in order[start : start + self.dates_per_batch] for index in self.groups[group]]

    def __len__(self) -> int:
        return math.ceil(len(self.groups) / self.dates_per_batch)


def _check_complete_date(members: Mapping[int, int], expected: set[int]) -> bool:
    observed = set(members)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Incomplete target date; missing={missing}, extra={extra}")
    return True


def complete_date_loader(dataset: Dataset[Any], dates_per_batch: int, seed: int, shuffle: bool) -> tuple[DataLoader[Any], CompleteDateBatchSampler]:
    sampler = CompleteDateBatchSampler(dataset, dates_per_batch, seed, shuffle)
    return DataLoader(dataset, batch_sampler=sampler, num_workers=0), sampler


def regular_loader(dataset: Dataset[Any], batch_size: int, shuffle: bool, seed: int) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def cpu_state_dict(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().cpu().contiguous().clone() for name, value in model.state_dict().items()}


def collect_predictions(
    model: nn.Module,
    dataset: Dataset[Any],
    device: torch.device,
    *,
    batch_size: int = 64,
    parent_only: bool = False,
) -> PredictionBundle:
    loader = regular_loader(dataset, batch_size, False, 0)
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    stations: list[np.ndarray] = []
    resources: list[np.ndarray] = []
    ordinals: list[np.ndarray] = []
    sample_keys: list[str] = []
    with torch.no_grad():
        for batch in loader:
            device_batch = move_batch(batch, device)
            inputs = forecast_inputs(device_batch)
            if parent_only:
                if not isinstance(model, MSWPFuse):
                    raise TypeError("parent_only evaluation requires MSWPFuse")
                output = model.forward_parent(inputs)
            else:
                output = model(inputs)
            if not isinstance(output, Tensor) or output.ndim != 2 or output.size(1) != 96:
                raise ValueError("Model output must have shape (B, 96)")
            predictions.append(torch.clamp_min(output, 0.0).float().cpu().numpy())
            targets.append(batch["target"].detach().cpu().numpy())
            raw_targets.append(batch["raw_target"].detach().cpu().numpy())
            masks.append(batch["target_valid_mask"].detach().cpu().numpy().astype(bool))
            stations.append(batch["original_station_id"].detach().cpu().numpy())
            resources.append(batch["resource_id"].detach().cpu().numpy())
            ordinals.append(batch["target_ordinal"].detach().cpu().numpy())
            sample_keys.extend(str(value) for value in batch["sample_key"])
    return PredictionBundle(
        predictions=np.concatenate(predictions),
        targets=np.concatenate(targets),
        raw_targets=np.concatenate(raw_targets),
        valid_target_mask=np.concatenate(masks),
        station_ids=np.concatenate(stations).reshape(-1),
        resource_ids=np.concatenate(resources).reshape(-1),
        target_ordinals=np.concatenate(ordinals).reshape(-1),
        sample_keys=tuple(sample_keys),
    )


def _stack_training_tensors(dataset: Dataset[Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    target: list[Tensor] = []
    mask: list[Tensor] = []
    stations: list[Tensor] = []
    history: list[Tensor] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target.append(sample["target"])
        mask.append(sample["target_valid_mask"])
        stations.append(sample["original_station_id"])
        history.append(sample["hist_power"])
    if not target:
        raise ValueError("Training data is empty")
    return torch.stack(target), torch.stack(mask), torch.stack(stations), torch.stack(history)


def gradient_balance(
    dataset: Dataset[Any],
    shares: Mapping[str, Any],
) -> GradientBalance:
    target, mask, stations, history = _stack_training_tensors(dataset)
    return calibrate_gradient_weights(
        target.double(),
        mask.bool(),
        stations,
        history[:, -96:, 0].double(),
        {
            "station_macro_c": float(shares["station_macro_c"]),
            "charbonnier_mae": float(shares["charbonnier_mae"]),
            "pooled_rmse": float(shares["pooled_rmse"]),
        },
    )


def selected_gradient_balance(dataset: Dataset[Any], settings: Mapping[str, Any]) -> GradientBalance:
    return gradient_balance(dataset, settings["loss_shares"])


def _set_learning_rates(optimizer: torch.optim.Optimizer, epoch: int, max_epochs: int, warmup_epochs: int) -> None:
    if epoch <= warmup_epochs:
        fraction = epoch / max(1, warmup_epochs)
        for group in optimizer.param_groups:
            group["lr"] = float(group["base_lr"]) * fraction
        return
    denominator = max(1, max_epochs - warmup_epochs - 1)
    progress = min(1.0, (epoch - warmup_epochs - 1) / denominator)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        base = float(group["base_lr"])
        minimum = float(group["minimum_lr"])
        group["lr"] = minimum + (base - minimum) * cosine


def configure_selected_scope(model: MSWPFuse, settings: Mapping[str, Any]) -> list[dict[str, object]]:
    if settings["trainable_scope"] != "amendment_plus_head":
        raise ValueError("The public selected stage uses amendment_plus_head")
    amendment: list[Tensor] = []
    head: list[Tensor] = []
    for name, parameter in model.named_parameters():
        if name.startswith(AMENDMENT_PREFIXES):
            parameter.requires_grad_(True)
            amendment.append(parameter)
        elif name.startswith("head."):
            parameter.requires_grad_(True)
            head.append(parameter)
        else:
            parameter.requires_grad_(False)
    if not amendment or not head:
        raise RuntimeError("Selected training scope is empty")
    learning_rate = float(settings["learning_rate"])
    multiplier = float(settings["parent_lr_multiplier"])
    minimum = float(settings["minimum_lr"])
    return [
        {"params": amendment, "lr": learning_rate, "base_lr": learning_rate, "minimum_lr": minimum, "weight_decay": float(settings["weight_decay"])},
        {"params": head, "lr": learning_rate * multiplier, "base_lr": learning_rate * multiplier, "minimum_lr": minimum * multiplier, "weight_decay": float(settings["weight_decay"])},
    ]


def _selected_loss(
    model: MSWPFuse,
    states: Mapping[str, Tensor],
    batch: Mapping[str, object],
    composite: CompositeForecastLoss,
    settings: Mapping[str, Any],
) -> Tensor:
    prediction = states["prediction"]
    target = batch["target"]
    target_mask = batch["target_valid_mask"]
    station_ids = batch["original_station_id"]
    if not all(isinstance(value, Tensor) for value in (target, target_mask, station_ids)):
        raise TypeError("Training labels must be tensors")
    target = target.float()
    mask = valid_mask(target, target_mask.bool())
    composite_value = composite(prediction.float(), target, station_ids, mask, model.station_delta_penalty())
    inactive_logit = states["inactive_logit"]
    inactive_target = (target.detach() <= float(settings["activity_threshold"])).to(inactive_logit.dtype)
    bce = masked_mean(nn.functional.binary_cross_entropy_with_logits(inactive_logit, inactive_target, reduction="none"), mask)
    return (
        composite_value
        + float(settings["inactive_bce"]) * bce
        + float(settings["suppression_budget"]) * states["inactive_probability"].float().mean()
        + float(settings["correction_budget"]) * states["correction_gate"].float().mean()
    )


def _is_better(candidate: Mapping[str, float], incumbent: Mapping[str, float] | None, tolerance: float) -> bool:
    if incumbent is None:
        return True
    if candidate["c"] > incumbent["c"] + tolerance:
        return True
    if candidate["c"] < incumbent["c"] - tolerance:
        return False
    return (candidate["rmse"], candidate["mae"], candidate["epoch"]) < (incumbent["rmse"], incumbent["mae"], incumbent["epoch"])


def train_selected(
    model: MSWPFuse,
    train_dataset: Dataset[Any],
    validation_dataset: Dataset[Any],
    settings: Mapping[str, Any],
    device: torch.device,
) -> TrainingOutcome:
    """Train the selected amendment-plus-head configuration."""

    seed = int(settings["seed"])
    seed_everything(seed)
    model.to(device)
    balance = selected_gradient_balance(train_dataset, settings)
    composite = CompositeForecastLoss(balance.weights).to(device)
    groups = configure_selected_scope(model, settings)
    optimizer = AdamW(groups, betas=(0.9, 0.95), eps=1.0e-8)
    loader, sampler = complete_date_loader(train_dataset, int(settings["dates_per_batch"]), seed, True)
    max_epochs = int(settings["max_epochs"])
    patience = int(settings["patience"])
    tolerance = float(settings["checkpoint_c_tolerance"])
    best: dict[str, float] | None = None
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.set_source_dropout(0.0 if epoch < int(settings["source_dropout_start_epoch"]) else float(settings["source_dropout_probability"]))
        _set_learning_rates(optimizer, epoch, max_epochs, int(settings["warmup_epochs"]))
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            device_batch = move_batch(batch, device)
            states = model.forward_features(forecast_inputs(device_batch))
            loss = _selected_loss(model, states, device_batch, composite, settings)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite selected loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [value for value in model.parameters() if value.requires_grad]
            nn.utils.clip_grad_norm_(trainable, float(settings["gradient_clip"]), error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        validation = collect_predictions(model, validation_dataset, device)
        metrics = validation.metrics()
        if metrics["mae"] is None or metrics["rmse"] is None:
            raise FloatingPointError("Validation contains non-finite predictions")
        candidate = {"epoch": float(epoch), "c": float(metrics["c"]), "mae": float(metrics["mae"]), "rmse": float(metrics["rmse"])}
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, batches), "c": candidate["c"], "mae": candidate["mae"], "rmse": candidate["rmse"], "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if _is_better(candidate, best, tolerance):
            best = candidate
            best_state = cpu_state_dict(model)
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best is None or best_state is None:
        raise RuntimeError("Selected training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    return TrainingOutcome(
        best_epoch=int(best["epoch"]),
        best_c=best["c"],
        best_mae=best["mae"],
        best_rmse=best["rmse"],
        elapsed_seconds=time.perf_counter() - started,
        history=tuple(history),
        state_dict=best_state,
        gradient_balance=balance,
    )


def train_parent(
    model: MSWPFuse,
    train_dataset: Dataset[Any],
    selected_settings: Mapping[str, Any],
    parent_settings: Mapping[str, Any],
    device: torch.device,
) -> TrainingOutcome:
    """Train the public parent path before the selected amendment stage."""

    seed = int(selected_settings["seed"])
    seed_everything(seed)
    model.to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith(AMENDMENT_PREFIXES))
    balance = gradient_balance(train_dataset, parent_settings["loss_shares"])
    composite = CompositeForecastLoss(balance.weights).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    learning_rate = float(parent_settings["learning_rate"])
    minimum_lr = float(parent_settings["minimum_lr"])
    optimizer = AdamW(
        [{"params": parameters, "lr": learning_rate, "base_lr": learning_rate, "minimum_lr": minimum_lr, "weight_decay": float(parent_settings["weight_decay"])}],
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )
    loader, sampler = complete_date_loader(train_dataset, int(parent_settings["dates_per_batch"]), seed, True)
    max_epochs = int(parent_settings["epochs"])
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        sampler.set_epoch(epoch)
        model.set_source_dropout(0.0 if epoch < int(parent_settings["source_dropout_start_epoch"]) else float(parent_settings["source_dropout_probability"]))
        _set_learning_rates(optimizer, epoch, max_epochs, int(parent_settings["warmup_epochs"]))
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            device_batch = move_batch(batch, device)
            prediction = model.forward_parent(forecast_inputs(device_batch))
            target = device_batch["target"]
            mask = device_batch["target_valid_mask"]
            station_ids = device_batch["original_station_id"]
            if not all(isinstance(value, Tensor) for value in (target, mask, station_ids)):
                raise TypeError("Training labels must be tensors")
            loss = composite(prediction, target.float(), station_ids, mask.bool(), model.station_delta_penalty())
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite parent loss at epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, float(parent_settings["gradient_clip"]), error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, batches), "learning_rate": float(optimizer.param_groups[0]["lr"])})
    state = cpu_state_dict(model)
    return TrainingOutcome(
        best_epoch=max_epochs,
        best_c=None,
        best_mae=None,
        best_rmse=None,
        elapsed_seconds=time.perf_counter() - started,
        history=tuple(history),
        state_dict=state,
        gradient_balance=balance,
    )
