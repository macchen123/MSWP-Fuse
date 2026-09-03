"""Public adapters and trainers for every comparison model in the paper."""

from __future__ import annotations

import importlib
import math
import pickle
import subprocess
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset
import yaml

from .data import NWPScaler
from .training import PredictionBundle, collect_predictions, forecast_inputs, move_batch, regular_loader, seed_everything

TSLIB_MODELS = {
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "iTransformer": "iTransformer",
    "TimeMixer": "TimeMixer",
    "TimeXer": "TimeXer",
    "WPMixer": "WPMixer",
    "MultiPatchFormer": "MultiPatchFormer",
}


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_baseline_settings() -> dict[str, Any]:
    value = yaml.safe_load((package_root() / "configs" / "baselines.yaml").read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("models"), dict):
        raise TypeError("baselines.yaml must contain a models mapping")
    return value


def baseline_names() -> tuple[str, ...]:
    return tuple(load_baseline_settings()["models"])


def _configured_model(name: str) -> dict[str, Any]:
    try:
        value = load_baseline_settings()["models"][name]
    except KeyError as error:
        raise KeyError(f"Unknown baseline: {name}") from error
    if not isinstance(value, dict):
        raise TypeError(f"Invalid configuration for {name}")
    return value


def _upstream_root(root: Path | str | None, directory: str, commit: str) -> Path:
    value = Path(root) if root is not None else package_root() / "third_party" / directory
    value = value.expanduser().resolve()
    if not value.is_dir():
        raise FileNotFoundError(f"Required upstream checkout is missing: {value}")
    completed = subprocess.run(
        ["git", "-C", str(value), "rev-parse", "HEAD"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0 or completed.stdout.strip() != commit:
        raise RuntimeError(f"Upstream checkout does not match the required commit: {value}")
    return value


def _tslib_root(root: Path | str | None) -> Path:
    commit = str(load_baseline_settings()["upstreams"]["time_series_library_commit"])
    return _upstream_root(root, "Time-Series-Library", commit)


def _pytorch_forecasting_root(root: Path | str | None) -> Path:
    commit = str(load_baseline_settings()["upstreams"]["pytorch_forecasting_commit"])
    return _upstream_root(root, "pytorch-forecasting", commit)


def _module_from_root(module_name: str, root: Path) -> Any:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module(module_name)
    source = Path(module.__file__ or "").resolve()
    if not source.is_relative_to(root):
        raise RuntimeError(f"{module_name} resolved outside the requested upstream checkout")
    return module


class Persistence(nn.Module):
    uses_nwp = False

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return batch["hist_power"][:, -96:, 0]


class FutureNWPResidual(nn.Module):
    """The common future-NWP residual adapter for history-only upstream models."""

    def __init__(self, hidden_size: int = 64, dropout: float = 0.1, residual_scale: float = 1.0):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.input_projection = nn.Sequential(nn.Linear(30, hidden_size), nn.LayerNorm(hidden_size), nn.GELU())
        self.temporal_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.output_head = nn.Sequential(nn.Linear(hidden_size + 1, hidden_size), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size, 1))
        nn.init.zeros_(self.output_head[-1].weight)
        nn.init.zeros_(self.output_head[-1].bias)

    def forward(self, base_forecast: Tensor, nwp: Tensor, availability: Tensor) -> Tensor:
        if nwp.ndim != 3 or tuple(nwp.shape[1:]) != (96, 27):
            raise ValueError("NWP adapter expects shape (B, 96, 27)")
        if base_forecast.shape != nwp.shape[:2] or availability.shape != (nwp.size(0), 3):
            raise ValueError("NWP adapter inputs do not agree")
        providers = availability.to(nwp.dtype).unsqueeze(1).expand(-1, 96, -1)
        values, _ = self.temporal_encoder(self.input_projection(torch.cat([nwp, providers], dim=-1)))
        correction = self.output_head(torch.cat([values, base_forecast.unsqueeze(-1)], dim=-1)).squeeze(-1)
        return base_forecast + self.residual_scale * correction


class NWPSeq2Seq(nn.Module):
    uses_nwp = True

    def __init__(self, architecture: Mapping[str, Any]):
        super().__init__()
        hidden = int(architecture["hidden"])
        layers = int(architecture["layers"])
        dropout = float(architecture["dropout"]) if layers > 1 else 0.0
        self.history_encoder = nn.GRU(2, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.weather_encoder = nn.GRU(30, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.station_embedding = nn.Embedding(10, hidden)
        self.decoder = nn.GRU(2 * hidden, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(float(architecture["dropout"])), nn.Linear(hidden, 1))

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        _, history_state = self.history_encoder(batch["hist_power"])
        nwp = batch["nwp"]
        providers = batch["nwp_availability"].to(nwp.dtype).unsqueeze(1).expand(-1, 96, -1)
        weather, _ = self.weather_encoder(torch.cat([nwp, providers], dim=-1))
        context = history_state[-1] + self.station_embedding(batch["station_id"].long())
        decoded, _ = self.decoder(torch.cat([weather, context.unsqueeze(1).expand(-1, 96, -1)], dim=-1), history_state)
        return self.head(decoded).squeeze(-1)


class TSLibAdapter(nn.Module):
    uses_nwp = True

    def __init__(
        self,
        name: str,
        architecture: Mapping[str, Any],
        root: Path | str | None,
        device: torch.device,
        batch_size: int,
    ):
        super().__init__()
        if name not in TSLIB_MODELS:
            raise KeyError(name)
        self.name = name
        self.architecture = dict(architecture)
        upstream = _tslib_root(root)
        module = _module_from_root(f"models.{TSLIB_MODELS[name]}", upstream)
        construction_device = torch.device("cpu") if name == "WPMixer" and device.type == "mps" else device
        self.config = _tslib_config(name, self.architecture, construction_device, batch_size)
        self.model = module.Model(self.config)
        adapter = load_baseline_settings()["nwp_adapter"]
        self.nwp_adapter = FutureNWPResidual(
            hidden_size=int(adapter["hidden_size"]),
            dropout=float(adapter["dropout"]),
            residual_scale=float(adapter["residual_scale"]),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        power = batch["hist_power"][:, :, :1]
        history_time = batch["hist_time"]
        future_time = batch["future_time"]
        if self.name == "WPMixer":
            power = power[:, -512:]
            history_time = history_time[:, -512:]
        if self.name == "MultiPatchFormer":
            batch_size = power.size(0)
            encoder = power.reshape(batch_size * 7, 96, 1)
            encoder_marks = history_time.reshape(batch_size * 7, 96, 5)
            future_marks = future_time[:, None].expand(-1, 7, -1, -1).reshape(batch_size * 7, 96, 5)
            output = self._forecast(encoder, encoder_marks, future_marks)
            base = output[:, :, -1].reshape(batch_size, 7, 96).mean(dim=1)
        else:
            encoder = torch.cat([history_time, power], dim=-1) if self.name == "TimeXer" else power
            base = self._forecast(encoder, history_time, future_time)[:, :, -1]
        return self.nwp_adapter(base, batch["nwp"], batch["nwp_availability"])

    def _forecast(self, encoder: Tensor, history_time: Tensor, future_time: Tensor) -> Tensor:
        label_length = int(self.config.label_len)
        tail = history_time[:, -label_length:] if label_length else history_time[:, :0]
        decoder = torch.zeros(
            encoder.size(0),
            label_length + int(self.config.pred_len),
            int(self.config.dec_in),
            dtype=encoder.dtype,
            device=encoder.device,
        )
        output = self.model(encoder, history_time, decoder, torch.cat([tail, future_time], dim=1))
        if isinstance(output, tuple):
            output = output[0]
        if output.ndim != 3 or output.shape[1] != 96:
            raise ValueError(f"Unexpected {self.name} output shape: {tuple(output.shape)}")
        return output


def _tslib_config(
    name: str,
    architecture: Mapping[str, Any],
    device: torch.device,
    batch_size: int,
) -> SimpleNamespace:
    is_timexer = name == "TimeXer"
    values: dict[str, Any] = {
        "task_name": "long_term_forecast",
        "features": "MS" if is_timexer else "S",
        "seq_len": 672,
        "label_len": 48,
        "pred_len": 96,
        "enc_in": 6 if is_timexer else 1,
        "dec_in": 6 if is_timexer else 1,
        "c_out": 1,
        "d_model": 128,
        "n_heads": 4,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 512,
        "moving_avg": 25,
        "factor": 1,
        "dropout": 0.1,
        "embed": "timeF",
        "freq": "t",
        "activation": "gelu",
        "channel_independence": 1,
        "decomp_method": "moving_avg",
        "use_norm": 1,
        "down_sampling_layers": 2,
        "down_sampling_window": 2,
        "down_sampling_method": "avg",
        "top_k": 5,
        "patch_len": 16,
        "stride": 8,
        "batch_size": int(batch_size),
        "device": device,
        "use_amp": False,
    }
    values.update(architecture)
    return SimpleNamespace(**values)


TFT_KNOWN_REALS = tuple([f"time_{index}" for index in range(5)] + [f"nwp_{index}" for index in range(27)] + [f"nwp_available_{index}" for index in range(3)] + ["is_decoder"])


class TemporalFusionTransformerAdapter(nn.Module):
    uses_nwp = True

    def __init__(self, architecture: Mapping[str, Any], root: Path | str | None):
        super().__init__()
        upstream = _pytorch_forecasting_root(root)
        package = _module_from_root("pytorch_forecasting", upstream)
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss

        self.quantiles = tuple(float(value) for value in architecture["quantiles"])
        self.median_index = self.quantiles.index(0.5)
        schema = _tft_schema(str(upstream))
        station_encoder = schema.get_transformer("station")
        resource_encoder = schema.get_transformer("resource")
        self.register_buffer("station_codes", torch.as_tensor(station_encoder.transform([str(index) for index in range(10)]), dtype=torch.long))
        self.register_buffer("resource_codes", torch.as_tensor(resource_encoder.transform(["0", "1"]), dtype=torch.long))
        self.model = TemporalFusionTransformer.from_dataset(
            schema,
            hidden_size=int(architecture["hidden_size"]),
            lstm_layers=int(architecture["lstm_layers"]),
            attention_head_size=int(architecture["attention_head_size"]),
            dropout=float(architecture["dropout"]),
            hidden_continuous_size=int(architecture["hidden_continuous_size"]),
            output_size=int(architecture["output_size"]),
            loss=QuantileLoss(quantiles=list(self.quantiles)),
            learning_rate=0.03,
            log_interval=-1,
            log_val_interval=-1,
            reduce_on_plateau_patience=4,
        )
        package_path = Path(package.__file__ or "").resolve()
        if not package_path.is_relative_to(upstream):
            raise RuntimeError("pytorch_forecasting resolved outside the requested upstream checkout")

    def _quantiles(self, batch: Mapping[str, Tensor]) -> Tensor:
        history = batch["hist_power"][:, :, :1]
        nwp = batch["nwp"]
        history_time = batch["hist_time"]
        future_time = batch["future_time"]
        availability = batch["nwp_availability"].to(nwp.dtype)
        batch_size = history.size(0)
        encoder_known = torch.cat([history_time, torch.zeros(batch_size, 672, 27, device=nwp.device, dtype=nwp.dtype), torch.zeros(batch_size, 672, 3, device=nwp.device, dtype=nwp.dtype), torch.zeros(batch_size, 672, 1, device=nwp.device, dtype=nwp.dtype)], dim=-1)
        decoder_known = torch.cat([future_time, nwp, availability.unsqueeze(1).expand(-1, 96, -1), torch.ones(batch_size, 96, 1, device=nwp.device, dtype=nwp.dtype)], dim=-1)
        station = self.station_codes[batch["station_id"].long()]
        resource = self.resource_codes[batch["resource_id"].long()]
        categoricals = torch.stack([station, resource], dim=-1)
        result = self.model(
            {
                "encoder_cat": categoricals.unsqueeze(1).expand(-1, 672, -1),
                "encoder_cont": torch.cat([encoder_known, history], dim=-1),
                "encoder_target": history.squeeze(-1),
                "encoder_lengths": torch.full((batch_size,), 672, device=nwp.device, dtype=torch.long),
                "decoder_cat": categoricals.unsqueeze(1).expand(-1, 96, -1),
                "decoder_cont": torch.cat([decoder_known, torch.zeros(batch_size, 96, 1, device=nwp.device, dtype=nwp.dtype)], dim=-1),
                "decoder_target": torch.zeros(batch_size, 96, device=nwp.device, dtype=nwp.dtype),
                "decoder_lengths": torch.full((batch_size,), 96, device=nwp.device, dtype=torch.long),
                "decoder_time_idx": torch.arange(96, device=nwp.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1),
                "groups": batch["station_id"].long().unsqueeze(-1),
                "target_scale": torch.tensor([0.0, 1.0], device=nwp.device, dtype=nwp.dtype).unsqueeze(0).expand(batch_size, -1),
            }
        )["prediction"]
        if result.shape != (batch_size, 96, len(self.quantiles)):
            raise ValueError(f"Unexpected TFT output shape: {tuple(result.shape)}")
        return result

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self._quantiles(batch)[..., self.median_index]

    def native_training_loss(self, batch: Mapping[str, Tensor]) -> Tensor:
        output = self._quantiles(batch)
        target = batch["target"]
        mask = batch["target_valid_mask"].bool()
        values = self.model.loss.loss(output, target)
        if values.shape != output.shape or not bool(mask.any()):
            raise ValueError("Invalid TFT training batch")
        return values[mask].mean()


@lru_cache(maxsize=4)
def _tft_schema(root_text: str):
    root = Path(root_text)
    _module_from_root("pytorch_forecasting", root)
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import TorchNormalizer

    steps = np.arange(768, dtype=np.int64)
    base = np.linspace(-1.0, 1.0, 768, dtype=np.float32)
    frames = []
    for station in range(10):
        row: dict[str, object] = {
            "sample_group": f"schema-{station}",
            "station": str(station),
            "resource": str(station % 2),
            "time_idx": steps,
            "target": np.sin(2.0 * np.pi * steps / 96.0).astype(np.float32),
        }
        for index, name in enumerate(TFT_KNOWN_REALS):
            row[name] = base + np.float32(index * 1.0e-3)
        frames.append(pd.DataFrame(row))
    data = pd.concat(frames, ignore_index=True)
    return TimeSeriesDataSet(
        data,
        time_idx="time_idx",
        target="target",
        group_ids=["sample_group"],
        min_encoder_length=672,
        max_encoder_length=672,
        min_prediction_length=96,
        max_prediction_length=96,
        static_categoricals=["station", "resource"],
        time_varying_known_reals=list(TFT_KNOWN_REALS),
        time_varying_unknown_reals=["target"],
        target_normalizer=TorchNormalizer(method="identity", center=False),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
    )


class IndependentHorizonRegressors:
    def __init__(self, estimators: list[Any]):
        self.estimators = estimators

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.column_stack([estimator.predict(features) for estimator in self.estimators]).astype(np.float32)


def _as_numpy(value: object) -> np.ndarray:
    return value.detach().cpu().numpy() if isinstance(value, Tensor) else np.asarray(value)


def tree_arrays(dataset: Dataset[Any]) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | tuple[str, ...]]]:
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    raw_targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    stations: list[int] = []
    resources: list[int] = []
    ordinals: list[int] = []
    keys: list[str] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        history = _as_numpy(sample["hist_power"])
        history_time = _as_numpy(sample["hist_time"])
        future_time = _as_numpy(sample["future_time"])
        nwp = _as_numpy(sample["nwp"])
        availability = _as_numpy(sample["nwp_availability"])
        station = np.zeros(10, dtype=np.float32)
        station[int(_as_numpy(sample["station_id"]))] = 1.0
        resource = np.zeros(2, dtype=np.float32)
        resource[int(_as_numpy(sample["resource_id"]))] = 1.0
        features.append(np.concatenate([history[:, 0], history[:, 1], history_time.reshape(-1), future_time.reshape(-1), nwp.reshape(-1), availability.astype(np.float32), station, resource]).astype(np.float32))
        targets.append(_as_numpy(sample["target"]).astype(np.float32))
        raw_targets.append(_as_numpy(sample["raw_target"]).astype(np.float32))
        masks.append(_as_numpy(sample["target_valid_mask"]).astype(bool))
        stations.append(int(_as_numpy(sample["original_station_id"])))
        resources.append(int(_as_numpy(sample["resource_id"])))
        ordinals.append(int(_as_numpy(sample["target_ordinal"])))
        keys.append(str(sample["sample_key"]))
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), {
        "raw_targets": np.asarray(raw_targets, dtype=np.float32),
        "valid_target_mask": np.asarray(masks, dtype=bool),
        "station_ids": np.asarray(stations, dtype=np.int16),
        "resource_ids": np.asarray(resources, dtype=np.int8),
        "target_ordinals": np.asarray(ordinals, dtype=np.int32),
        "sample_keys": tuple(keys),
    }


def train_tree_baseline(name: str, dataset: Dataset[Any], device: torch.device, seed: int, n_jobs: int) -> IndependentHorizonRegressors:
    features, targets, metadata = tree_arrays(dataset)
    mask = np.asarray(metadata["valid_target_mask"], dtype=bool)
    estimators: list[Any] = []
    for horizon in range(96):
        selected = mask[:, horizon]
        if not bool(selected.any()):
            raise ValueError(f"No valid training target at horizon {horizon + 1}")
        if name == "LightGBM":
            from lightgbm import LGBMRegressor

            estimator = LGBMRegressor(random_state=seed, n_jobs=n_jobs, deterministic=True)
        elif name == "XGBoost":
            from xgboost import XGBRegressor

            estimator = XGBRegressor(random_state=seed, n_jobs=n_jobs, tree_method="hist", device="cuda" if device.type == "cuda" else "cpu")
        else:
            raise KeyError(name)
        estimator.fit(features[selected], targets[selected, horizon])
        estimators.append(estimator)
    return IndependentHorizonRegressors(estimators)


def collect_tree_predictions(model: IndependentHorizonRegressors, dataset: Dataset[Any]) -> PredictionBundle:
    features, targets, metadata = tree_arrays(dataset)
    return PredictionBundle(
        predictions=np.clip(model.predict(features), 0.0, None),
        targets=targets,
        raw_targets=np.asarray(metadata["raw_targets"]),
        valid_target_mask=np.asarray(metadata["valid_target_mask"], dtype=bool),
        station_ids=np.asarray(metadata["station_ids"]),
        resource_ids=np.asarray(metadata["resource_ids"]),
        target_ordinals=np.asarray(metadata["target_ordinals"]),
        sample_keys=tuple(metadata["sample_keys"]),
    )


def build_neural_baseline(
    name: str,
    device: torch.device,
    *,
    tslib_root: Path | str | None = None,
    pytorch_forecasting_root: Path | str | None = None,
) -> nn.Module:
    config = _configured_model(name)
    kind = config["kind"]
    if kind == "persistence":
        model: nn.Module = Persistence()
    elif kind == "native":
        model = NWPSeq2Seq(config["architecture"])
    elif kind == "tslib":
        model = TSLibAdapter(
            name,
            config["architecture"],
            tslib_root,
            device,
            int(config["training"]["batch_size"]),
        )
    elif kind == "tft":
        model = TemporalFusionTransformerAdapter(config["architecture"], pytorch_forecasting_root)
    else:
        raise ValueError(f"{name} is not a neural baseline")
    return model.to(device)


def _adjust_tslib_learning_rate(optimizer: torch.optim.Optimizer, epoch: int, schedule: str, learning_rate: float, root: Path | str | None) -> None:
    upstream = _tslib_root(root)
    module = _module_from_root("utils.tools", upstream)
    adjust = getattr(module, "adjust_learning_rate", None)
    if not callable(adjust):
        raise RuntimeError("Time-Series-Library does not expose adjust_learning_rate")
    adjust(optimizer, epoch, SimpleNamespace(lradj=schedule, learning_rate=learning_rate))


def train_neural_baseline(
    name: str,
    model: nn.Module,
    dataset: Dataset[Any],
    device: torch.device,
    seed: int,
    *,
    tslib_root: Path | str | None = None,
) -> list[dict[str, float | int]]:
    config = _configured_model(name)
    if config["kind"] == "persistence":
        return []
    training = config["training"]
    seed_everything(seed)
    model.to(device)
    optimizer_name = str(training["optimizer"])
    if optimizer_name == "Adam":
        optimizer: torch.optim.Optimizer = Adam(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    elif optimizer_name == "AdamW":
        optimizer = AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]), betas=tuple(float(value) for value in training.get("betas", (0.9, 0.999))))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    schedule = str(training["schedule"])
    epochs = int(training["epochs"])
    warmup = int(training.get("warmup_epochs", 0))
    cosine = CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=float(training.get("minimum_lr", 1.0e-6))) if schedule == "five_epoch_linear_warmup_then_cosine" else None
    use_amp = bool(training.get("use_amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    loader = regular_loader(dataset, int(training["batch_size"]), True, seed)
    history: list[dict[str, float | int]] = []
    native_loss = getattr(model, "native_training_loss", None)
    for epoch in range(1, epochs + 1):
        if schedule == "five_epoch_linear_warmup_then_cosine" and epoch <= warmup:
            for group in optimizer.param_groups:
                group["lr"] = float(training["learning_rate"]) * epoch / max(1, warmup)
        model.train()
        total_loss = 0.0
        batches = 0
        for batch in loader:
            device_batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                if callable(native_loss):
                    loss = native_loss(forecast_inputs(device_batch))
                else:
                    output = model(forecast_inputs(device_batch))
                    target = device_batch["target"]
                    mask = device_batch["target_valid_mask"]
                    if not isinstance(target, Tensor) or not isinstance(mask, Tensor):
                        raise TypeError("Training labels must be tensors")
                    selected = mask.bool()
                    if not bool(selected.any()):
                        raise ValueError("No valid targets are available")
                    loss = (output[selected] - target[selected]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            if scaler is None:
                loss.backward()
                if training.get("gradient_clip") is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                optimizer.step()
            else:
                scaler.scale(loss).backward()
                if training.get("gradient_clip") is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip"]))
                scaler.step(optimizer)
                scaler.update()
            total_loss += float(loss.detach().cpu())
            batches += 1
        history.append({"epoch": epoch, "train_loss": total_loss / max(1, batches), "learning_rate": float(optimizer.param_groups[0]["lr"])})
        if schedule in {"type1", "type3"}:
            _adjust_tslib_learning_rate(optimizer, epoch, schedule, float(training["learning_rate"]), tslib_root)
        elif cosine is not None and epoch > warmup:
            cosine.step()
    return history


@dataclass(frozen=True)
class BaselineRun:
    name: str
    kind: str
    model: nn.Module | IndependentHorizonRegressors
    history: tuple[dict[str, float | int], ...]


def train_baseline(
    name: str,
    dataset: Dataset[Any],
    device: torch.device,
    seed: int,
    *,
    tslib_root: Path | str | None = None,
    pytorch_forecasting_root: Path | str | None = None,
) -> BaselineRun:
    config = _configured_model(name)
    kind = str(config["kind"])
    seed_everything(seed)
    started = time.perf_counter()
    if kind == "tree":
        model = train_tree_baseline(name, dataset, device, seed, int(config["training"]["n_jobs"]))
        history = ({"epoch": 1, "train_seconds": time.perf_counter() - started},)
    else:
        model = build_neural_baseline(name, device, tslib_root=tslib_root, pytorch_forecasting_root=pytorch_forecasting_root)
        history = tuple(train_neural_baseline(name, model, dataset, device, seed, tslib_root=tslib_root))
    return BaselineRun(name=name, kind=kind, model=model, history=history)


def save_baseline(path: Path | str, run: BaselineRun, scaler: NWPScaler) -> None:
    path = Path(path)
    if run.kind == "tree":
        with path.open("wb") as handle:
            pickle.dump({"name": run.name, "kind": run.kind, "model": run.model, "scaler": scaler.to_dict(), "history": list(run.history)}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return
    if not isinstance(run.model, nn.Module):
        raise TypeError("Neural baseline is not a torch module")
    torch.save({"name": run.name, "kind": run.kind, "state_dict": {key: value.detach().cpu() for key, value in run.model.state_dict().items()}, "scaler": scaler.to_dict(), "history": list(run.history)}, path)


def load_baseline(
    path: Path | str,
    name: str,
    device: torch.device,
    *,
    tslib_root: Path | str | None = None,
    pytorch_forecasting_root: Path | str | None = None,
) -> tuple[nn.Module | IndependentHorizonRegressors, NWPScaler, str]:
    config = _configured_model(name)
    path = Path(path)
    if config["kind"] == "tree":
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("name") != name or payload.get("kind") != "tree":
            raise ValueError("Tree file does not match the requested baseline")
        return payload["model"], NWPScaler.from_dict(payload["scaler"]), "tree"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("name") != name or payload.get("kind") != config["kind"]:
        raise ValueError("Checkpoint does not match the requested baseline")
    model = build_neural_baseline(name, device, tslib_root=tslib_root, pytorch_forecasting_root=pytorch_forecasting_root)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return model, NWPScaler.from_dict(payload["scaler"]), str(config["kind"])
