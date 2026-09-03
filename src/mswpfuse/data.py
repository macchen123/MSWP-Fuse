"""Dataset loading and preprocessing for the public MSWP-Fuse release."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr
import yaml
from torch.utils.data import Dataset

HISTORY_STEPS = 672
FORECAST_STEPS = 96
N_SOURCES = 3
N_CHANNELS = 9
STATION_IDS = tuple(range(1, 11))
SOURCES = ("NWP_1", "NWP_2", "NWP_3")
CHANNELS = (
    "ghi",
    "poai",
    "pressure",
    "t2m",
    "tcc",
    "tp",
    "u100",
    "v100",
    "wind_speed",
)
RESOURCE_BY_STATION = {station: "wind" if station <= 5 else "solar" for station in STATION_IDS}


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_paper_settings() -> dict[str, Any]:
    path = package_root() / "configs" / "paper.yaml"
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("paper.yaml must contain a mapping")
    return value


def default_data_root() -> Path:
    settings = load_paper_settings()
    relative = settings["data"]["default_root"]
    if not isinstance(relative, str):
        raise TypeError("data.default_root must be a string")
    return Path(relative)


def fold_names() -> tuple[str, ...]:
    settings = load_paper_settings()
    return tuple(settings["folds"])


def _as_date(value: object) -> date:
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"Expected an ISO date, received {type(value).__name__}")


def fold_dates(fold: str, split: str) -> tuple[date, date]:
    settings = load_paper_settings()
    try:
        values = settings["folds"][fold][split]
    except KeyError as error:
        raise KeyError(f"Unknown fold or split: {fold}/{split}") from error
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or len(values) != 2:
        raise ValueError(f"Invalid date range for {fold}/{split}")
    start, end = (_as_date(values[0]), _as_date(values[1]))
    if start > end:
        raise ValueError(f"Invalid date order for {fold}/{split}")
    return start, end


@dataclass(frozen=True)
class SampleRef:
    station_id: int
    target_date: date

    @property
    def resource(self) -> str:
        return RESOURCE_BY_STATION[self.station_id]

    @property
    def nwp_file_date(self) -> str:
        return (self.target_date - timedelta(days=1)).strftime("%Y%m%d")

    @property
    def key(self) -> str:
        return f"{self.station_id}:{self.target_date.isoformat()}"


@dataclass(frozen=True)
class NWPScaler:
    """Per-provider and per-channel standardization statistics."""

    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        if self.mean.shape != (N_SOURCES * N_CHANNELS,) or self.std.shape != self.mean.shape:
            raise ValueError("NWP statistics must have 27 values")
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("NWP statistics must be finite with positive standard deviations")

    def transform(self, values: np.ndarray, availability: np.ndarray) -> np.ndarray:
        if values.shape != (FORECAST_STEPS, N_SOURCES * N_CHANNELS):
            raise ValueError(f"Expected NWP shape (96, 27), received {values.shape}")
        if availability.shape != (N_SOURCES,):
            raise ValueError("NWP availability must have three values")
        result = (values - self.mean[None, :]) / self.std[None, :]
        for source_index, available in enumerate(availability):
            if not bool(available):
                first = source_index * N_CHANNELS
                result[:, first : first + N_CHANNELS] = 0.0
        return result.astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "mean": self.mean.astype(float).tolist(),
            "std": self.std.astype(float).tolist(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "NWPScaler":
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float32),
            std=np.asarray(value["std"], dtype=np.float32),
        )


class RawDataStore:
    """Reads the released CSV and NetCDF files without target-day imputation."""

    def __init__(self, root: Path | str):
        self.root = Path(root).expanduser()
        self._power_cache: dict[int, pd.Series] = {}
        self._nwp_cache: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
        settings = load_paper_settings()
        self._channel_mapping = settings["data"]["nwp"]["channel_mapping"]

    def power_path(self, station_id: int) -> Path:
        return self.root / "train" / "fact_data_train" / f"{station_id}_normalization_train.csv"

    def nwp_path(self, ref: SampleRef, source: str) -> Path:
        if source not in SOURCES:
            raise KeyError(f"Unknown NWP source: {source}")
        return (
            self.root
            / "train"
            / "nwp_data_train"
            / str(ref.station_id)
            / source
            / f"{ref.nwp_file_date}.nc"
        )

    def history_target(self, ref: SampleRef) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        start = pd.Timestamp(ref.target_date)
        history_index = pd.date_range(start - timedelta(days=7), periods=HISTORY_STEPS, freq="15min")
        target_index = pd.date_range(start, periods=FORECAST_STEPS, freq="15min")
        series = self._power_series(ref.station_id)

        history = series.reindex(series.index.union(history_index)).sort_index().ffill()
        history_values = history.reindex(history_index).fillna(0.0).clip(lower=0.0).to_numpy(np.float32)
        raw_target = series.reindex(target_index).to_numpy(np.float32)
        valid_target_mask = np.isfinite(raw_target) & (raw_target >= 0.0)
        target = np.where(valid_target_mask, raw_target, 0.0).astype(np.float32)
        if history_values.shape != (HISTORY_STEPS,) or target.shape != (FORECAST_STEPS,):
            raise ValueError(f"Unexpected power shape for {ref.key}")
        return history_values, target, raw_target, valid_target_mask

    def nwp(self, ref: SampleRef) -> tuple[np.ndarray, np.ndarray]:
        key = (ref.station_id, ref.nwp_file_date)
        cached = self._nwp_cache.get(key)
        if cached is not None:
            return cached
        blocks: list[np.ndarray] = []
        availability = np.zeros(N_SOURCES, dtype=bool)
        for source_index, source in enumerate(SOURCES):
            path = self.nwp_path(ref, source)
            if path.is_file():
                blocks.append(self._load_source(path, source))
                availability[source_index] = True
            else:
                blocks.append(np.zeros((FORECAST_STEPS, N_CHANNELS), dtype=np.float32))
        if not bool(availability.any()):
            raise FileNotFoundError(f"No NWP input is available for {ref.key}")
        value = (np.concatenate(blocks, axis=1).astype(np.float32, copy=False), availability)
        self._nwp_cache[key] = value
        return value

    def _power_series(self, station_id: int) -> pd.Series:
        cached = self._power_cache.get(station_id)
        if cached is not None:
            return cached
        path = self.power_path(station_id)
        if not path.is_file():
            raise FileNotFoundError(f"Power file is missing: {path}")
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if frame.shape[1] != 2:
            raise ValueError(f"Expected two columns in {path}")
        frame.columns = ["timestamp", "power"]
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["power"] = pd.to_numeric(frame["power"], errors="coerce")
        value = frame.drop_duplicates("timestamp", keep="last").set_index("timestamp")["power"].sort_index()
        self._power_cache[station_id] = value
        return value

    def _load_source(self, path: Path, source: str) -> np.ndarray:
        with xr.open_dataset(path) as dataset:
            if "channel" not in dataset.coords or "data" not in dataset:
                raise ValueError(f"NetCDF file lacks data or channel: {path}")
            values = dataset["data"].isel(time=0)
            if values.sizes.get("lead_time") != 24:
                raise ValueError(f"Expected 24 lead times in {path}")
            labels = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in dataset.coords["channel"].values]
            center_lat = values.sizes["lat"] // 2
            center_lon = values.sizes["lon"] // 2
            spatial_mean = values.isel(
                lat=slice(center_lat - 1, center_lat + 2),
                lon=slice(center_lon - 1, center_lon + 2),
            ).mean(dim=("lat", "lon"))
            hourly = np.asarray(spatial_mean.transpose("lead_time", "channel").values, dtype=np.float64)
        if hourly.shape[1] != len(labels) or len(set(labels)) != len(labels):
            raise ValueError(f"Invalid NWP channel labels in {path}")
        by_label = {label: hourly[:, index] for index, label in enumerate(labels)}
        mapping = self._channel_mapping[source]
        columns = []
        for name in CHANNELS[:-1]:
            source_name = mapping[name]
            if source_name not in by_label:
                raise ValueError(f"Missing NWP channel {source_name!r} in {path}")
            columns.append(by_label[source_name])
        base = interpolate_hourly_nwp(np.stack(columns, axis=1))
        wind_speed = np.sqrt(np.square(base[:, 6]) + np.square(base[:, 7]))
        result = np.concatenate([base, wind_speed[:, None]], axis=1)
        for name in ("ghi", "poai", "tp", "wind_speed"):
            result[:, CHANNELS.index(name)] = np.maximum(result[:, CHANNELS.index(name)], 0.0)
        return result.astype(np.float32, copy=False)


class PowerNWPDataset(Dataset[dict[str, torch.Tensor | str]]):
    """One 96-step prediction target per station and date."""

    def __init__(
        self,
        refs: Sequence[SampleRef],
        store: RawDataStore,
        scaler: NWPScaler,
        station_ids: Sequence[int] = STATION_IDS,
    ):
        self.refs = tuple(refs)
        self.store = store
        self.scaler = scaler
        self.station_index = {station: index for index, station in enumerate(station_ids)}
        if tuple(station_ids) != STATION_IDS:
            raise ValueError("The public release uses stations 1 through 10 in order")

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        ref = self.refs[index]
        history, target, raw_target, valid_target_mask = self.store.history_target(ref)
        difference = np.zeros_like(history)
        difference[1:] = history[1:] - history[:-1]
        raw_nwp, availability = self.store.nwp(ref)
        history_time, future_time = sample_time_features(ref.target_date)
        return {
            "hist_power": torch.from_numpy(np.stack([history, difference], axis=1)),
            "hist_time": torch.from_numpy(history_time),
            "future_time": torch.from_numpy(future_time),
            "nwp": torch.from_numpy(self.scaler.transform(raw_nwp, availability)),
            "nwp_availability": torch.from_numpy(availability.astype(np.int64)),
            "target": torch.from_numpy(target),
            "raw_target": torch.from_numpy(raw_target),
            "target_valid_mask": torch.from_numpy(valid_target_mask),
            "station_id": torch.tensor(self.station_index[ref.station_id], dtype=torch.long),
            "original_station_id": torch.tensor(ref.station_id, dtype=torch.long),
            "resource_id": torch.tensor(0 if ref.resource == "wind" else 1, dtype=torch.long),
            "target_ordinal": torch.tensor(ref.target_date.toordinal(), dtype=torch.long),
            "sample_key": ref.key,
        }


def sample_time_features(target_date: date) -> tuple[np.ndarray, np.ndarray]:
    start = pd.Timestamp(target_date)
    history = pd.date_range(start - timedelta(days=7), periods=HISTORY_STEPS, freq="15min")
    future = pd.date_range(start, periods=FORECAST_STEPS, freq="15min")
    return _time_features(history), _time_features(future)


def _time_features(index: pd.DatetimeIndex) -> np.ndarray:
    quarter = (index.hour * 4 + index.minute // 15).to_numpy(dtype=np.float64)
    day_of_year = index.dayofyear.to_numpy(dtype=np.float64)
    weekday = index.dayofweek.to_numpy(dtype=np.float64)
    return np.stack(
        [
            np.sin(2.0 * np.pi * quarter / 96.0),
            np.cos(2.0 * np.pi * quarter / 96.0),
            np.sin(2.0 * np.pi * day_of_year / 366.0),
            np.cos(2.0 * np.pi * day_of_year / 366.0),
            weekday / 6.0 - 0.5,
        ],
        axis=1,
    ).astype(np.float32)


def interpolate_hourly_nwp(hourly: np.ndarray) -> np.ndarray:
    if hourly.ndim != 2 or hourly.shape[0] != 24:
        raise ValueError(f"Expected hourly NWP shape (24, C), received {hourly.shape}")
    grid = np.arange(24, dtype=np.float64)
    target_grid = np.arange(FORECAST_STEPS, dtype=np.float64) / 4.0
    result = np.empty((FORECAST_STEPS, hourly.shape[1]), dtype=np.float64)
    for index in range(hourly.shape[1]):
        result[:, index] = np.interp(target_grid, grid, hourly[:, index], left=hourly[0, index], right=hourly[-1, index])
    return result


def build_references(start: date, end: date, station_ids: Sequence[int] = STATION_IDS) -> list[SampleRef]:
    if tuple(station_ids) != STATION_IDS:
        raise ValueError("The public release requires all ten stations")
    result: list[SampleRef] = []
    current = start
    while current <= end:
        result.extend(SampleRef(station, current) for station in station_ids)
        current += timedelta(days=1)
    return result


def fit_nwp_scaler(store: RawDataStore, refs: Iterable[SampleRef]) -> NWPScaler:
    dimension = N_SOURCES * N_CHANNELS
    total = np.zeros(dimension, dtype=np.float64)
    total_sq = np.zeros(dimension, dtype=np.float64)
    count = np.zeros(dimension, dtype=np.int64)
    for ref in refs:
        values, available = store.nwp(ref)
        for source_index, present in enumerate(available):
            if not bool(present):
                continue
            first = source_index * N_CHANNELS
            block = values[:, first : first + N_CHANNELS].astype(np.float64)
            total[first : first + N_CHANNELS] += block.sum(axis=0)
            total_sq[first : first + N_CHANNELS] += np.square(block).sum(axis=0)
            count[first : first + N_CHANNELS] += block.shape[0]
    if np.any(count == 0):
        raise ValueError(f"No training values for NWP positions {np.flatnonzero(count == 0).tolist()}")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 1.0e-12)
    std = np.sqrt(variance)
    std = np.where(std < 1.0e-6, 1.0, std)
    return NWPScaler(mean.astype(np.float32), std.astype(np.float32))


def build_fold_datasets(
    data_root: Path | str,
    fold: str,
) -> tuple[dict[str, PowerNWPDataset], NWPScaler]:
    store = RawDataStore(data_root)
    refs = {name: build_references(*fold_dates(fold, name)) for name in ("train", "validation", "test")}
    scaler = fit_nwp_scaler(store, refs["train"])
    return {name: PowerNWPDataset(value, store, scaler) for name, value in refs.items()}, scaler


def build_split_dataset(
    data_root: Path | str,
    fold: str,
    split: str,
    scaler: NWPScaler,
) -> PowerNWPDataset:
    store = RawDataStore(data_root)
    return PowerNWPDataset(build_references(*fold_dates(fold, split)), store, scaler)
