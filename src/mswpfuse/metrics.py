"""Official score and pointwise metrics used by the release commands."""

from __future__ import annotations

from datetime import date
from typing import Sequence

import numpy as np

RESOURCE_NAMES = {0: "wind", 1: "solar"}


def compute_metrics(
    predictions: np.ndarray,
    raw_targets: np.ndarray,
    valid_target_mask: np.ndarray,
    station_ids: np.ndarray,
    resource_ids: np.ndarray,
    target_ordinals: np.ndarray,
    sample_keys: Sequence[str] | None = None,
) -> dict[str, object]:
    """Compute the paper's C, MAE, RMSE, coefficient of determination, and SMAPE.

    Missing and negative targets are excluded. A non-finite prediction on a valid
    station-day gives that station-day a C value of zero and makes the pointwise
    metrics unavailable.
    """

    predictions = np.asarray(predictions, dtype=np.float64)
    raw_targets = np.asarray(raw_targets, dtype=np.float64)
    valid_target_mask = np.asarray(valid_target_mask, dtype=bool)
    station_ids = np.asarray(station_ids, dtype=np.int64).reshape(-1)
    resource_ids = np.asarray(resource_ids, dtype=np.int64).reshape(-1)
    target_ordinals = np.asarray(target_ordinals, dtype=np.int64).reshape(-1)
    if predictions.ndim != 2 or predictions.shape != raw_targets.shape:
        raise ValueError("Predictions and raw targets must share shape (N, 96)")
    if predictions.shape[1] != 96 or valid_target_mask.shape != predictions.shape:
        raise ValueError("Metric arrays must use a 96-step horizon")
    rows = predictions.shape[0]
    if rows == 0 or any(values.shape != (rows,) for values in (station_ids, resource_ids, target_ordinals)):
        raise ValueError("Metadata does not match prediction rows")
    if not set(np.unique(resource_ids)).issubset(RESOURCE_NAMES):
        raise ValueError("Unknown resource ID")
    station_days = list(zip(station_ids.tolist(), target_ordinals.tolist(), strict=True))
    if len(set(station_days)) != rows:
        raise ValueError("Station-day rows must be unique")

    mask = valid_target_mask & np.isfinite(raw_targets) & (raw_targets >= 0.0)
    daily_c = np.full(rows, np.nan, dtype=np.float64)
    failed_rows = np.zeros(rows, dtype=bool)
    for row in range(rows):
        valid = mask[row]
        if not bool(valid.any()):
            continue
        predicted = predictions[row, valid]
        if not bool(np.isfinite(predicted).all()):
            daily_c[row] = 0.0
            failed_rows[row] = True
            continue
        target = raw_targets[row, valid]
        normalized_error = (target - predicted) / np.maximum(target, 0.2)
        daily_c[row] = 1.0 - np.sqrt(np.mean(np.square(normalized_error)))

    per_station: dict[str, dict[str, object]] = {}
    station_scores: list[float] = []
    for station in sorted(np.unique(station_ids)):
        selected = station_ids == station
        included = selected & np.isfinite(daily_c)
        if not bool(included.any()):
            continue
        resources = np.unique(resource_ids[selected])
        if resources.size != 1:
            raise ValueError(f"Station {int(station)} has inconsistent resources")
        score = float(np.mean(daily_c[included]))
        station_scores.append(score)
        per_station[str(int(station))] = {
            "c": score,
            "resource": RESOURCE_NAMES[int(resources[0])],
            "included_days": int(included.sum()),
            "excluded_days": int(selected.sum() - included.sum()),
        }
    if not station_scores:
        raise ValueError("No station-day has a valid target")

    per_resource: dict[str, float] = {}
    for resource_id, name in RESOURCE_NAMES.items():
        scores = [
            float(value["c"])
            for station, value in per_station.items()
            if value["resource"] == name
        ]
        if scores:
            per_resource[name] = float(np.mean(scores))

    nonfinite_prediction_count = int((mask & ~np.isfinite(predictions)).sum())
    r2: float | None
    smape_percent: float | None
    both_zero_count = 0
    if nonfinite_prediction_count:
        mae = None
        rmse = None
        r2 = None
        smape_percent = None
    else:
        predicted = predictions[mask]
        target = raw_targets[mask]
        error = predicted - target
        mae = float(np.mean(np.abs(error), dtype=np.float64))
        residual = float(np.sum(np.square(error), dtype=np.float64))
        rmse = float(np.sqrt(residual / target.size))
        target_mean = float(np.mean(target, dtype=np.float64))
        total = float(np.sum(np.square(target - target_mean), dtype=np.float64))
        r2 = None if total <= 0.0 else 1.0 - residual / total
        denominator = (np.abs(target) + np.abs(predicted)) / 2.0
        terms = np.zeros_like(denominator)
        nonzero = denominator > 0.0
        terms[nonzero] = np.abs(error[nonzero]) / denominator[nonzero]
        smape_percent = float(100.0 * np.mean(terms, dtype=np.float64))
        both_zero_count = int((~nonzero).sum())

    keys = tuple(sample_keys) if sample_keys is not None else tuple(
        f"{station}:{date.fromordinal(ordinal).isoformat()}" for station, ordinal in station_days
    )
    if len(keys) != rows or len(set(keys)) != rows:
        raise ValueError("Sample keys must be unique and align with predictions")

    return {
        "c": float(np.mean(station_scores)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "smape_percent": smape_percent,
        "both_zero_count": both_zero_count,
        "smape_zero_zero_convention": "zero_contribution",
        "wind_c": per_resource.get("wind"),
        "solar_c": per_resource.get("solar"),
        "valid_target_count": int(mask.sum()),
        "invalid_target_count": int(mask.size - mask.sum()),
        "failed_prediction_station_days": int(failed_rows.sum()),
        "nonfinite_prediction_count": nonfinite_prediction_count,
        "per_station": per_station,
    }
