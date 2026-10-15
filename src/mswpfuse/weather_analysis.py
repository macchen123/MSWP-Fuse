"""The paper's matched Full/Head-only, training-tertile weather comparison."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import (
    CHANNELS, FORECAST_STEPS, N_CHANNELS, N_SOURCES, NWPScaler,
    RawDataStore, SampleRef, build_references, fit_nwp_scaler, fold_dates, fold_names,
)
from .prediction_archives import METRIC_NAMES, align_prediction_archives, load_fold_archives, score_group

COVARIATES = ("wind_speed", "solar_daytime_poai", "source_disagreement")
TERTILES = ("low", "middle", "high")
_DISAGREEMENT_CHANNELS = ("ghi", "poai", "t2m", "tcc", "u100", "v100", "wind_speed")


def weather_covariates(store: RawDataStore, ref: SampleRef, scaler: NWPScaler) -> dict[str, float | None]:
    """Use issued NWP only; unavailable providers never contribute zero padding."""
    raw, availability = store.nwp(ref)
    availability = np.asarray(availability)
    if raw.shape != (FORECAST_STEPS, N_SOURCES * N_CHANNELS) or availability.shape != (N_SOURCES,):
        raise ValueError(f"Invalid issued NWP shape for {ref.key}")
    if availability.dtype.kind != "b" or not bool(availability.any()):
        raise ValueError(f"Invalid NWP availability for {ref.key}")
    available = raw.reshape(FORECAST_STEPS, N_SOURCES, N_CHANNELS)[:, availability, :]
    if not bool(np.isfinite(available).all()):
        raise ValueError(f"Non-finite available NWP input for {ref.key}")
    values: dict[str, float | None] = dict.fromkeys(COVARIATES)
    if ref.resource == "wind":
        values["wind_speed"] = float(np.mean(available[:, :, CHANNELS.index("wind_speed")], dtype=np.float64))
    else:
        values["solar_daytime_poai"] = float(np.mean(available[24:72, :, CHANNELS.index("poai")], dtype=np.float64))
    if int(availability.sum()) >= 2:
        standardized = scaler.transform(raw, availability).reshape(FORECAST_STEPS, N_SOURCES, N_CHANNELS)
        channels = [CHANNELS.index(name) for name in _DISAGREEMENT_CHANNELS]
        selected = standardized[:, availability, :][:, :, channels]
        disagreement = np.std(selected, axis=1, ddof=0, dtype=np.float64)
        values["source_disagreement"] = float(np.mean(disagreement, dtype=np.float64))
    if any(value is not None and not np.isfinite(value) for value in values.values()):
        raise ValueError(f"Non-finite weather covariate for {ref.key}")
    return values


def training_weather_boundaries(
    store: RawDataStore, fold: str,
) -> tuple[NWPScaler, dict[str, dict[str, object]]]:
    """Fit the canonical scaler and linear empirical tertiles on TRAIN refs only."""
    refs = build_references(*fold_dates(fold, "train"), store=store)
    scaler = fit_nwp_scaler(store, refs)
    values: dict[str, list[float]] = {name: [] for name in COVARIATES}
    for ref in refs:
        for name, value in weather_covariates(store, ref, scaler).items():
            if value is not None:
                values[name].append(value)
    boundaries: dict[str, dict[str, object]] = {}
    for name, observed in values.items():
        quantiles = np.quantile(observed, [1.0 / 3.0, 2.0 / 3.0], method="linear") if observed else (None, None)
        boundaries[name] = {
            "q1": float(quantiles[0]) if observed else None,
            "q2": float(quantiles[1]) if observed else None,
            "training_station_day_count": len(observed),
        }
    return scaler, boundaries


def assign_tertiles(values: np.ndarray, q1: float | None, q2: float | None) -> np.ndarray:
    """Return low=0, middle=1, high=2; undefined covariates remain unassigned=-1."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("Weather covariates must be a vector")
    groups = np.full(values.shape, -1, dtype=np.int8)
    if q1 is None and q2 is None:
        return groups
    if q1 is None or q2 is None or not np.isfinite([q1, q2]).all() or q1 > q2:
        raise ValueError("Weather boundaries must be finite and ordered, or both undefined")
    defined = np.isfinite(values)
    groups[defined & (values <= q1)] = 0
    groups[defined & (values > q1) & (values <= q2)] = 1
    groups[defined & (values > q2)] = 2
    return groups


def analyze_weather(data_root: Path | str, full_directory: Path | str, head_only_directory: Path | str) -> dict[str, object]:
    """Return exactly the three covariates × three windows × three tertiles."""
    full = load_fold_archives(full_directory, model="MSWP-Fuse", variant="full", require_full_lineage=True, require_stage_seeds=True)
    head_only = load_fold_archives(head_only_directory, model="MSWP-Fuse", variant="head_only", require_full_lineage=True, require_stage_seeds=True)
    for fold in fold_names():
        head_only[fold] = align_prediction_archives(full[fold], head_only[fold])
        if full[fold].metadata["seed"] != head_only[fold].metadata["seed"]:
            raise ValueError(f"Full and Head-only must share the same training seed for {fold}")
        for field in ("initializer_sha256", "parent_sha256"):
            if full[fold].metadata[field] != head_only[fold].metadata[field]:
                raise ValueError(f"Full and Head-only must share the same {field} for {fold}")
        if full[fold].metadata["checkpoint_sha256"] == head_only[fold].metadata["checkpoint_sha256"]:
            raise ValueError(f"Full and Head-only correction checkpoints are identical for {fold}")

    store = RawDataStore(data_root)
    thresholds: dict[str, object] = {}
    coverage: dict[str, object] = {}
    contrasts: list[dict[str, object]] = []
    for fold in fold_names():
        archive = full[fold]
        refs = {ref.key: ref for ref in build_references(*fold_dates(fold, "test"), store=store)}
        if set(refs) != set(archive.sample_keys.tolist()):
            raise ValueError(f"Local NWP references do not match the complete {fold} prediction station-day set")
        scaler, boundaries = training_weather_boundaries(store, fold)
        thresholds[fold] = {
            "training_dates": [value.isoformat() for value in fold_dates(fold, "train")],
            "covariates": boundaries,
        }
        observations = [weather_covariates(store, refs[str(key)], scaler) for key in archive.sample_keys]
        fold_coverage: dict[str, object] = {}
        for name in COVARIATES:
            values = np.asarray([value[name] if value[name] is not None else np.nan for value in observations])
            groups = assign_tertiles(values, boundaries[name]["q1"], boundaries[name]["q2"])
            resource = 0 if name == "wind_speed" else 1 if name == "solar_daytime_poai" else None
            eligible = np.ones(values.size, dtype=bool) if resource is None else archive.resource_ids == resource
            fold_coverage[name] = {
                "resource_station_day_count": int(eligible.sum()),
                "assigned_station_day_count": int((groups >= 0).sum()),
                "undefined_station_day_count": int((eligible & (groups < 0)).sum()),
            }
            for index, tertile in enumerate(TERTILES):
                selected = groups == index
                full_scores = score_group(archive, selected)
                head_scores = score_group(head_only[fold], selected)
                full_metrics, head_metrics = full_scores["metrics"], head_scores["metrics"]
                deltas = {
                    metric: None if full_metrics[metric] is None or head_metrics[metric] is None else full_metrics[metric] - head_metrics[metric]
                    for metric in METRIC_NAMES
                }
                contrasts.append({
                    "covariate": name, "window": fold, "tertile": tertile,
                    "matched_station_day_count": full_scores["station_day_count"],
                    "scored_station_day_count": full_scores["scored_station_day_count"],
                    "valid_target_count": full_scores["valid_target_count"],
                    "full": full_metrics, "head_only": head_metrics,
                    "full_minus_head_only": deltas,
                })
        coverage[fold] = fold_coverage
    return {
        "analysis": "matched_weather_tertiles",
        "windows": list(fold_names()),
        "matched_station_day_count": sum(archive.station_ids.size for archive in full.values()),
        "threshold_source": "fold_training_issued_nwp",
        "quantile_method": "linear",
        "tertile_intervals": {"low": "x <= q1", "middle": "q1 < x <= q2", "high": "x > q2"},
        "score_horizon": "all_96_positions_with_recorded_valid_targets_including_pv_nights",
        "thresholds": thresholds, "coverage": coverage, "groups": contrasts,
    }
