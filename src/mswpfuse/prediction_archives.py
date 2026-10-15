"""Validated station-day prediction archives shared by the paper analyses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
import json
from pathlib import Path
import re

import numpy as np

from .data import FORECAST_STEPS, RESOURCE_BY_STATION, STATION_IDS, SampleRef, build_references, fold_dates, fold_names, load_paper_settings
from .metrics import compute_metrics

METRIC_NAMES = ("c", "mae", "rmse", "r2", "smape_percent")
LINEAGE_FIELDS = ("initializer_sha256", "parent_sha256", "checkpoint_sha256")
ARRAY_FIELDS = (
    "predictions", "targets", "raw_targets", "valid_target_mask",
    "station_ids", "resource_ids", "target_ordinals", "sample_keys",
)
_METADATA_FIELDS = {"model", "fold", "split", "variant", "seed", "stage_seeds", "configuration", *LINEAGE_FIELDS}


@dataclass(frozen=True)
class PredictionArchive:
    predictions: np.ndarray
    targets: np.ndarray
    raw_targets: np.ndarray
    valid_target_mask: np.ndarray
    station_ids: np.ndarray
    resource_ids: np.ndarray
    target_ordinals: np.ndarray
    sample_keys: np.ndarray
    metadata: Mapping[str, object]

    def take(self, indices: np.ndarray) -> "PredictionArchive":
        return replace(self, **{name: getattr(self, name)[indices] for name in ARRAY_FIELDS})


def _text(values: np.ndarray, name: str) -> np.ndarray:
    if values.dtype.kind == "S":
        try:
            return np.char.decode(values, "utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} must contain UTF-8 text") from error
    if values.dtype.kind != "U":
        raise ValueError(f"{name} must contain UTF-8 text, not object arrays")
    return values


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate metadata field: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON metadata value: {value}")


def _metadata(value: np.ndarray) -> dict[str, object]:
    if value.shape != ():
        raise ValueError("metadata must be a scalar UTF-8 JSON object")
    text = _text(value, "metadata").item()
    try:
        result = json.loads(text, object_pairs_hook=_unique_json_object, parse_constant=_invalid_json_constant)
    except (ValueError, TypeError) as error:
        raise ValueError("metadata must be a strict JSON object with unique fields") from error
    if not isinstance(result, dict) or set(result) - _METADATA_FIELDS:
        raise ValueError("metadata contains unsupported fields or is not an object")
    for field in ("model", "fold", "split"):
        if not isinstance(result.get(field), str) or not result[field].strip():
            raise ValueError(f"metadata requires a nonempty {field}")
    if result["fold"] not in fold_names() or result["split"] != "test":
        raise ValueError("Paper analyses require a configured fold and split='test'")
    if "variant" in result and (not isinstance(result["variant"], str) or not result["variant"].strip()):
        raise ValueError("metadata variant must be nonempty text")
    if result.get("configuration") is not None:
        configuration = result["configuration"]
        if type(configuration) is not int or configuration not in load_paper_settings()["configurations"]:
            raise ValueError("metadata configuration must identify an existing paper configuration")
        if result["model"] == "MSWP-Fuse" and configuration != load_paper_settings(result["fold"])["configuration"]:
            raise ValueError("Prediction configuration differs from the final paper configuration for this window")
    if "seed" in result and (type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32):
        raise ValueError("metadata seed must be an integer accepted by the training RNGs")
    if "stage_seeds" in result:
        seeds = result["stage_seeds"]
        expected_stages = {"initializer", "parent"}
        if result.get("variant") != "stage1_only":
            expected_stages.add("selected")
        if not isinstance(seeds, dict) or set(seeds) != expected_stages:
            raise ValueError("metadata stage_seeds must identify exactly the model's completed training stages")
        if any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds.values()):
            raise ValueError("metadata stage_seeds must be integers accepted by the training RNGs")
    for field in LINEAGE_FIELDS:
        if field not in result:
            continue
        digest = result[field]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"metadata {field} must be a SHA-256 hex digest")
        result[field] = digest.lower()
    if "checkpoint_sha256" not in result:
        raise ValueError("Missing checkpoint identity; rerun prediction export with checkpoint metadata")
    return result


def load_prediction_archive(
    path: Path | str,
    *,
    expected_fold: str | None = None,
    expected_model: str | None = None,
    expected_variant: str | None = None,
    require_full_lineage: bool = False,
    require_stage_seeds: bool = False,
) -> PredictionArchive:
    """Read without pickle, validate identities, and sort by actual station/date.

    Raw missing/negative observations remain intact. Prediction values and the
    processed targets must be finite; masks must retain every legal raw target.
    Legacy exports need to be regenerated, not assigned inferred identities.
    """
    path = Path(path).expanduser()
    with np.load(path, allow_pickle=False) as source:
        if len(source.files) != len(set(source.files)):
            raise ValueError(f"Duplicate NPZ members in {path}")
        missing = set(ARRAY_FIELDS) - set(source.files)
        if missing:
            raise ValueError(f"Prediction archive lacks arrays: {', '.join(sorted(missing))}")
        if "metadata" not in source.files:
            raise ValueError("Missing prediction metadata; rerun prediction export with model, fold, split and checkpoint lineage")
        metadata = _metadata(source["metadata"])
        arrays = {name: source[name] for name in ARRAY_FIELDS}
    for field, expected in (("fold", expected_fold), ("model", expected_model), ("variant", expected_variant)):
        if expected is not None and metadata.get(field) != expected:
            raise ValueError(f"Expected metadata {field}={expected!r} in {path}")
    if require_full_lineage:
        missing = {"seed", *LINEAGE_FIELDS} - set(metadata)
        if missing:
            raise ValueError("Missing full-stage identity; rerun prediction export with seed and all three checkpoint lineage hashes")
        if len({metadata[field] for field in LINEAGE_FIELDS}) != len(LINEAGE_FIELDS):
            raise ValueError("Initializer, parent and correction checkpoints must have different identities")
    if require_stage_seeds:
        seeds = metadata.get("stage_seeds")
        if seeds is None or "seed" not in metadata:
            raise ValueError("Missing stage seed identity; rerun prediction export with actual three-stage seed lineage")
        if any(seed != metadata["seed"] for seed in seeds.values()):
            raise ValueError("All three stage_seeds must equal the supplied instance seed")

    predictions = arrays["predictions"]
    if predictions.ndim != 2 or predictions.shape[0] == 0 or predictions.shape[1] != FORECAST_STEPS:
        raise ValueError("Prediction arrays must have nonempty shape (N, 96)")
    rows = predictions.shape[0]
    for name in ("predictions", "targets", "raw_targets"):
        value = arrays[name]
        if value.shape != predictions.shape or value.dtype.kind not in "fiu":
            raise ValueError(f"{name} must be a real numeric array of shape (N, 96)")
        if name != "raw_targets" and not bool(np.isfinite(value).all()):
            raise ValueError(f"{name} must contain only finite values")
    mask = arrays["valid_target_mask"]
    if mask.shape != predictions.shape or mask.dtype.kind != "b":
        raise ValueError("valid_target_mask must be a boolean array of shape (N, 96)")
    raw = arrays["raw_targets"]
    if not np.array_equal(mask, np.isfinite(raw) & (raw >= 0.0)):
        raise ValueError("valid_target_mask must preserve exactly the finite nonnegative recorded targets")
    if not np.array_equal(arrays["targets"], np.where(mask, raw, 0.0)):
        raise ValueError("Processed targets disagree with raw targets and their recorded-target mask")
    for name in ("station_ids", "resource_ids", "target_ordinals"):
        value = arrays[name]
        if value.shape != (rows,) or value.dtype.kind not in "iu":
            raise ValueError(f"{name} must be an integer vector matching prediction rows")
    stations = arrays["station_ids"]
    ordinals = arrays["target_ordinals"]
    if not set(stations.tolist()).issubset(STATION_IDS):
        raise ValueError("station_ids must use the original station IDs 1 through 10")
    expected_resources = np.asarray([0 if RESOURCE_BY_STATION[int(station)] == "wind" else 1 for station in stations])
    if not np.array_equal(arrays["resource_ids"], expected_resources):
        raise ValueError("Resource IDs disagree with the recorded station identities")
    start, end = fold_dates(str(metadata["fold"]), "test")
    if bool(((ordinals < start.toordinal()) | (ordinals > end.toordinal())).any()):
        raise ValueError("Prediction target dates fall outside the declared test window")
    keys = _text(arrays["sample_keys"], "sample_keys")
    if keys.shape != (rows,):
        raise ValueError("sample_keys must be a vector matching prediction rows")
    identities = list(zip(stations.tolist(), ordinals.tolist(), strict=True))
    if len(set(identities)) != rows:
        raise ValueError("Duplicate station-day identities in prediction archive")
    expected_keys = [SampleRef(int(station), date.fromordinal(int(ordinal))).key for station, ordinal in identities]
    if keys.tolist() != expected_keys:
        raise ValueError("sample_keys disagree with actual station/date identities")
    arrays["sample_keys"] = keys
    archive = PredictionArchive(**arrays, metadata=metadata)
    return archive.take(np.lexsort((ordinals, stations)))


def align_prediction_archives(reference: PredictionArchive, other: PredictionArchive) -> PredictionArchive:
    """Match the complete station-day set; never intersect or drop observations."""
    first = list(zip(reference.station_ids.tolist(), reference.target_ordinals.tolist(), strict=True))
    second = list(zip(other.station_ids.tolist(), other.target_ordinals.tolist(), strict=True))
    if len(set(first)) != len(first) or len(set(second)) != len(second):
        raise ValueError("Duplicate station-day identities cannot be aligned")
    if set(first) != set(second):
        raise ValueError("Prediction station-day sets differ; refusing to intersect or drop samples")
    positions = {identity: index for index, identity in enumerate(second)}
    aligned = other.take(np.asarray([positions[identity] for identity in first], dtype=np.int64))
    for name in ARRAY_FIELDS[1:]:
        left, right = getattr(reference, name), getattr(aligned, name)
        equal = np.array_equal(left, right, equal_nan=True) if name in ("targets", "raw_targets") else np.array_equal(left, right)
        if not equal:
            raise ValueError(f"Matched prediction archives disagree on {name}")
    return aligned


def load_fold_archives(
    directory: Path | str,
    *,
    model: str,
    variant: str | None = None,
    require_full_lineage: bool = False,
    require_stage_seeds: bool = False,
) -> dict[str, PredictionArchive]:
    """Require every fixed paper window and its complete ten-station date grid."""
    directory = Path(directory).expanduser()
    paths = {fold: directory / f"{fold}.npz" for fold in fold_names()}
    missing = [path.name for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"All paper windows are required; missing archives: {', '.join(missing)}")
    archives = {}
    for fold, path in paths.items():
        archive = load_prediction_archive(
            path, expected_fold=fold, expected_model=model,
            expected_variant=variant, require_full_lineage=require_full_lineage,
            require_stage_seeds=require_stage_seeds,
        )
        expected = {ref.key for ref in build_references(*fold_dates(fold, "test"))}
        if set(archive.sample_keys.tolist()) != expected:
            raise ValueError(f"Incomplete {fold} test window; supply predictions for the complete paper station-day grid")
        archives[fold] = archive
    return archives


def pool_archives(archives: Sequence[PredictionArchive]) -> PredictionArchive:
    """Pool disjoint windows before computing metrics, rather than averaging scores."""
    if not archives:
        raise ValueError("No prediction archives supplied")
    pooled = PredictionArchive(
        **{name: np.concatenate([getattr(archive, name) for archive in archives]) for name in ARRAY_FIELDS},
        metadata={},
    )
    if len(set(pooled.sample_keys.tolist())) != pooled.sample_keys.size:
        raise ValueError("Pooled prediction archives contain overlapping station-days")
    return pooled


def score_group(archive: PredictionArchive, selected: np.ndarray | None = None) -> dict[str, object]:
    """Score all 96 positions using the single metric authority, including PV nights."""
    group = archive if selected is None else archive.take(selected)
    scored = group.valid_target_mask.any(axis=1)
    result: dict[str, object] = {
        "station_day_count": int(group.station_ids.size),
        "scored_station_day_count": int(scored.sum()),
        "valid_target_count": int(group.valid_target_mask.sum()),
        "metrics": dict.fromkeys(METRIC_NAMES),
    }
    if not bool(scored.any()):
        return result
    metrics = compute_metrics(
        group.predictions, group.raw_targets, group.valid_target_mask,
        group.station_ids, group.resource_ids, group.target_ordinals, group.sample_keys,
    )
    values = {name: metrics[name] for name in METRIC_NAMES}
    if any(value is not None and not np.isfinite(value) for value in values.values()):
        raise ValueError("Non-finite metric result; check the prediction magnitudes")
    result["metrics"] = values
    return result
