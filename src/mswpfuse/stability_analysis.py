"""Descriptive mean/sample-SD results for exactly six supplied final instances."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .data import fold_dates, fold_names
from .metrics import RESOURCE_NAMES
from .prediction_archives import (
    LINEAGE_FIELDS, METRIC_NAMES, PredictionArchive, align_prediction_archives,
    load_fold_archives, pool_archives, score_group,
)

REFERENCE_MODEL = "NWPSeq2Seq"
INSTANCE_COUNT = 6


def _resolved_run_directories(runs: Sequence[Path | str]) -> list[Path]:
    if len(runs) != INSTANCE_COUNT:
        raise ValueError("Supply exactly six final instance directories; no candidate search or top-k selection is performed")
    paths = [Path(path).expanduser().resolve() for path in runs]
    if len(set(paths)) != INSTANCE_COUNT:
        raise ValueError("The six final instance directories must be distinct")
    return paths


def _group_scores(folds: dict[str, PredictionArchive]) -> dict[tuple[str, str], dict[str, object]]:
    archives = {"pooled": pool_archives([folds[fold] for fold in fold_names()]), **folds}
    groups = {}
    for window, archive in archives.items():
        groups[(window, "portfolio")] = score_group(archive)
        for resource_id, name in RESOURCE_NAMES.items():
            groups[(window, name)] = score_group(archive, archive.resource_ids == resource_id)
    return groups


def summarize_stability(runs: Sequence[Path | str], reference_directory: Path | str) -> dict[str, object]:
    """Validate the supplied cohort and report it without selecting or replacing runs.

    The six instances, not an unselected population of repeated trials, are the
    unit of dispersion. Pooled metrics are recomputed on all three windows.
    Numeric seed identities are used only for validation and are not reported.
    """
    paths = _resolved_run_directories(runs)
    reference = load_fold_archives(reference_directory, model=REFERENCE_MODEL)
    instances = []
    instance_seeds = set()
    lineage = {fold: {field: set() for field in LINEAGE_FIELDS} for fold in fold_names()}
    for path in paths:
        folds = load_fold_archives(
            path, model="MSWP-Fuse", variant="full",
            require_full_lineage=True, require_stage_seeds=True,
        )
        seeds = {archive.metadata["seed"] for archive in folds.values()}
        if len(seeds) != 1:
            raise ValueError("Each final instance must use the same seed across all three windows")
        seed = next(iter(seeds))
        if seed in instance_seeds:
            raise ValueError("The six supplied final instances must have six distinct seeds")
        instance_seeds.add(seed)
        for fold, archive in folds.items():
            folds[fold] = align_prediction_archives(reference[fold], archive)
            for field in LINEAGE_FIELDS:
                digest = archive.metadata[field]
                if digest in lineage[fold][field]:
                    raise ValueError(f"Instances share {field} in {fold}; independently fit all three training stages")
                lineage[fold][field].add(digest)
        instances.append(_group_scores(folds))

    reference_groups = _group_scores(reference)
    reference_pooled = reference_groups[("pooled", "portfolio")]["metrics"]
    reference_c = reference_pooled["c"]
    if reference_c is None:
        raise ValueError("The fixed reference has no defined pooled C")
    conditions = []
    for index, groups in enumerate(instances, start=1):
        metrics = groups[("pooled", "portfolio")]["metrics"]
        pooled_c = metrics["c"]
        conditions.append({
            "instance": index,
            "pooled_metrics": metrics,
            "pooled_c_minus_reference": None if pooled_c is None else pooled_c - reference_c,
            "pooled_c_above_fixed_reference": pooled_c is not None and pooled_c > reference_c,
        })

    summaries = []
    for (window, resource), reference_group in reference_groups.items():
        scores = [groups[(window, resource)] for groups in instances]
        metrics = {}
        for name in METRIC_NAMES:
            values = [score["metrics"][name] for score in scores if score["metrics"][name] is not None]
            complete = len(values) == INSTANCE_COUNT
            metrics[name] = {
                "mean": float(np.mean(values, dtype=np.float64)) if complete else None,
                "sample_sd": float(np.std(values, ddof=1, dtype=np.float64)) if complete else None,
                "defined_instance_count": len(values),
            }
        summaries.append({
            "window": window, "resource": resource,
            "instance_count": INSTANCE_COUNT,
            "station_day_count": reference_group["station_day_count"],
            "scored_station_day_count": reference_group["scored_station_day_count"],
            "valid_target_count": reference_group["valid_target_count"],
            "metrics": metrics,
        })
    return {
        "analysis": "six_supplied_final_instances",
        "instance_count": INSTANCE_COUNT,
        "windows": {
            fold: [value.isoformat() for value in fold_dates(fold, "test")]
            for fold in fold_names()
        },
        "sample_sd_ddof": 1,
        "scope": "descriptive_statistics_of_the_six_supplied_instances",
        "selection_condition": "pooled_c_above_fixed_nwp_gru_reference",
        "all_six_meet_selection_condition": all(row["pooled_c_above_fixed_reference"] for row in conditions),
        "reference": {"model": REFERENCE_MODEL, "display_name": "NWP-GRU Seq2Seq", "pooled_metrics": reference_pooled},
        "instances": conditions,
        "groups": summaries,
    }
