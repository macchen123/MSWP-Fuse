"""Synthetic archive, alignment and six-instance analysis contracts; no training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import pytest

from mswpfuse.analysis_cli import add_analysis_parsers, run_analysis
from mswpfuse.data import build_references, fold_dates, fold_names, load_paper_settings
from mswpfuse.metrics import compute_metrics
from mswpfuse.prediction_archives import (
    ARRAY_FIELDS, LINEAGE_FIELDS, METRIC_NAMES,
    align_prediction_archives, load_fold_archives, load_prediction_archive, pool_archives, score_group,
)
from mswpfuse.stability_analysis import summarize_stability


def synthetic_metadata(fold, instance=0, model="MSWP-Fuse", variant="full"):
    """Instance indices are artificial fixture identities, not paper seed values."""
    value = {
        "model": model, "fold": fold, "split": "test",
        "checkpoint_sha256": hashlib.sha256(f"synthetic:{fold}:{instance}:{variant}:selected".encode()).hexdigest(),
    }
    if model == "MSWP-Fuse":
        value.update({
            "variant": variant, "seed": instance,
            "stage_seeds": dict.fromkeys(("initializer", "parent", "selected"), instance),
            **{field: hashlib.sha256(f"synthetic:{fold}:{instance}:{field}".encode()).hexdigest() for field in LINEAGE_FIELDS[:-1]},
        })
    return value


def synthetic_payload(fold="summer", instance=0, error=0.02, *, complete=False, model="MSWP-Fuse", variant="full"):
    refs = build_references(*fold_dates(fold, "test"))
    if not complete:
        refs = refs[:12]
    stations = np.asarray([ref.station_id for ref in refs], dtype=np.int16)
    raw = np.tile(np.linspace(0.05, 0.95, 96), (len(refs), 1)) + stations[:, None] * 0.002
    solar = stations > 5
    raw[solar, :24] = 0.0
    raw[solar, 72:] = 0.0
    raw[0, 0] = np.nan
    raw[1, 1] = -0.1
    mask = np.isfinite(raw) & (raw >= 0.0)
    targets = np.where(mask, raw, 0.0)
    return {
        "predictions": targets + error, "targets": targets, "raw_targets": raw,
        "valid_target_mask": mask, "station_ids": stations,
        "resource_ids": solar.astype(np.int8),
        "target_ordinals": np.asarray([ref.target_date.toordinal() for ref in refs], dtype=np.int32),
        "sample_keys": np.asarray([ref.key for ref in refs]),
        "metadata": np.asarray(json.dumps(synthetic_metadata(fold, instance, model, variant))),
    }


def write_fold_archives(directory, instance=0, error=0.02, *, model="MSWP-Fuse", variant="full"):
    directory.mkdir(parents=True, exist_ok=True)
    for fold in fold_names():
        payload = synthetic_payload(fold, instance, error, complete=True, model=model, variant=variant)
        # The consumer must align identities, not assume producer row ordering.
        if instance % 2:
            payload.update({name: payload[name][::-1] for name in ARRAY_FIELDS})
        np.savez_compressed(directory / f"{fold}.npz", **payload)
    return directory


def rewrite_metadata(path, mutate):
    with np.load(path, allow_pickle=False) as source:
        payload = {name: source[name] for name in source.files}
    metadata = json.loads(payload["metadata"].item())
    mutate(metadata)
    payload["metadata"] = np.asarray(json.dumps(metadata))
    np.savez_compressed(path, **payload)


@pytest.fixture
def cohort(tmp_path):
    reference = write_fold_archives(tmp_path / "reference", error=0.18, model="NWPSeq2Seq")
    runs = [write_fold_archives(tmp_path / f"instance_{index}", index, 0.01 * (index + 1)) for index in range(6)]
    return runs, reference


def test_backbone_prediction_has_two_completed_stages(tmp_path):
    from mswpfuse.training import PredictionBundle
    from mswpfuse.training_commands import _save_predictions

    payload = synthetic_payload(variant="stage1_only")
    metadata = json.loads(payload.pop("metadata").item())
    metadata["stage_seeds"].pop("selected")
    metadata["parent_sha256"] = metadata["checkpoint_sha256"]
    path = tmp_path / "backbone.npz"
    _save_predictions(path, PredictionBundle(**payload), metadata)
    archive = load_prediction_archive(path, expected_variant="stage1_only")
    assert set(archive.metadata["stage_seeds"]) == {"initializer", "parent"}
    with pytest.raises(ValueError, match="different identities"):
        load_prediction_archive(path, require_full_lineage=True)


def test_current_prediction_producer_and_archive_consumer_agree(tmp_path):
    from mswpfuse.training import PredictionBundle
    from mswpfuse.training_commands import _save_predictions

    payload = synthetic_payload()
    metadata = json.loads(payload.pop("metadata").item())
    metadata["configuration"] = load_paper_settings("summer")["configuration"]
    bundle = PredictionBundle(**payload)
    path = tmp_path / "producer.npz"
    _save_predictions(path, bundle, metadata)
    archive = load_prediction_archive(
        path, expected_fold="summer", expected_model="MSWP-Fuse", expected_variant="full",
        require_full_lineage=True, require_stage_seeds=True,
    )
    assert archive.metadata == metadata
    np.testing.assert_allclose(np.sort(archive.predictions[:, 1]), np.sort(bundle.predictions[:, 1]))


def test_archive_aligns_actual_keys_and_preserves_missing_targets(tmp_path):
    payload = synthetic_payload()
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    np.savez(first, **payload)
    payload.update({name: payload[name][::-1] for name in ARRAY_FIELDS})
    payload["sample_keys"] = np.char.encode(payload["sample_keys"], "utf-8")
    payload["metadata"] = np.asarray(payload["metadata"].item().encode("utf-8"))
    np.savez(second, **payload)
    reference = load_prediction_archive(first)
    aligned = align_prediction_archives(reference, load_prediction_archive(second))
    np.testing.assert_equal(aligned.raw_targets, reference.raw_targets)
    np.testing.assert_array_equal(aligned.predictions, reference.predictions)
    assert np.isnan(aligned.raw_targets).any()
    assert (aligned.raw_targets < 0).any()


@pytest.mark.parametrize("fault,match", [
    ("duplicate", "Duplicate station-day"), ("key", "actual station/date"),
    ("resource", "Resource IDs"), ("date", "outside.*test window"),
    ("nonfinite", "finite values"), ("mask", "finite nonnegative"),
    ("shape", "real numeric array"), ("object", "Object arrays cannot be loaded"),
])
def test_archive_rejects_invalid_arrays(tmp_path, fault, match):
    payload = synthetic_payload()
    if fault == "duplicate":
        for name in ARRAY_FIELDS:
            payload[name][1] = payload[name][0]
    elif fault == "key":
        payload["sample_keys"][0] = "1:1900-01-01"
    elif fault == "resource":
        payload["resource_ids"][0] = 1
    elif fault == "date":
        payload["target_ordinals"][0] = fold_dates("summer", "train")[0].toordinal()
    elif fault == "nonfinite":
        payload["predictions"][0, 2] = np.nan
    elif fault == "mask":
        payload["valid_target_mask"][0, 2] = False
    elif fault == "shape":
        payload["targets"] = payload["targets"][:, :-1]
    else:
        payload["sample_keys"] = payload["sample_keys"].astype(object)
    path = tmp_path / "invalid.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=match):
        load_prediction_archive(path)


@pytest.mark.parametrize("metadata,match", [
    (None, "rerun prediction export"),
    ('{"model":"MSWP-Fuse","seed":NaN}', "strict JSON"),
    ('{"model":"MSWP-Fuse","model":"MSWP-Fuse"}', "strict JSON"),
])
def test_legacy_or_non_strict_identity_is_not_fabricated(tmp_path, metadata, match):
    payload = synthetic_payload()
    if metadata is None:
        del payload["metadata"]
    else:
        payload["metadata"] = np.asarray(metadata)
    path = tmp_path / "legacy.npz"
    np.savez(path, **payload)
    with pytest.raises(ValueError, match=match):
        load_prediction_archive(path)


@pytest.mark.parametrize("fault", ["targets", "raw_targets", "valid_target_mask", "sample_keys", "missing_row"])
def test_alignment_refuses_mismatch_instead_of_intersection(tmp_path, fault):
    path = tmp_path / "archive.npz"
    np.savez(path, **synthetic_payload())
    first = load_prediction_archive(path)
    if fault == "missing_row":
        other = first.take(np.arange(first.station_ids.size - 1))
    else:
        values = getattr(first, fault).copy()
        if fault == "sample_keys":
            values[0] = "incorrect-key"
        elif fault == "valid_target_mask":
            values[0, 2] = False
        else:
            values[0, 2] += 0.1
        other = replace(first, **{fault: values})
    with pytest.raises(ValueError, match="differ|disagree"):
        align_prediction_archives(first, other)


def test_paper_windows_and_complete_grid_are_required(tmp_path):
    directory = tmp_path / "incomplete"
    directory.mkdir()
    np.savez(directory / "summer.npz", **synthetic_payload())
    with pytest.raises(ValueError, match="All paper windows"):
        load_fold_archives(directory, model="MSWP-Fuse")
    for fold in fold_names()[1:]:
        np.savez(directory / f"{fold}.npz", **synthetic_payload(fold))
    with pytest.raises(ValueError, match="Incomplete summer"):
        load_fold_archives(directory, model="MSWP-Fuse")


def test_stability_recomputes_all_five_metrics_and_sample_sd_for_all_groups(cohort):
    runs, reference = cohort
    result = summarize_stability(runs, reference)
    assert result["all_six_meet_selection_condition"] is True
    assert len(result["groups"]) == 12  # pooled/window × portfolio/wind/solar
    loaded = [load_fold_archives(path, model="MSWP-Fuse") for path in runs]
    for group in result["groups"]:
        values = {name: [] for name in METRIC_NAMES}
        for folds in loaded:
            archive = pool_archives(list(folds.values())) if group["window"] == "pooled" else folds[group["window"]]
            selected = np.ones(archive.station_ids.size, dtype=bool)
            if group["resource"] != "portfolio":
                selected = archive.resource_ids == int(group["resource"] == "solar")
            archive = archive.take(selected)
            metrics = compute_metrics(
                archive.predictions, archive.raw_targets, archive.valid_target_mask,
                archive.station_ids, archive.resource_ids, archive.target_ordinals, archive.sample_keys,
            )
            for name in METRIC_NAMES:
                values[name].append(metrics[name])
        for name in METRIC_NAMES:
            stats = group["metrics"][name]
            assert stats["defined_instance_count"] == 6
            assert stats["mean"] == pytest.approx(np.mean(values[name]))
            assert stats["sample_sd"] == pytest.approx(np.std(values[name], ddof=1))
    # Output must not publish the caller's numeric seed identities.
    assert "seed" not in json.dumps(result)


@pytest.mark.parametrize("field", LINEAGE_FIELDS)
def test_stability_rejects_shared_actual_stage_identity(cohort, field):
    runs, reference = cohort
    shared = synthetic_metadata("summer")[field]
    rewrite_metadata(runs[1] / "summer.npz", lambda value: value.update({field: shared}))
    with pytest.raises(ValueError, match=f"share {field}"):
        summarize_stability(runs, reference)


@pytest.mark.parametrize("fault,match", [
    ("duplicate_seed", "six distinct seeds"), ("cross_window_seed", "same seed across"),
    ("stage_seed", "stage_seeds must equal"), ("missing_stage_seed", "rerun prediction export"),
])
def test_stability_enforces_six_distinct_coherent_three_stage_instances(cohort, fault, match):
    runs, reference = cohort
    if fault == "duplicate_seed":
        for fold in fold_names():
            rewrite_metadata(runs[1] / f"{fold}.npz", lambda value: value.update(seed=0, stage_seeds=dict.fromkeys(("initializer", "parent", "selected"), 0)))
    elif fault == "cross_window_seed":
        rewrite_metadata(runs[0] / "autumn.npz", lambda value: value.update(seed=1, stage_seeds=dict.fromkeys(("initializer", "parent", "selected"), 1)))
    elif fault == "stage_seed":
        rewrite_metadata(runs[0] / "summer.npz", lambda value: value["stage_seeds"].update(parent=1))
    else:
        rewrite_metadata(runs[0] / "summer.npz", lambda value: value.pop("stage_seeds"))
    with pytest.raises(ValueError, match=match):
        summarize_stability(runs, reference)


def test_stability_reports_failed_strict_selection_without_dropping_a_run(cohort):
    runs, reference = cohort
    write_fold_archives(runs[-1], len(runs) - 1, error=0.18)
    result = summarize_stability(runs, reference)
    assert result["all_six_meet_selection_condition"] is False
    assert len(result["instances"]) == 6
    assert result["instances"][-1]["pooled_c_above_fixed_reference"] is False
    assert all(group["instance_count"] == 6 for group in result["groups"])


def test_exactly_six_distinct_directories_are_required_before_reads():
    with patch("mswpfuse.stability_analysis.load_fold_archives") as load:
        with pytest.raises(ValueError, match="exactly six"):
            summarize_stability([Path("unused")] * 5, "reference")
        with pytest.raises(ValueError, match="directories must be distinct"):
            summarize_stability([Path("unused")] * 6, "reference")
        load.assert_not_called()


def test_empty_and_unscorable_groups_have_null_metrics(tmp_path):
    payload = synthetic_payload()
    path = tmp_path / "archive.npz"
    np.savez(path, **payload)
    archive = load_prediction_archive(path)
    empty = score_group(archive, np.zeros(archive.station_ids.size, dtype=bool))
    assert empty["station_day_count"] == 0
    assert empty["metrics"] == dict.fromkeys(METRIC_NAMES)
    invalid = replace(archive, raw_targets=np.full_like(archive.raw_targets, np.nan), valid_target_mask=np.zeros_like(archive.valid_target_mask))
    assert score_group(invalid)["scored_station_day_count"] == 0
    assert score_group(invalid)["metrics"] == dict.fromkeys(METRIC_NAMES)


def analysis_parser():
    parser = argparse.ArgumentParser()
    add_analysis_parsers(parser.add_subparsers(dest="command", required=True))
    return parser


@pytest.mark.parametrize("command", ["summarize-stability", "analyze-weather"])
def test_output_is_required_and_overwrite_is_refused_before_input_reads(tmp_path, command):
    parser = analysis_parser()
    argv = [command, "--runs", *[str(tmp_path / f"run_{index}") for index in range(6)], "--reference", "missing"] if command == "summarize-stability" else [command, "--data-root", "missing", "--full", "missing", "--head-only", "missing"]
    with pytest.raises(SystemExit):
        parser.parse_args(argv)
    output = tmp_path / "existing.json"
    output.write_text("preserved", encoding="utf-8")
    with patch("mswpfuse.analysis_cli.summarize_stability") as stability, patch("mswpfuse.analysis_cli.analyze_weather") as weather:
        with pytest.raises(FileExistsError):
            run_analysis(parser.parse_args([*argv, "--output", str(output)]))
        stability.assert_not_called()
        weather.assert_not_called()
    assert output.read_text(encoding="utf-8") == "preserved"


def test_strict_json_serialization_does_not_leave_an_invalid_output(tmp_path):
    output = tmp_path / "invalid.json"
    args = argparse.Namespace(command="summarize-stability", runs=[], reference="unused", output=output)
    report = {"instance_count": 6, "all_six_meet_selection_condition": False, "bad_metric": float("nan")}
    with patch("mswpfuse.analysis_cli.summarize_stability", return_value=report):
        with pytest.raises(ValueError, match="Out of range float values"):
            run_analysis(args)
    assert not output.exists()


@pytest.mark.parametrize("entry_module", ["mswpfuse.analysis_cli", "mswpfuse.cli"])
def test_real_npz_to_public_analysis_command_to_parsed_strict_json(cohort, tmp_path, entry_module):
    runs, reference = cohort
    output = tmp_path / "stability.json"
    command = [sys.executable, "-m", entry_module, "summarize-stability", "--runs", *map(str, runs), "--reference", str(reference), "--output", str(output)]
    completed = subprocess.run(command, cwd=tmp_path, env=os.environ.copy(), capture_output=True, text=True, timeout=60)
    assert completed.returncode == 0, completed.stderr
    stdout = json.loads(completed.stdout)
    assert stdout["instance_count"] == 6
    assert stdout["all_six_meet_selection_condition"] is True
    assert stdout["output"] == str(output)
    def reject_constant(value):
        raise AssertionError(f"Non-strict JSON constant: {value}")
    report = json.loads(output.read_text(encoding="utf-8"), parse_constant=reject_constant)
    assert len(report["groups"]) == 12
    assert report["sample_sd_ddof"] == 1
