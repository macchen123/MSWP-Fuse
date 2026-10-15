"""Weather covariates and matched full-day contrasts using synthetic local inputs."""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import json
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from mswpfuse.analysis_cli import add_analysis_parsers, run_analysis
from mswpfuse.data import CHANNELS, NWPScaler, RawDataStore, SampleRef, fold_dates, fold_names, load_paper_settings
from mswpfuse.metrics import compute_metrics
from mswpfuse.prediction_archives import METRIC_NAMES, load_fold_archives, score_group
from mswpfuse.weather_analysis import (
    COVARIATES, TERTILES, analyze_weather, assign_tertiles, training_weather_boundaries, weather_covariates,
)
from test_prediction_analysis import rewrite_metadata, synthetic_metadata, write_fold_archives


class SyntheticNWPStore:
    def __init__(self, root=None):
        self.root = root

    def nwp_path(self, ref, source):
        return SimpleNamespace(is_file=lambda: True)

    def nwp(self, ref):
        positions = np.arange(96, dtype=np.float32)[:, None, None]
        sources = np.arange(3, dtype=np.float32)[None, :, None]
        channels = np.arange(9, dtype=np.float32)[None, None, :]
        day = ref.target_date.toordinal() % 17
        raw = positions * 0.01 + sources * (0.3 + day * 0.02) + channels + ref.station_id * 0.02 + day * 0.1
        raw[:, :, CHANNELS.index("wind_speed")] = np.hypot(raw[:, :, CHANNELS.index("u100")], raw[:, :, CHANNELS.index("v100")])
        availability = np.asarray([True, True, True])
        if ref.station_id == 1 and ref.target_date.day % 7 == 0:
            availability[1:] = False
            raw[:, 1:] = 0.0
        return raw.reshape(96, 27), availability

    def history_target(self, ref):
        raise AssertionError("Weather covariates must not read power targets")


def test_physical_covariates_and_seven_channel_population_disagreement():
    raw = np.arange(96 * 3 * 9, dtype=np.float32).reshape(96, 3, 9) / 100.0
    raw[:, 0, CHANNELS.index("wind_speed")] = 2.0
    raw[:, 2, CHANNELS.index("wind_speed")] = 8.0
    raw[:, 1] = 1.0e6  # An unavailable source must never affect any covariate.
    availability = np.asarray([True, False, True])
    scaler = NWPScaler(np.arange(27, dtype=np.float32) * 0.1, np.linspace(1.0, 2.0, 27, dtype=np.float32))
    store = SimpleNamespace(nwp=lambda ref: (raw.reshape(96, 27), availability))
    wind = weather_covariates(store, SampleRef(1, date(2024, 7, 1)), scaler)
    solar = weather_covariates(store, SampleRef(6, date(2024, 7, 1)), scaler)
    assert wind["wind_speed"] == 5.0
    assert wind["solar_daytime_poai"] is None
    assert solar["wind_speed"] is None
    assert solar["solar_daytime_poai"] == pytest.approx(np.mean(raw[24:72, [0, 2], CHANNELS.index("poai")], dtype=np.float64))
    standardized = scaler.transform(raw.reshape(96, 27), availability).reshape(96, 3, 9)
    indices = [CHANNELS.index(name) for name in ("ghi", "poai", "t2m", "tcc", "u100", "v100", "wind_speed")]
    expected = np.std(standardized[:, [0, 2], :][:, :, indices], axis=1, ddof=0, dtype=np.float64).mean()
    assert wind["source_disagreement"] == pytest.approx(expected)
    assert solar["source_disagreement"] == pytest.approx(expected)
    availability[2] = False
    assert weather_covariates(store, SampleRef(1, date(2024, 7, 1)), scaler)["source_disagreement"] is None


def test_tertile_boundaries_have_exact_reported_inclusion_and_keep_undefined():
    values = np.asarray([0.0, 1.0, 1.5, 2.0, 3.0, np.nan])
    np.testing.assert_array_equal(assign_tertiles(values, 1.0, 2.0), [0, 0, 1, 1, 2, -1])
    np.testing.assert_array_equal(assign_tertiles(values, 1.0, 1.0), [0, 0, 2, 2, 2, -1])
    np.testing.assert_array_equal(assign_tertiles(values, None, None), [-1] * values.size)
    with pytest.raises(ValueError, match="finite and ordered"):
        assign_tertiles(values, 2.0, 1.0)


def test_boundaries_and_scaler_use_only_train_refs_and_linear_quantiles():
    start = fold_dates("summer", "train")[0]
    refs = [SampleRef(station, start + timedelta(days=day)) for day in range(4) for station in (1, 6)]
    store = SyntheticNWPStore()
    with patch("mswpfuse.weather_analysis.build_references", return_value=refs) as build:
        scaler, boundaries = training_weather_boundaries(store, "summer")
    build.assert_called_once_with(*fold_dates("summer", "train"), store=store)
    observed = [weather_covariates(store, ref, scaler) for ref in refs]
    for name in COVARIATES:
        values = [row[name] for row in observed if row[name] is not None]
        assert boundaries[name]["training_station_day_count"] == len(values)
        np.testing.assert_allclose([boundaries[name]["q1"], boundaries[name]["q2"]], np.quantile(values, [1 / 3, 2 / 3], method="linear"))


def test_no_two_source_training_days_produce_explicit_undefined_disagreement():
    start = fold_dates("summer", "train")[0]
    refs = [SampleRef(station, start + timedelta(days=day)) for day in range(3) for station in (1, 6)]
    def nwp(ref):
        source = (ref.target_date - start).days
        availability = np.arange(3) == source
        raw = np.ones((96, 3, 9), np.float32) * (source + ref.station_id)
        raw[:, ~availability] = 0.0
        return raw.reshape(96, 27), availability
    store = SimpleNamespace(nwp=nwp)
    with patch("mswpfuse.weather_analysis.build_references", return_value=refs):
        _, boundaries = training_weather_boundaries(store, "summer")
    assert boundaries["source_disagreement"] == {"q1": None, "q2": None, "training_station_day_count": 0}


def test_real_local_netcdf_uses_existing_raw_store_and_available_sources(tmp_path):
    root = tmp_path / "raw_data"
    ref = SampleRef(6, fold_dates("summer", "test")[0])
    store = RawDataStore(root)
    mapping = load_paper_settings()["data"]["nwp"]["channel_mapping"]
    for source, poai, u, v in (("NWP_1", 64.0, 3.0, 4.0), ("NWP_3", 128.0, 0.0, 10.0)):
        labels = [mapping[source][name] for name in CHANNELS[:-1]]
        data = np.ones((1, 24, 8, 3, 3), dtype=np.float32)
        data[:, :, CHANNELS.index("poai")] = poai
        data[:, :, CHANNELS.index("u100")] = u
        data[:, :, CHANNELS.index("v100")] = v
        path = store.nwp_path(ref, source)
        path.parent.mkdir(parents=True, exist_ok=True)
        xr.Dataset(
            {"data": (("time", "lead_time", "channel", "lat", "lon"), data)},
            coords={"time": [0], "lead_time": np.arange(24), "channel": labels, "lat": np.arange(3), "lon": np.arange(3)},
        ).to_netcdf(path)
    scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))
    solar = weather_covariates(store, ref, scaler)
    assert solar["solar_daytime_poai"] == 96.0
    assert solar["source_disagreement"] is not None
    # Same real loader, with the station changed only to evaluate the wind definition.
    wind_store = SimpleNamespace(nwp=lambda ignored: store.nwp(ref))
    wind = weather_covariates(wind_store, SampleRef(1, ref.target_date), scaler)
    assert wind["wind_speed"] == 7.5


@pytest.fixture
def weather_pair(tmp_path):
    full = write_fold_archives(tmp_path / "full", error=0.02)
    head = write_fold_archives(tmp_path / "head", error=0.06, variant="head_only")
    return full, head


def test_weather_command_writes_exactly_27_matched_full_day_five_metric_contrasts(weather_pair, tmp_path):
    full, head = weather_pair
    output = tmp_path / "weather.json"
    parser = argparse.ArgumentParser()
    add_analysis_parsers(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(["analyze-weather", "--data-root", str(tmp_path / "synthetic"), "--full", str(full), "--head-only", str(head), "--output", str(output)])
    with patch("mswpfuse.weather_analysis.RawDataStore", SyntheticNWPStore):
        result = run_analysis(args)
    assert result["group_count"] == 27
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["groups"]) == 27
    assert {(row["covariate"], row["window"], row["tertile"]) for row in report["groups"]} == {
        (name, fold, tertile) for name in COVARIATES for fold in fold_names() for tertile in TERTILES
    }
    folds = load_fold_archives(full, model="MSWP-Fuse")
    head_folds = load_fold_archives(head, model="MSWP-Fuse")
    store = SyntheticNWPStore()
    for fold in fold_names():
        scaler, boundaries = training_weather_boundaries(store, fold)
        archive = folds[fold]
        observations = [weather_covariates(store, SampleRef(int(station), date.fromordinal(int(ordinal))), scaler) for station, ordinal in zip(archive.station_ids, archive.target_ordinals, strict=True)]
        for row in (group for group in report["groups"] if group["window"] == fold):
            name = row["covariate"]
            values = np.asarray([item[name] if item[name] is not None else np.nan for item in observations])
            selected = assign_tertiles(values, boundaries[name]["q1"], boundaries[name]["q2"]) == TERTILES.index(row["tertile"])
            assert row["matched_station_day_count"] == int(selected.sum())
            expected_full = score_group(archive, selected)
            expected_head = score_group(head_folds[fold], selected)
            for metric in METRIC_NAMES:
                assert row["full"][metric] == pytest.approx(expected_full["metrics"][metric])
                assert row["head_only"][metric] == pytest.approx(expected_head["metrics"][metric])
                assert row["full_minus_head_only"][metric] == pytest.approx(row["full"][metric] - row["head_only"][metric])
        assert report["coverage"][fold]["source_disagreement"]["undefined_station_day_count"] > 0
        for name, resource in (("wind_speed", 0), ("solar_daytime_poai", 1)):
            rows = [row for row in report["groups"] if row["window"] == fold and row["covariate"] == name]
            assert sum(row["matched_station_day_count"] for row in rows) == int((archive.resource_ids == resource).sum())


def test_empty_weather_tertiles_stay_null_instead_of_zero(weather_pair):
    full, head = weather_pair
    scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))
    boundaries = {name: {"q1": None, "q2": None, "training_station_day_count": 0} for name in COVARIATES}
    with patch("mswpfuse.weather_analysis.RawDataStore", SyntheticNWPStore), patch("mswpfuse.weather_analysis.training_weather_boundaries", return_value=(scaler, boundaries)):
        report = analyze_weather("synthetic", full, head)
    assert len(report["groups"]) == 27
    for row in report["groups"]:
        assert row["matched_station_day_count"] == 0
        assert row["full"] == dict.fromkeys(METRIC_NAMES)
        assert row["head_only"] == dict.fromkeys(METRIC_NAMES)
        assert row["full_minus_head_only"] == dict.fromkeys(METRIC_NAMES)


@pytest.mark.parametrize("field", ["initializer_sha256", "parent_sha256"])
def test_unrelated_full_and_head_only_models_are_rejected_before_weather_reads(weather_pair, field):
    full, head = weather_pair
    rewrite_metadata(head / "summer.npz", lambda value: value.update({field: synthetic_metadata("summer", instance=1)[field]}))
    with patch("mswpfuse.weather_analysis.RawDataStore") as store:
        with pytest.raises(ValueError, match=f"same {field}"):
            analyze_weather("unused", full, head)
        store.assert_not_called()


@pytest.mark.parametrize("missing_lineage", [False, True])
def test_weather_requires_matching_complete_stage_seeds(weather_pair, missing_lineage):
    full, head = weather_pair
    def mutate(value):
        if missing_lineage:
            value.pop("stage_seeds")
        else:
            value["seed"] += 1
            value["stage_seeds"] = dict.fromkeys(value["stage_seeds"], value["seed"])
    rewrite_metadata(head / "summer.npz", mutate)
    with patch("mswpfuse.weather_analysis.RawDataStore") as store:
        with pytest.raises(ValueError, match="stage seed identity|same training seed"):
            analyze_weather("unused", full, head)
        store.assert_not_called()


def test_pv_nights_remain_in_the_full_day_score(weather_pair):
    full, _ = weather_pair
    archive = load_fold_archives(full, model="MSWP-Fuse")["summer"]
    selected = archive.resource_ids == 1
    solar = archive.take(selected)
    predicted = solar.predictions.copy()
    predicted[:, :24] = 0.2
    predicted[:, 72:] = 0.2
    from dataclasses import replace
    solar = replace(solar, predictions=predicted)
    group = score_group(solar)
    assert group["valid_target_count"] == solar.station_ids.size * 96
    assert group["metrics"]["mae"] == pytest.approx(0.11)
    expected = compute_metrics(predicted, solar.raw_targets, solar.valid_target_mask, solar.station_ids, solar.resource_ids, solar.target_ordinals, solar.sample_keys)
    assert group["metrics"]["c"] == expected["c"]
