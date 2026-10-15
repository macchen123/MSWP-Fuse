"""Paper-scoped CLI and actual checkpoint-to-prediction connections on synthetic data."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from mswpfuse.checkpoints import (
    _checkpoint_hash, _load_model_checkpoint, _write_model_checkpoint,
    checked_seed, checkpoint_seed, checkpoint_stage_seeds,
)
from mswpfuse.cli import build_parser, main
from mswpfuse.data import NWPScaler, load_paper_settings
from mswpfuse.initializer import HistoryInitializer
from mswpfuse.model import MSWPFuse
from mswpfuse.training_commands import _check_reported_training, _training_seed
from test_training import SyntheticDate


def _short_settings(fold=None, configuration=None):
    settings = load_paper_settings(fold, configuration)
    settings["initializer_training"].update(max_epochs=1, warmup_epochs=1)
    settings["parent_stage"].update(epochs=1)
    settings["selected_training"].update(max_epochs=1)
    return settings


@pytest.mark.parametrize("value", [True, -1, 2**32, 0.5, "7"])
def test_seed_is_a_valid_explicit_integer(value):
    with pytest.raises(ValueError, match="seed must"):
        checked_seed(value)


def test_stage_seed_lineage_is_not_fabricated_for_old_parents():
    legacy = {"stage": "parent", "training": {"settings": {}}}
    assert checkpoint_seed(legacy) is None
    assert checkpoint_stage_seeds(legacy) == {}
    with pytest.raises(ValueError, match="lacks complete stage seed identity"):
        _training_seed(None, legacy, {"initializer", "parent"})
    with pytest.raises(ValueError, match="inconsistent random seeds"):
        checkpoint_stage_seeds({"stage": "parent", "seed": 7, "stage_seeds": {"initializer": 8}})
    with pytest.raises(ValueError, match="disagrees with its training settings"):
        checkpoint_seed({"seed": 7, "training": {"settings": {"seed": 8}}})
    with pytest.raises(ValueError, match="future training stage"):
        checkpoint_stage_seeds({"stage": "initializer", "seed": 7, "stage_seeds": {"selected": 7}})


def test_unreported_seed_by_control_or_configuration_search_is_rejected():
    settings = load_paper_settings("summer")
    with pytest.raises(ValueError, match="fixed reference seed"):
        _check_reported_training(settings, "summer", "head_only", 7)
    with pytest.raises(ValueError, match="not another configuration search"):
        _check_reported_training(load_paper_settings("summer", 1), "summer", "full", 7)
    _check_reported_training(settings, "summer", "full", 7)


def test_only_paper_variants_are_public_choices():
    parser = build_parser()
    for variant in ("head_only", "without_history_multiscale", "without_source_aware_nwp_fusion"):
        args = parser.parse_args(["train-selected", "--fold", "summer", "--variant", variant,
                                  "--parent-checkpoint", "parent.pt", "--output", "selected.pt"])
        assert args.variant == variant and args.seed is None
    for variant in ("unsupported_variant", "stage1_only", "all_parameters"):
        with pytest.raises(SystemExit):
            parser.parse_args(["train-selected", "--fold", "summer", "--variant", variant,
                               "--parent-checkpoint", "parent.pt", "--output", "selected.pt"])
    args = parser.parse_args(["predict", "--fold", "summer", "--backbone", "--checkpoint", "parent.pt", "--output", "backbone.npz"])
    assert args.backbone is True


def test_mismatched_stage_seed_is_rejected_before_data_access(tmp_path):
    settings = load_paper_settings("summer")
    settings["initializer_training"]["seed"] = 7
    scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))
    checkpoint = tmp_path / "initializer.pt"
    _write_model_checkpoint(checkpoint, "initializer", "summer", HistoryInitializer(), scaler,
                            {"settings": settings["initializer_training"]}, settings)
    with patch("mswpfuse.training_commands.build_fold_datasets") as build:
        with pytest.raises(ValueError, match="same seed"):
            main(["train-parent", "--fold", "summer", "--device", "cpu", "--seed", "8",
                  "--initializer-checkpoint", str(checkpoint), "--output", str(tmp_path / "parent.pt")])
        build.assert_not_called()
    assert not (tmp_path / "parent.pt").exists()


def test_original_full_checkpoint_remains_readable(tmp_path):
    settings = load_paper_settings("summer")
    scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))
    model = MSWPFuse(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std)
    checkpoint = tmp_path / "current.pt"
    _write_model_checkpoint(checkpoint, "selected", "summer", model, scaler,
                            {"settings": settings["selected_training"]}, settings)
    payload = torch.load(checkpoint, weights_only=True)
    for name in ("seed", "stage_seeds", "variant"):
        payload.pop(name)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    restored, _, restored_payload = _load_model_checkpoint(legacy, torch.device("cpu"), expected_stage="selected", expected_fold="summer")
    assert checkpoint_seed(restored_payload) == settings["selected_training"]["seed"]
    assert checkpoint_stage_seeds(restored_payload) == {"selected": settings["selected_training"]["seed"]}
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items())


def test_actual_seeded_three_stage_cli_to_npz_metadata(tmp_path, capsys):
    torch.set_num_threads(2)
    dataset = SyntheticDate()
    scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))

    common = ["--fold", "summer", "--device", "cpu"]
    initializer, parent, selected = (tmp_path / name for name in ("initializer.pt", "parent.pt", "selected.pt"))
    prediction = tmp_path / "prediction.archive"
    with patch("mswpfuse.training_commands.load_paper_settings", side_effect=_short_settings), patch(
        "mswpfuse.training_commands.build_fold_datasets", return_value=({"train": dataset, "validation": dataset}, scaler),
    ), patch("mswpfuse.training_commands.build_split_dataset", return_value=dataset):
        assert main(["train-initializer", *common, "--seed", "7", "--output", str(initializer)]) == 0
        assert main(["train-parent", *common, "--initializer-checkpoint", str(initializer), "--output", str(parent)]) == 0
        assert main(["train-selected", *common, "--parent-checkpoint", str(parent), "--output", str(selected)]) == 0
        assert main(["predict", *common, "--checkpoint", str(selected), "--output", str(prediction)]) == 0
        with pytest.raises(ValueError, match="fixed-reference Full first-stage"):
            main(["predict", *common, "--backbone", "--checkpoint", str(parent), "--output", str(tmp_path / "backbone.npz")])
        assert not (tmp_path / "backbone.npz").exists()
    payload = torch.load(selected, weights_only=True)
    assert payload["seed"] == 7
    assert payload["stage_seeds"] == {"initializer": 7, "parent": 7, "selected": 7}
    assert payload["training"]["initializer_sha256"] == _checkpoint_hash(initializer)
    assert payload["training"]["parent_sha256"] == _checkpoint_hash(parent)
    assert prediction.is_file() and not Path(str(prediction) + ".npz").exists()
    with np.load(prediction, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        assert metadata["stage_seeds"] == payload["stage_seeds"]
        assert metadata["seed"] == 7 and metadata["variant"] == "full"
        assert metadata["checkpoint_sha256"] == _checkpoint_hash(selected)
        assert archive["predictions"].shape == (10, 96)
        assert np.isfinite(archive["predictions"]).all()
    outputs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert len(outputs) == 4 and outputs[-1]["variant"] == "full"


@pytest.mark.parametrize("variant", ["without_history_multiscale", "without_source_aware_nwp_fusion", "head_only"])
def test_actual_reported_control_cli_checkpoint_roundtrip(tmp_path, variant):
    torch.set_num_threads(2)
    dataset = SyntheticDate()
    scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))
    parent_variant = "full" if variant == "head_only" else variant
    initializer, parent, selected = (tmp_path / name for name in ("initializer.pt", "parent.pt", "selected.pt"))
    common = ["--fold", "summer", "--device", "cpu"]
    with patch("mswpfuse.training_commands.load_paper_settings", side_effect=_short_settings), patch(
        "mswpfuse.training_commands.build_fold_datasets", return_value=({"train": dataset, "validation": dataset}, scaler),
    ) as build, patch("mswpfuse.training_commands.build_split_dataset", return_value=dataset):
        main(["train-initializer", *common, "--output", str(initializer)])
        main(["train-parent", *common, "--variant", parent_variant, "--initializer-checkpoint", str(initializer), "--output", str(parent)])
        calls = build.call_count
        wrong_variant = "without_history_multiscale" if parent_variant == "full" else "full"
        with pytest.raises(ValueError, match="own variant's backbone"):
            main(["train-selected", *common, "--variant", wrong_variant, "--parent-checkpoint", str(parent), "--output", str(tmp_path / "wrong.pt")])
        assert build.call_count == calls and not (tmp_path / "wrong.pt").exists()
        main(["train-selected", *common, "--variant", variant, "--parent-checkpoint", str(parent), "--output", str(selected)])
        main(["predict", *common, "--checkpoint", str(selected), "--output", str(tmp_path / "predictions.npz")])
        if variant == "head_only":
            main(["predict", *common, "--backbone", "--checkpoint", str(parent), "--output", str(tmp_path / "backbone.npz")])
            with np.load(tmp_path / "backbone.npz", allow_pickle=False) as archive:
                assert json.loads(str(archive["metadata"]))["variant"] == "stage1_only"
    payload = torch.load(selected, weights_only=True)
    expected_seed = load_paper_settings("summer")["selected_training"]["seed"]
    assert payload["variant"] == variant and payload["seed"] == expected_seed
    assert payload["stage_seeds"] == dict.fromkeys(("initializer", "parent", "selected"), expected_seed)
    assert payload["training"]["parent_sha256"] == _checkpoint_hash(parent)
    with np.load(tmp_path / "predictions.npz", allow_pickle=False) as archive:
        assert json.loads(str(archive["metadata"]))["variant"] == variant
        assert np.isfinite(archive["predictions"]).all()
