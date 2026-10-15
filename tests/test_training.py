"""Configuration, selection, and checkpoint regression tests (no dataset needed)."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from mswpfuse.checkpoints import _load_model_checkpoint, _write_model_checkpoint
from mswpfuse.cli import build_parser, main
from mswpfuse.data import NWPScaler, build_fold_datasets, load_paper_settings
from mswpfuse.initializer import HistoryInitializer
from mswpfuse.model import AMENDMENT_PREFIXES, MSWPFuse, MSWPFuseBackbone
from mswpfuse.training import configure_selected_scope, parent_parameter_groups, select_validation_row, train_initializer, train_parent, train_selected


class SyntheticDate(torch.utils.data.Dataset):
    def __init__(self):
        generator = torch.Generator().manual_seed(7)
        self.refs = [SimpleNamespace(station_id=station, target_date=date(2024, 1, 8)) for station in range(1, 11)]
        self.samples = []
        for ref in self.refs:
            target = torch.rand(96, generator=generator)
            self.samples.append({
                "hist_power": torch.rand(672, 2, generator=generator),
                "hist_time": torch.rand(672, 5, generator=generator),
                "future_time": torch.rand(96, 5, generator=generator),
                "nwp": torch.rand(96, 27, generator=generator),
                "nwp_availability": torch.ones(3),
                "station_id": torch.tensor(ref.station_id - 1),
                "resource_id": torch.tensor(int(ref.station_id > 5)),
                "original_station_id": torch.tensor(ref.station_id),
                "target_ordinal": torch.tensor(ref.target_date.toordinal()),
                "target": target, "raw_target": target.clone(),
                "target_valid_mask": torch.ones(96, dtype=torch.bool),
                "sample_key": f"{ref.station_id}:2024-01-08",
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class TrainingContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_selected_configurations(self):
        summer = load_paper_settings("summer")
        self.assertEqual(summer["configuration"], 4)
        self.assertEqual(summer["model"]["temporal_hidden"], 32)
        self.assertEqual(summer["selected_training"]["learning_rate"], 0.0003)
        self.assertEqual(summer["selected_training"]["activity_threshold"], 0.02)
        for fold in ("autumn", "early_winter"):
            settings = load_paper_settings(fold)
            self.assertEqual(settings["configuration"], 1)
            self.assertEqual(settings["model"]["temporal_hidden"], 24)
            self.assertEqual(settings["selected_training"]["seed"], 42)
        self.assertEqual([load_paper_settings(configuration=c)["model"]["interaction_rank"] for c in range(1, 7)], [12, 16, 16, 12, 8, 16])
        self.assertEqual(load_paper_settings("summer", 1)["configuration"], 1)
        with self.assertRaises(ValueError):
            load_paper_settings(configuration=7)
        with self.assertRaises(KeyError):
            load_paper_settings("not_a_fold")

    def test_selection_uses_global_maximum_band(self):
        rows = [
            {"c": 0.8, "rmse": 0.3, "mae": 0.2, "epoch": 1, "configuration": 1},
            {"c": 0.79991, "rmse": 0.2, "mae": 0.1, "epoch": 2, "configuration": 2},
            {"c": 0.79982, "rmse": 0.1, "mae": 0.05, "epoch": 3, "configuration": 3},
        ]
        self.assertIs(select_validation_row(rows), rows[1])
        for ordering in (rows[::-1], [rows[1], rows[2], rows[0]]):
            self.assertIs(select_validation_row(ordering), rows[1])
        with self.assertRaises(ValueError):
            select_validation_row([{**rows[0], "c": float("nan")}])
        with self.assertRaises(ValueError):
            select_validation_row([])

    def test_selection_ties_use_epoch_then_configuration(self):
        base = {"c": 0.7, "rmse": 0.1, "mae": 0.05}
        rows = [{**base, "epoch": 3, "configuration": 1}, {**base, "epoch": 2, "configuration": 4}, {**base, "epoch": 2, "configuration": 2}]
        self.assertIs(select_validation_row(rows), rows[2])

    def test_training_does_not_construct_test_samples(self):
        scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))
        with patch("mswpfuse.data.RawDataStore"), patch("mswpfuse.data.build_references", return_value=[]) as refs, patch("mswpfuse.data.fit_nwp_scaler", return_value=scaler):
            datasets, _ = build_fold_datasets("unused", "summer")
        self.assertEqual(set(datasets), {"train", "validation"})
        self.assertEqual(refs.call_count, 2)

    def test_parent_parameter_groups_preserve_transfer_rates(self):
        settings = load_paper_settings("summer")
        initializer = HistoryInitializer()
        model = MSWPFuseBackbone(settings["model"], nwp_mean=torch.zeros(27), nwp_std=torch.ones(27), history_state_dict=initializer.state_dict())
        groups = parent_parameter_groups(model, settings["parent_stage"])
        assignments = {id(parameter): group for group in groups for parameter in group["params"]}
        self.assertEqual(len(assignments), len(list(model.parameters())))
        for name, parameter in model.named_parameters():
            group = assignments[id(parameter)]
            if name == "conditioner.station_delta.weight":
                self.assertEqual((group["lr"], group["weight_decay"]), (1e-4, 0.0))
            elif name.startswith("patch_history."):
                self.assertEqual(group["lr"], 5e-5)

    def test_checkpoint_roundtrip_preserves_configuration_and_scope(self):
        scaler = NWPScaler(np.zeros(27, np.float32), np.ones(27, np.float32))
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            for fold in ("summer", "autumn"):
                settings = load_paper_settings(fold)
                model = MSWPFuse(settings["model"], nwp_mean=scaler.mean, nwp_std=scaler.std)
                configure_selected_scope(model, settings["selected_training"])
                path = Path(temporary) / f"{fold}.pt"
                _write_model_checkpoint(path, "selected", fold, model, scaler, {}, settings)
                restored, _, payload = _load_model_checkpoint(path, torch.device("cpu"), expected_stage="selected", expected_fold=fold)
                self.assertEqual(payload["configuration"], settings["configuration"])
                self.assertEqual({name: value.requires_grad for name, value in restored.named_parameters()}, {name: value.requires_grad for name, value in model.named_parameters()})
                self.assertTrue(all(torch.equal(value, restored.state_dict()[name]) for name, value in model.state_dict().items()))
                with self.assertRaises(ValueError):
                    _load_model_checkpoint(path, torch.device("cpu"), expected_stage="parent", expected_fold=fold)
                with self.assertRaises(FileExistsError):
                    _write_model_checkpoint(path, "selected", fold, model, scaler, {}, settings)

    def test_existing_output_is_rejected_before_reading_data(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            path = Path(temporary) / "existing.pt"
            path.write_bytes(b"existing")
            with patch("mswpfuse.training_commands.build_fold_datasets") as build:
                with self.assertRaises(FileExistsError):
                    main(["train-initializer", "--fold", "summer", "--output", str(path)])
                build.assert_not_called()
            self.assertEqual(path.read_bytes(), b"existing")

    def test_synthetic_three_stage_training_chain(self):
        settings = load_paper_settings("summer")
        dataset = SyntheticDate()
        device = torch.device("cpu")
        initializer = HistoryInitializer()
        init_outcome = train_initializer(initializer, dataset, dataset, {**settings["initializer_training"], "max_epochs": 1, "warmup_epochs": 1}, device)
        self.assertEqual(init_outcome.best_epoch, 1)
        parent = MSWPFuseBackbone(settings["model"], nwp_mean=torch.zeros(27), nwp_std=torch.ones(27), history_state_dict=initializer.state_dict())
        before_history = {name: value.clone() for name, value in parent.patch_history.state_dict().items()}
        parent_outcome = train_parent(parent, dataset, settings["selected_training"], {**settings["parent_stage"], "epochs": 1}, device)
        self.assertEqual(parent_outcome.best_epoch, 1)
        self.assertTrue(all(torch.equal(value, parent.patch_history.state_dict()[name]) for name, value in before_history.items()))
        model = MSWPFuse(settings["model"], nwp_mean=torch.zeros(27), nwp_std=torch.ones(27))
        model.load_state_dict(parent.state_dict(), strict=False)
        frozen = {name: value.clone() for name, value in model.named_parameters() if not name.startswith(AMENDMENT_PREFIXES + ("head.",))}
        outcome = train_selected(model, dataset, dataset, {**settings["selected_training"], "max_epochs": 1}, device)
        self.assertEqual(outcome.best_epoch, 1)
        self.assertTrue(np.isfinite([outcome.best_c, outcome.best_mae, outcome.best_rmse]).all())
        parameters = dict(model.named_parameters())
        self.assertTrue(all(torch.equal(value, parameters[name]) for name, value in frozen.items()))

    def test_cli_exposes_configuration_and_initializer(self):
        parser = build_parser()
        args = parser.parse_args(["train-selected", "--fold", "summer", "--configuration", "2", "--parent-checkpoint", "parent.pt", "--output", "selected.pt"])
        self.assertEqual(args.configuration, 2)
        self.assertEqual(args.parent_checkpoint, Path("parent.pt"))


if __name__ == "__main__":
    unittest.main()
