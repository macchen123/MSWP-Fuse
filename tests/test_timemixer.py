"""Focused public unit tests for the source-faithful tuned TimeMixer baseline."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.utils.data import Dataset

from mswpfuse import baselines as b
from mswpfuse.data import NWPScaler


def _tslib_root() -> Path | None:
    override = os.environ.get("MSWPFUSE_TSLIB_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_dir():
            return path
    candidate = Path(__file__).resolve().parents[1] / "third_party" / "Time-Series-Library"
    if candidate.is_dir():
        return candidate
    return None


class SyntheticDataset(Dataset):
    def __init__(self, n_samples: int = 8, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        self.samples = []
        for index in range(n_samples):
            self.samples.append(
                {
                    "hist_power": torch.randn(672, 2),
                    "hist_time": torch.randn(672, 5),
                    "future_time": torch.randn(96, 5),
                    "nwp": torch.randn(96, 27),
                    "nwp_availability": torch.ones(3),
                    "station_id": torch.tensor(index % 10),
                    "original_station_id": torch.tensor((index % 10) + 1),
                    "resource_id": torch.tensor(index % 2),
                    "target": torch.randn(96),
                    "raw_target": torch.randn(96).abs(),
                    "target_valid_mask": torch.ones(96, dtype=torch.bool),
                    "target_ordinal": torch.tensor(738890 + (index // 10)),
                    "sample_key": f"sample_{index}",
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        return self.samples[idx]


class TimeMixerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tslib_root = _tslib_root()

    def _require_tslib(self) -> Path:
        if self.tslib_root is None:
            self.skipTest("Time-Series-Library upstream not present; set MSWPFUSE_TSLIB_ROOT to run")
        return self.tslib_root

    def test_configuration_matches_accepted_r6(self) -> None:
        settings = b.load_baseline_settings()
        self.assertIn("TimeMixer", settings["models"])
        config = settings["models"]["TimeMixer"]
        self.assertEqual(config["display_name"], "TimeMixer")
        self.assertEqual(config["kind"], "tslib")

        arch = config["architecture"]
        self.assertEqual(arch["d_model"], 128)
        self.assertEqual(arch["d_ff"], 512)
        self.assertEqual(arch["e_layers"], 2)
        self.assertEqual(arch["n_heads"], 4)
        self.assertEqual(arch["seq_len"], 672)
        self.assertEqual(arch["label_len"], 48)
        self.assertEqual(arch["pred_len"], 96)
        self.assertFalse(arch["use_nwp_adapter"])

        training = config["training"]
        self.assertEqual(training["optimizer"], "AdamW")
        self.assertEqual(training["learning_rate"], 0.0003)
        self.assertEqual(training["weight_decay"], 0.0005)
        self.assertEqual(tuple(training["betas"]), (0.9, 0.95))
        self.assertEqual(training["schedule"], "cosine")
        self.assertEqual(training["warmup_epochs"], 8)
        self.assertEqual(training["epochs"], 120)
        self.assertEqual(training["patience"], 15)
        self.assertEqual(training["monitor"], "validation_mae")

    def test_model_construction_and_identity(self) -> None:
        tslib = self._require_tslib()
        model = b.build_neural_baseline("TimeMixer", torch.device("cpu"), tslib_root=tslib)
        self.assertFalse(model.uses_nwp)
        self.assertIsNone(model.nwp_adapter)

        state = model.state_dict()
        self.assertEqual(len(state), 61)
        self.assertTrue(all(key.startswith("model.") for key in state.keys()))

    def test_synthetic_forward_and_no_nwp_dependency(self) -> None:
        tslib = self._require_tslib()
        model = b.build_neural_baseline("TimeMixer", torch.device("cpu"), tslib_root=tslib)
        model.eval()

        batch = {
            "hist_power": torch.randn(2, 672, 2),
            "hist_time": torch.randn(2, 672, 5),
            "future_time": torch.randn(2, 96, 5),
            "nwp": torch.randn(2, 96, 27),
            "nwp_availability": torch.ones(2, 3),
        }

        with torch.no_grad():
            out1 = model(batch)
            self.assertEqual(tuple(out1.shape), (2, 96))

            # Modify NWP and availability completely; TimeMixer must be history+calendar only
            altered_batch = dict(batch)
            altered_batch["nwp"] = torch.randn(2, 96, 27) * 50.0
            altered_batch["nwp_availability"] = torch.zeros(2, 3)

            out2 = model(altered_batch)
            torch.testing.assert_close(out1, out2)

    def test_synthetic_gradient_step(self) -> None:
        tslib = self._require_tslib()
        model = b.build_neural_baseline("TimeMixer", torch.device("cpu"), tslib_root=tslib)
        model.train()
        batch = {
            "hist_power": torch.randn(2, 672, 2),
            "hist_time": torch.randn(2, 672, 5),
            "future_time": torch.randn(2, 96, 5),
            "nwp": torch.randn(2, 96, 27),
            "nwp_availability": torch.ones(2, 3),
            "target": torch.randn(2, 96),
            "target_valid_mask": torch.ones(2, 96, dtype=torch.bool),
        }

        output = model(batch)
        loss = ((output - batch["target"]) ** 2).mean()
        loss.backward()

        trainable_with_grad = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        self.assertGreater(len(trainable_with_grad), 50)
        for p in trainable_with_grad:
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_checkpoint_persistence_roundtrip(self) -> None:
        tslib = self._require_tslib()
        model = b.build_neural_baseline("TimeMixer", torch.device("cpu"), tslib_root=tslib)
        run = b.BaselineRun("TimeMixer", "tslib", model, ({"epoch": 1, "train_loss": 0.05, "val_mae": 0.04},))
        scaler = NWPScaler(mean=np.zeros(27, dtype=np.float32), std=np.ones(27, dtype=np.float32))

        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp_dir:
            path = Path(tmp_dir) / "test_timemixer.pt"
            b.save_baseline(path, run, scaler)
            self.assertTrue(path.exists())

            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload.get("float32_matmul_precision"), "high")
            self.assertIn("scaler", payload)

            loaded_model, loaded_scaler, kind = b.load_baseline(path, "TimeMixer", torch.device("cpu"), tslib_root=tslib)
            self.assertEqual(kind, "tslib")
            self.assertFalse(loaded_model.uses_nwp)
            self.assertEqual(len(loaded_model.state_dict()), 61)

            orig_dict = model.state_dict()
            loaded_dict = loaded_model.state_dict()
            self.assertEqual(set(orig_dict.keys()), set(loaded_dict.keys()))
            for key in orig_dict:
                torch.testing.assert_close(orig_dict[key], loaded_dict[key])

    def test_unmasked_processed_target_validation_selection(self) -> None:
        tslib = self._require_tslib()
        train_ds = SyntheticDataset(n_samples=8, seed=1)
        val_ds = SyntheticDataset(n_samples=4, seed=2)

        original_fn = b._configured_model

        def fast_config(name: str):
            cfg = copy.deepcopy(original_fn(name))
            if name == "TimeMixer":
                cfg["training"]["epochs"] = 3
                cfg["training"]["warmup_epochs"] = 1
                cfg["training"]["batch_size"] = 4
                cfg["training"]["patience"] = 2
            return cfg

        try:
            b._configured_model = fast_config
            model = b.build_neural_baseline("TimeMixer", torch.device("cpu"), tslib_root=tslib)
            history = b.train_neural_baseline(
                "TimeMixer",
                model,
                train_ds,
                torch.device("cpu"),
                seed=42,
                validation_dataset=val_ds,
                tslib_root=tslib,
            )
            self.assertGreater(len(history), 0)
            self.assertIn("val_mae", history[0])
            self.assertIn("val_rmse", history[0])
        finally:
            b._configured_model = original_fn


if __name__ == "__main__":
    unittest.main()
