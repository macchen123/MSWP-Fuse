"""Focused public unit tests for comparison baselines and validation stopping rules."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import unittest

import torch
from torch.utils.data import Dataset

from mswpfuse import baselines as b


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


def _pytorch_forecasting_root() -> Path | None:
    override = os.environ.get("MSWPFUSE_PYTORCH_FORECASTING_ROOT")
    if override:
        path = Path(override).expanduser().resolve()
        if path.is_dir():
            return path
    candidate = Path(__file__).resolve().parents[1] / "third_party" / "pytorch-forecasting"
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


class BaselinesIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tslib_root = _tslib_root()
        self.pt_root = _pytorch_forecasting_root()

    def test_baseline_registry_and_nwp_flags(self) -> None:
        names = b.baseline_names()
        self.assertEqual(len(names), 12)
        expected = [
            "Persistence", "DLinear", "LightGBM", "XGBoost", "NWPSeq2Seq",
            "TemporalFusionTransformer", "PatchTST", "iTransformer",
            "TimeMixer", "TimeXer", "WPMixer", "MultiPatchFormer"
        ]
        self.assertEqual(list(names), expected)

        p = b.build_neural_baseline("Persistence", torch.device("cpu"))
        self.assertFalse(p.uses_nwp)

        seq = b.build_neural_baseline("NWPSeq2Seq", torch.device("cpu"))
        self.assertTrue(seq.uses_nwp)

        if self.tslib_root is not None:
            tm = b.build_neural_baseline("TimeMixer", torch.device("cpu"), tslib_root=self.tslib_root)
            self.assertFalse(tm.uses_nwp)
            self.assertIsNone(tm.nwp_adapter)

            dl = b.build_neural_baseline("DLinear", torch.device("cpu"), tslib_root=self.tslib_root)
            self.assertTrue(dl.uses_nwp)
            self.assertIsNotNone(dl.nwp_adapter)

    def test_validation_dataset_required_for_trained_neural_baseline(self) -> None:
        train_ds = SyntheticDataset(n_samples=4)
        model = b.build_neural_baseline("NWPSeq2Seq", torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "requires validation_dataset"):
            b.train_neural_baseline("NWPSeq2Seq", model, train_ds, torch.device("cpu"), seed=42, validation_dataset=None)

    def test_nwpseq2seq_synthetic_train_with_validation(self) -> None:
        train_ds = SyntheticDataset(n_samples=8, seed=10)
        val_ds = SyntheticDataset(n_samples=4, seed=20)

        original_fn = b._configured_model

        def fast_config(name: str):
            cfg = copy.deepcopy(original_fn(name))
            if name == "NWPSeq2Seq":
                cfg["training"]["epochs"] = 2
                cfg["training"]["warmup_epochs"] = 1
                cfg["training"]["batch_size"] = 4
                cfg["training"]["patience"] = 2
            return cfg

        try:
            b._configured_model = fast_config
            model = b.build_neural_baseline("NWPSeq2Seq", torch.device("cpu"))
            history = b.train_neural_baseline(
                "NWPSeq2Seq",
                model,
                train_ds,
                torch.device("cpu"),
                seed=42,
                validation_dataset=val_ds,
            )
            self.assertEqual(len(history), 2)
            self.assertIn("val_c", history[0])
            self.assertIn("val_rmse", history[0])
            self.assertIn("val_mae", history[0])
        finally:
            b._configured_model = original_fn

    def test_dlinear_synthetic_train_with_validation(self) -> None:
        if self.tslib_root is None:
            self.skipTest("Time-Series-Library upstream not present; set MSWPFUSE_TSLIB_ROOT to run")

        train_ds = SyntheticDataset(n_samples=8, seed=30)
        val_ds = SyntheticDataset(n_samples=4, seed=40)

        original_fn = b._configured_model

        def fast_config(name: str):
            cfg = copy.deepcopy(original_fn(name))
            if name == "DLinear":
                cfg["training"]["epochs"] = 2
                cfg["training"]["batch_size"] = 4
                cfg["training"]["patience"] = 2
            return cfg

        try:
            b._configured_model = fast_config
            model = b.build_neural_baseline("DLinear", torch.device("cpu"), tslib_root=self.tslib_root)
            history = b.train_neural_baseline(
                "DLinear",
                model,
                train_ds,
                torch.device("cpu"),
                seed=42,
                validation_dataset=val_ds,
                tslib_root=self.tslib_root,
            )
            self.assertEqual(len(history), 2)
            self.assertIn("val_loss", history[0])
        finally:
            b._configured_model = original_fn

    def test_tft_synthetic_train_and_validation(self) -> None:
        if self.pt_root is None:
            self.skipTest("pytorch-forecasting upstream not present; set MSWPFUSE_PYTORCH_FORECASTING_ROOT to run")

        train_ds = SyntheticDataset(n_samples=4, seed=50)
        val_ds = SyntheticDataset(n_samples=2, seed=60)

        original_fn = b._configured_model

        def fast_config(name: str):
            cfg = copy.deepcopy(original_fn(name))
            if name == "TemporalFusionTransformer":
                cfg["training"]["epochs"] = 1
                cfg["training"]["batch_size"] = 2
                cfg["training"]["patience"] = 1
            return cfg

        try:
            b._configured_model = fast_config
            model = b.build_neural_baseline("TemporalFusionTransformer", torch.device("cpu"), pytorch_forecasting_root=self.pt_root)
            history = b.train_neural_baseline(
                "TemporalFusionTransformer",
                model,
                train_ds,
                torch.device("cpu"),
                seed=42,
                validation_dataset=val_ds,
                pytorch_forecasting_root=self.pt_root,
            )
            self.assertEqual(len(history), 1)
            self.assertIn("val_loss", history[0])
        finally:
            b._configured_model = original_fn


if __name__ == "__main__":
    unittest.main()
