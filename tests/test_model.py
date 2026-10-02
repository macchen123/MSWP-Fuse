"""Standalone public test suite for MSWP-Fuse models and history initializer."""

from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
import torch

from mswpfuse.initializer import HistoryInitializer
from mswpfuse.model import (
    AMENDMENT_PREFIXES,
    FORECAST_STEPS,
    HISTORY_STEPS,
    InputShapeError,
    MSWPFuse,
    MSWPFuseBackbone,
    MSWPFuseConfig,
    MSWPFuseSRATR,
)


@pytest.fixture
def scaler_stats() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    mean = torch.randn(27)
    std = torch.rand(27) + 0.1
    return mean, std


@pytest.fixture
def synthetic_batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(123)
    batch_size = 2
    return {
        "hist_power": torch.rand(batch_size, HISTORY_STEPS, 2),
        "hist_time": torch.rand(batch_size, HISTORY_STEPS, 5),
        "future_time": torch.rand(batch_size, FORECAST_STEPS, 5),
        "nwp": torch.randn(batch_size, FORECAST_STEPS, 27),
        "nwp_availability": torch.ones(batch_size, 3),
        "station_id": torch.tensor([1, 2]),
        "resource_id": torch.tensor([0, 1]),
    }


def test_config_dataclass_and_serialization() -> None:
    config = MSWPFuseConfig()
    config.validate()
    d = dataclasses.asdict(config)
    assert isinstance(d, dict)
    assert d["d_model"] == 128
    assert d["history_patch_length"] == 8
    assert d["temporal_hidden"] == 24
    assert d["interaction_rank"] == 12

    # Check from_mapping
    custom = MSWPFuseConfig.from_mapping({"temporal_hidden": 32, "interaction_rank": 16})
    assert custom.temporal_hidden == 32
    assert custom.interaction_rank == 16
    assert custom.d_model == 128
    custom.validate()

    # Alias check
    aliased = MSWPFuseConfig.from_mapping({"history_patch_len": 8})
    assert aliased.history_patch_length == 8

    # Invalid config checks
    with pytest.raises(ValueError, match="divisible by n_heads"):
        MSWPFuseConfig(d_model=127).validate()
    with pytest.raises(ValueError, match="history patches"):
        MSWPFuseConfig(history_patch_length=16).validate()


def test_history_initializer_topology_and_forward(synthetic_batch: dict[str, torch.Tensor]) -> None:
    model = HistoryInitializer()
    state = model.state_dict()
    assert len(state) == 80, f"Expected 80 tensors in history initializer, got {len(state)}"

    model.eval()
    with torch.no_grad():
        out = model(synthetic_batch)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, FORECAST_STEPS)
    assert torch.isfinite(out).all()


def test_backbone_topology_and_history_transfer(scaler_stats: tuple[torch.Tensor, torch.Tensor], synthetic_batch: dict[str, torch.Tensor]) -> None:
    mean, std = scaler_stats
    initializer = HistoryInitializer()

    backbone = MSWPFuseBackbone(
        nwp_mean=mean,
        nwp_std=std,
        history_state_dict=initializer.state_dict(),
    )
    state = backbone.state_dict()
    assert len(state) == 119, f"Expected 119 tensors in backbone, got {len(state)}"
    assert len(backbone.loaded_history_keys) == 29
    assert len(backbone.patch_history.state_dict()) == 29

    # Verify loaded tensors equal initializer history tensors
    init_state = initializer.state_dict()
    for key in backbone.loaded_history_keys:
        assert torch.equal(backbone.patch_history.state_dict()[key], init_state[key])

    # Dropout and penalty
    backbone.set_source_dropout(0.15)
    assert backbone._source_dropout_probability == 0.15
    penalty = backbone.station_delta_penalty()
    assert penalty.ndim == 0
    assert penalty >= 0.0

    # Forward
    backbone.eval()
    with torch.no_grad():
        out = backbone(synthetic_batch)
    assert out.shape == (2, FORECAST_STEPS)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize(
    "config_overrides",
    [
        {"temporal_hidden": 24, "interaction_rank": 12, "bounded_scale_maximum": 0.15, "initial_scale_fraction": 0.4125854203825821, "initial_inactive_bias": -6.0, "initial_correction_gate_bias": -4.0},
        {"temporal_hidden": 32, "interaction_rank": 16, "bounded_scale_maximum": 0.15, "initial_scale_fraction": 0.40, "initial_inactive_bias": -6.0, "initial_correction_gate_bias": -4.0},
        {"temporal_hidden": 24, "interaction_rank": 16, "bounded_scale_maximum": 0.20, "initial_scale_fraction": 0.25, "initial_inactive_bias": -4.0, "initial_correction_gate_bias": -2.0},
        {"temporal_hidden": 32, "interaction_rank": 12, "bounded_scale_maximum": 0.15, "initial_scale_fraction": 0.10, "initial_inactive_bias": -4.0, "initial_correction_gate_bias": -4.0},
        {"temporal_hidden": 16, "interaction_rank": 8, "bounded_scale_maximum": 0.20, "initial_scale_fraction": 0.40, "initial_inactive_bias": -5.0, "initial_correction_gate_bias": -2.0},
        {"temporal_hidden": 32, "interaction_rank": 16, "bounded_scale_maximum": 0.15, "initial_scale_fraction": 0.25, "initial_inactive_bias": -6.0, "initial_correction_gate_bias": -2.0},
    ],
)
def test_mswpfuse_candidate_configurations(
    scaler_stats: tuple[torch.Tensor, torch.Tensor],
    synthetic_batch: dict[str, torch.Tensor],
    config_overrides: dict[str, Any],
) -> None:
    mean, std = scaler_stats
    model = MSWPFuse(config_overrides, nwp_mean=mean, nwp_std=std)

    state = model.state_dict()
    assert len(state) == 155, f"Expected 155 state dict items, got {len(state)}"
    parent_names = model.canonical_m2_state_names()
    assert len(parent_names) == 119
    amendment_names = [name for name in state if any(name.startswith(p) for p in AMENDMENT_PREFIXES)]
    assert len(amendment_names) == 36

    model.eval()
    with torch.no_grad():
        parent_out = model.forward_parent(synthetic_batch)
        features = model.forward_features(synthetic_batch)
        out = model(synthetic_batch)

    assert parent_out.shape == (2, FORECAST_STEPS)
    assert out.shape == (2, FORECAST_STEPS)
    assert torch.equal(out, features["prediction"])
    assert "inactive_logit" in features
    assert "inactive_probability" in features
    assert "correction_gate" in features
    assert "temporal_delta" in features
    assert torch.isfinite(out).all()


def test_mswpfuse_differentiability(scaler_stats: tuple[torch.Tensor, torch.Tensor], synthetic_batch: dict[str, torch.Tensor]) -> None:
    mean, std = scaler_stats
    model = MSWPFuse(nwp_mean=mean, nwp_std=std)
    model.train()

    features = model.forward_features(synthetic_batch)
    loss = (
        features["prediction"].sum()
        + features["inactive_logit"].sum()
        + features["correction_gate"].sum()
    )
    loss.backward()

    # Check amendment gradients
    assert model.scale_logit.grad is not None
    assert model.inactive_head[0].weight.grad is not None
    assert model.correction_head.correction_gate[0].weight.grad is not None
    assert model.temporal_weather.input_projection.weight.grad is not None
    assert model.station_regime_interaction.output_projection.weight.grad is not None


def test_mswpfuse_alias() -> None:
    assert MSWPFuseSRATR is MSWPFuse


def test_input_shape_error(scaler_stats: tuple[torch.Tensor, torch.Tensor], synthetic_batch: dict[str, torch.Tensor]) -> None:
    mean, std = scaler_stats
    model = MSWPFuse(nwp_mean=mean, nwp_std=std)

    bad_batch = {k: v for k, v in synthetic_batch.items() if k != "hist_power"}
    with pytest.raises(InputShapeError, match="Missing required inputs"):
        model(bad_batch)

    bad_shape = {**synthetic_batch, "nwp": torch.randn(2, 50, 27)}
    with pytest.raises(InputShapeError, match="nwp must have shape"):
        model(bad_shape)
