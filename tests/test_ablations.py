"""Synthetic CPU checks for the paper-reported module and output-head controls.

The one-epoch connections below are software smoke tests, not paper runs.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from mswpfuse.ablations import (
    BACKBONE_VARIANTS,
    SELECTED_VARIANTS,
    BasicHistoryProjection,
    FixedAlignedFeatureFusion,
    build_backbone,
    build_selected,
)
from mswpfuse.data import load_paper_settings
from mswpfuse.initializer import HistoryInitializer
from mswpfuse.losses import CompositeForecastLoss, valid_mask
from mswpfuse.model import AMENDMENT_PREFIXES, AlignedFeatureFusion, InputShapeError, MSWPFuse, MSWPFuseBackbone
from mswpfuse.primitives import POAI_INDEX
from mswpfuse.training import (
    _selected_loss,
    _set_learning_rates,
    collect_predictions,
    configure_selected_scope,
    cpu_state_dict,
    selected_gradient_balance,
    train_parent,
    train_selected,
)


class SyntheticStationDate(torch.utils.data.Dataset):
    """One artificial complete date; no real data or experiment seed cohort."""

    def __init__(self):
        generator = torch.Generator().manual_seed(42)
        self.refs = [SimpleNamespace(station_id=station, target_date=date(2024, 1, 8)) for station in range(1, 11)]
        self.samples = []
        for ref in self.refs:
            target = 0.1 + 0.8 * torch.rand(96, generator=generator)
            mask = torch.ones(96, dtype=torch.bool)
            mask[:3] = False
            self.samples.append({
                "hist_power": torch.rand(672, 2, generator=generator),
                "hist_time": torch.rand(672, 5, generator=generator),
                "future_time": torch.rand(96, 5, generator=generator),
                "nwp": torch.rand(96, 27, generator=generator),
                "nwp_availability": torch.tensor([1.0, 0.0, 1.0]),
                "station_id": torch.tensor(ref.station_id - 1),
                "resource_id": torch.tensor(int(ref.station_id > 5)),
                "original_station_id": torch.tensor(ref.station_id),
                "target_ordinal": torch.tensor(ref.target_date.toordinal()),
                "target": target,
                "raw_target": target.clone(),
                "target_valid_mask": mask,
                "sample_key": f"synthetic:{ref.station_id}:{ref.target_date}",
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def stats():
    return {"nwp_mean": torch.zeros(27), "nwp_std": torch.ones(27)}


@pytest.fixture
def dataset():
    return SyntheticStationDate()


@pytest.fixture
def batch(dataset):
    names = ("hist_power", "hist_time", "future_time", "nwp", "nwp_availability", "station_id", "resource_id")
    return {name: torch.stack([dataset[index][name] for index in (0, 5)]) for name in names}


def test_exact_variant_whitelists():
    assert BACKBONE_VARIANTS == ("full", "without_history_multiscale", "without_source_aware_nwp_fusion")
    assert SELECTED_VARIANTS == (*BACKBONE_VARIANTS, "head_only")


@pytest.mark.parametrize("factory", [build_backbone, build_selected])
@pytest.mark.parametrize("variant", ["stage1_only", "unreported_control", "Full", ""])
def test_unsupported_variants_fail_before_construction(factory, variant, stats):
    before = torch.get_rng_state()
    with pytest.raises(ValueError, match="Unsupported"):
        factory(None, variant=variant, **stats)
    assert torch.equal(before, torch.get_rng_state())


def test_head_only_has_no_separate_backbone_fit(stats):
    with pytest.raises(ValueError, match="Unsupported backbone variant"):
        build_backbone(None, variant="head_only", **stats)


@pytest.mark.parametrize("factory,original", [(build_backbone, MSWPFuseBackbone), (build_selected, MSWPFuse)])
def test_full_factory_preserves_type_topology_rng_and_outputs(factory, original, stats, batch):
    torch.manual_seed(42)  # Synthetic regression fixture, not an additional paper seed.
    direct = original(None, **stats)
    expected_rng = torch.get_rng_state()
    torch.manual_seed(42)
    built = factory(None, **stats)
    assert type(built) is original
    assert list(built.state_dict()) == list(direct.state_dict())
    assert all(torch.equal(value, built.state_dict()[name]) for name, value in direct.state_dict().items())
    assert torch.equal(expected_rng, torch.get_rng_state())
    direct.eval()
    built.eval()
    with torch.no_grad():
        assert torch.equal(direct(batch), built(batch))
    assert len(built.state_dict()) == (119 if original is MSWPFuseBackbone else 155)


def test_basic_history_projection_is_daily_aligned_linear_mapping():
    module = BasicHistoryProjection(d_model=3, lag_dim=2)
    power = torch.arange(672 * 2, dtype=torch.float32).reshape(1, 672, 2) / 1000
    calendar = torch.arange(672 * 5, dtype=torch.float32).reshape(1, 672, 5) / 1000
    joined = torch.cat([power, calendar], dim=-1)
    # Independently gather each quarter-hour across its seven preceding days.
    expected_inputs = torch.stack([joined[:, step::96].flatten(1) for step in range(96)], dim=1)
    expected = nn.functional.linear(expected_inputs, module.projection.weight, module.projection.bias)
    history, lag = module(power, calendar)
    assert history.shape == (1, 96, 3)
    assert lag.shape == (1, 96, 2)
    assert torch.equal(torch.cat([history, lag], dim=-1), expected)
    (history.sum() + lag.sum()).backward()
    assert module.projection.weight.grad is not None
    with pytest.raises(InputShapeError, match="672 power/time steps"):
        module(power[:, :-1], calendar)


def test_fixed_fusion_reuses_projections_and_norm_with_equal_weights():
    source = AlignedFeatureFusion(d_model=8, lag_dim=3)
    fixed = FixedAlignedFeatureFusion(source)
    assert fixed.projections is source.projections
    assert fixed.output_norm is source.output_norm
    assert not any(name.startswith("gate.") for name in fixed.state_dict())
    generator = torch.Generator().manual_seed(42)
    inputs = [torch.rand(2, 96, width, generator=generator) for width in (8, 3, 8, 8)]
    branches = [layer(value) for layer, value in zip(source.projections, inputs)]
    expected = source.output_norm(torch.stack(branches, dim=2).mean(dim=2))
    assert torch.equal(fixed(*inputs), expected)


@pytest.mark.parametrize("factory", [build_backbone, build_selected])
@pytest.mark.parametrize("variant", BACKBONE_VARIANTS[1:])
def test_replacements_keep_all_surviving_canonical_initializations(factory, variant, stats):
    torch.manual_seed(42)
    full = factory(None, variant="full", **stats)
    full_rng = torch.get_rng_state()
    torch.manual_seed(42)
    replacement = factory(None, variant=variant, **stats)
    assert torch.equal(full_rng, torch.get_rng_state())
    full_state, replacement_state = full.state_dict(), replacement.state_dict()
    shared = full_state.keys() & replacement_state.keys()
    assert all(torch.equal(full_state[name], replacement_state[name]) for name in shared)
    if variant == "without_history_multiscale":
        removed = ("lag_history.", "patch_history.", "history_query.")
        assert "basic_history_projection.projection.weight" in replacement_state
        assert replacement.loaded_history_keys == ()
        assert not replacement.requires_history_initializer
    else:
        removed = ("scalar_provider.", "provider_fusion.", "aligned_fusion.gate.")
        assert "basic_weather_projection.weight" in replacement_state
        assert replacement.requires_history_initializer
    assert not any(name.startswith(removed) for name in replacement_state)
    if factory is build_selected:
        assert {name for name in replacement_state if name.startswith(AMENDMENT_PREFIXES)} == {
            name for name in full_state if name.startswith(AMENDMENT_PREFIXES)
        }


def test_weather_projection_masks_mean_and_all_missing_bias(stats, batch):
    model = build_selected(None, variant="without_source_aware_nwp_fusion", **stats).eval()
    expanded = {name: value.repeat_interleave(2, dim=0) for name, value in batch.items()}
    expanded["nwp_availability"] = torch.tensor([[1., 0., 1.], [0., 1., 0.], [0., 0., 0.], [1., 1., 1.]])
    with torch.no_grad():
        states = model.forward_features(expanded)
        scalar = expanded["nwp"].reshape(4, 96, 3, 9)
        mask = expanded["nwp_availability"][:, None, :, None]
        mean = (scalar * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1)
        expected = model.basic_weather_projection(mean) * mask.bool().any(dim=2).to(mean.dtype)
        assert torch.equal(states["fused_weather"], expected)
        assert torch.count_nonzero(states["fused_weather"][2]) == 0
        assert "provider_state" not in states
        # Values from missing sources cannot leak into the reduced weather path.
        changed = scalar.clone().masked_fill(~mask.bool().expand_as(scalar), 1000)
        changed_states = model.forward_features({**expanded, "nwp": changed.reshape(4, 96, 27)})
        assert torch.equal(states["fused_weather"], changed_states["fused_weather"])
        assert torch.equal(states["prediction"], changed_states["prediction"])


@pytest.mark.parametrize("variant", BACKBONE_VARIANTS)
def test_correction_uses_one_joint_dropout_mask(variant, stats, batch):
    model = build_selected(None, variant=variant, **stats).train()
    model.set_source_dropout(0.95)
    availability = torch.tensor([[0., 1., 0.], [1., 0., 1.]])
    seen = []
    handle = model.temporal_weather.register_forward_pre_hook(lambda module, args: seen.append(args[1].clone()))
    try:
        states = model.forward_features({**batch, "nwp_availability": availability})
    finally:
        handle.remove()
    assert torch.equal(seen[0], states["source_mask"])
    assert states["source_mask"].any(dim=1).all()
    assert not (states["source_mask"] & ~availability.bool()).any()
    mask = states["source_mask"][:, None, :, None]
    assert torch.count_nonzero(states["masked_scalar_nwp"].masked_select(~mask.expand_as(states["masked_scalar_nwp"]))) == 0


@pytest.mark.parametrize("variant", SELECTED_VARIANTS)
def test_output_constraints_are_retained(variant, stats, batch):
    model = build_selected(None, variant=variant, **stats).eval()
    with torch.no_grad():
        model.head.resource_weight.zero_()
        model.head.resource_factor.zero_()
        model.head.bias.fill_(0.5)
        scalar = batch["nwp"].reshape(2, 96, 3, 9).clone()
        scalar[..., POAI_INDEX] = 0
        inputs = {**batch, "nwp": scalar.reshape(2, 96, 27)}
        prediction = model(inputs)
        assert torch.isfinite(prediction).all() and (prediction >= 0).all()
        assert torch.count_nonzero(prediction[1]) == 0  # Available sources all say PV darkness.
        assert (prediction[0] > 0).all()  # Wind is unaffected.
        no_sources = model({**inputs, "nwp_availability": torch.zeros(2, 3)})
        assert (no_sources[1] > 0).all()  # No available source is not evidence of darkness.


@pytest.mark.parametrize("variant", ["full", "without_source_aware_nwp_fusion"])
def test_initializer_requirement_is_not_weakened(variant, stats, dataset):
    settings = load_paper_settings("autumn")
    model = build_backbone(settings["model"], variant=variant, **stats)
    with pytest.raises(ValueError, match="same-fold history initializer"):
        train_parent(model, dataset, settings["selected_training"], settings["parent_stage"], torch.device("cpu"))


@pytest.mark.parametrize("fold", ["summer", "autumn"])
def test_head_only_trainables_lr_calibration_loss_and_auxiliary_bypass(fold, stats, batch, dataset):
    settings = load_paper_settings(fold)
    full_spec = settings["selected_training"]
    head_spec = {**full_spec, "trainable_scope": "head_only"}
    torch.manual_seed(42)
    full = build_selected(settings["model"], **stats).eval()
    torch.manual_seed(42)
    head_only = build_selected(settings["model"], variant="head_only", **stats).eval()
    assert type(head_only) is MSWPFuseBackbone
    assert len(head_only.state_dict()) == 119
    assert not any(name.startswith(AMENDMENT_PREFIXES) for name in head_only.state_dict())
    with torch.no_grad():
        assert torch.equal(head_only(batch), full.forward_parent(batch))
    full_groups = configure_selected_scope(full, full_spec)
    head_groups = configure_selected_scope(head_only, head_spec)
    assert len(head_groups) == 1
    for key in ("lr", "base_lr", "minimum_lr", "weight_decay"):
        assert head_groups[0][key] == full_groups[1][key]
    full_optimizer = torch.optim.AdamW(full_groups)
    head_optimizer = torch.optim.AdamW(head_groups)
    for epoch in (1, 2, int(full_spec["max_epochs"])):
        _set_learning_rates(full_optimizer, epoch, int(full_spec["max_epochs"]), int(full_spec["warmup_epochs"]))
        _set_learning_rates(head_optimizer, epoch, int(head_spec["max_epochs"]), int(head_spec["warmup_epochs"]))
        assert head_optimizer.param_groups[0]["lr"] == full_optimizer.param_groups[1]["lr"]
    assert all(value.requires_grad == name.startswith("head.") for name, value in head_only.named_parameters())
    assert selected_gradient_balance(dataset, full_spec) == selected_gradient_balance(dataset, head_spec)
    balance = selected_gradient_balance(dataset, head_spec)
    composite = CompositeForecastLoss(balance.weights)
    labels = {
        "target": torch.stack([dataset[index]["target"] for index in (0, 5)]),
        "target_valid_mask": torch.stack([dataset[index]["target_valid_mask"] for index in (0, 5)]),
        "original_station_id": torch.tensor([1, 6]),
    }
    # No SRATR states or auxiliary settings exist in this loss invocation.
    states = head_only.forward_features(batch)
    assert not any(name in states for name in ("inactive_logit", "inactive_probability", "correction_gate"))
    loss = _selected_loss(head_only, states, labels, composite, {"trainable_scope": "head_only"})
    expected = composite(states["prediction"], labels["target"], labels["original_station_id"], valid_mask(labels["target"], labels["target_valid_mask"]), head_only.station_delta_penalty())
    assert torch.equal(loss, expected)
    altered_labels = {**labels, "target": labels["target"].masked_fill(~labels["target_valid_mask"], 1000)}
    assert torch.equal(loss, _selected_loss(head_only, states, altered_labels, composite, {"trainable_scope": "head_only"}))
    loss.backward()
    for name, parameter in head_only.named_parameters():
        if name.startswith("head."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None


def test_unreported_scopes_and_sratr_head_only_mismatch_are_rejected(stats):
    model = build_selected(None, **stats)
    for scope in ("full", "all", "unreported_scope"):
        with pytest.raises(ValueError, match="amendment_plus_head or head_only"):
            configure_selected_scope(model, {"trainable_scope": scope})
    invalid_models = [model, *(build_backbone(None, variant=variant, **stats) for variant in BACKBONE_VARIANTS[1:])]
    for invalid in invalid_models:
        with pytest.raises(ValueError, match="backbone-only model"):
            configure_selected_scope(invalid, {"trainable_scope": "head_only"})


@pytest.mark.parametrize("parent_variant", BACKBONE_VARIANTS)
def test_synthetic_parent_selected_prediction_connection(parent_variant, stats, dataset, tmp_path: Path):
    settings = load_paper_settings("autumn")
    history = None if parent_variant == "without_history_multiscale" else HistoryInitializer().state_dict()
    parent = build_backbone(settings["model"], variant=parent_variant, history_state_dict=history, **stats)
    before_parent = cpu_state_dict(parent)
    device = torch.device("cpu")
    parent_outcome = train_parent(parent, dataset, settings["selected_training"], {**settings["parent_stage"], "epochs": 1}, device)
    assert parent_outcome.best_epoch == 1
    assert not torch.equal(before_parent["head.bias"], parent.state_dict()["head.bias"])
    if history is not None:
        assert all(torch.equal(value, parent.state_dict()[name]) for name, value in before_parent.items() if name.startswith("patch_history."))
    else:
        assert not torch.equal(before_parent["basic_history_projection.projection.weight"], parent.state_dict()["basic_history_projection.projection.weight"])
    parent_path = tmp_path / "synthetic-parent-state.pt"
    torch.save(parent_outcome.state_dict, parent_path)
    saved_parent = torch.load(parent_path, map_location="cpu", weights_only=True)
    fresh_parent = build_backbone(settings["model"], variant=parent_variant, history_state_dict=history, **stats)
    fresh_parent.load_state_dict(saved_parent, strict=True)
    parent_prediction = collect_predictions(fresh_parent, dataset, device, batch_size=10)
    # Full and Head-only actually consume the very same trained parent artifact.
    variants = ("full", "head_only") if parent_variant == "full" else (parent_variant,)
    for variant in variants:
        spec = {**settings["selected_training"], "max_epochs": 1}
        if variant == "head_only":
            spec["trainable_scope"] = "head_only"
        selected = build_selected(settings["model"], variant=variant, **stats)
        incompatible = selected.load_state_dict(saved_parent, strict=False)
        assert not incompatible.unexpected_keys
        assert set(incompatible.missing_keys) == {name for name in selected.state_dict() if name.startswith(AMENDMENT_PREFIXES)}
        assert all(torch.equal(value, selected.state_dict()[name]) for name, value in saved_parent.items())
        selected_parent = collect_predictions(selected, dataset, device, batch_size=10, parent_only=isinstance(selected, MSWPFuse))
        np.testing.assert_array_equal(parent_prediction.predictions, selected_parent.predictions)
        before_selected = cpu_state_dict(selected)
        selected_outcome = train_selected(selected, dataset, dataset, spec, device)
        assert selected_outcome.best_epoch == 1
        assert len(selected_outcome.history) == 1
        assert np.isfinite([selected_outcome.best_c, selected_outcome.best_mae, selected_outcome.best_rmse]).all()
        assert selected_outcome.gradient_balance == selected_gradient_balance(dataset, spec)
        assert not torch.equal(before_selected["head.bias"], selected.state_dict()["head.bias"])
        for name, parameter in selected.named_parameters():
            expected_trainable = name.startswith("head.") or (variant != "head_only" and name.startswith(AMENDMENT_PREFIXES))
            assert parameter.requires_grad == expected_trainable
            if not parameter.requires_grad:
                assert torch.equal(before_selected[name], parameter)
        selected_path = tmp_path / f"synthetic-{variant}-selected-state.pt"
        torch.save(selected_outcome.state_dict, selected_path)
        fresh_selected = build_selected(settings["model"], variant=variant, **stats)
        fresh_selected.load_state_dict(torch.load(selected_path, map_location="cpu", weights_only=True), strict=True)
        configure_selected_scope(fresh_selected, spec)
        actual = collect_predictions(selected, dataset, device, batch_size=10)
        restored = collect_predictions(fresh_selected, dataset, device, batch_size=10)
        np.testing.assert_array_equal(actual.predictions, restored.predictions)
        np.testing.assert_array_equal(actual.valid_target_mask, restored.valid_target_mask)
        assert actual.sample_keys == restored.sample_keys
        assert actual.predictions.shape == (10, 96) and np.isfinite(actual.predictions).all()
