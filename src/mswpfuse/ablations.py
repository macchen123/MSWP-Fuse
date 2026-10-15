"""Paper-reported representation replacements and matched output-head control.

The representation variants fit their own backbone before correction training.
Head-only instead starts from the full backbone checkpoint. Removing SRATR at
inference uses that existing full backbone; it is not another trained variant.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .model import MSWPFuse, MSWPFuseBackbone, MSWPFuseConfig
from .primitives import HISTORY_STEPS, HORIZON, N_CHANNELS, InputShapeError, fp32_cuda_context

_REPRESENTATION_VARIANTS = (
    "without_history_multiscale",
    "without_source_aware_nwp_fusion",
)
BACKBONE_VARIANTS = ("full", *_REPRESENTATION_VARIANTS)
SELECTED_VARIANTS = (*BACKBONE_VARIANTS, "head_only")


class BasicHistoryProjection(nn.Module):
    """Map seven daily-aligned power/calendar vectors to the two history slots."""

    def __init__(self, *, d_model: int, lag_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(49, d_model + lag_dim)
        self.d_model = int(d_model)
        self.lag_dim = int(lag_dim)

    def forward(self, hist_power: Tensor, hist_time: Tensor) -> tuple[Tensor, Tensor]:
        if hist_power.shape[-2:] != (HISTORY_STEPS, 2) or hist_time.shape[-2:] != (HISTORY_STEPS, 5):
            raise InputShapeError("Basic history projection requires 672 power/time steps")
        values = torch.cat([hist_power, hist_time], dim=-1)
        values = values.reshape(values.size(0), 7, HORIZON, 7)
        values = values.transpose(1, 2).contiguous().reshape(values.size(0), HORIZON, 49)
        projected = self.projection(values.float())
        return projected[..., : self.d_model], projected[..., self.d_model :]


class FixedAlignedFeatureFusion(nn.Module):
    """Retain the original branch projections/norm, replacing gates by a mean."""

    def __init__(self, source: nn.Module) -> None:
        super().__init__()
        self.projections = source.projections
        self.output_norm = source.output_norm

    def forward(
        self,
        history: Tensor,
        lag_history: Tensor,
        weather: Tensor,
        condition: Tensor,
    ) -> Tensor:
        with fp32_cuda_context(history.device):
            branches = [
                self.projections[0](history.float()),
                self.projections[1](lag_history.float()),
                self.projections[2](weather.float()),
                self.projections[3](condition.float()),
            ]
            return self.output_norm(torch.stack(branches, dim=2).mean(dim=2))


class _RepresentationMixin:
    """Change only the reported encodings, sharing both canonical forwards."""

    def __init__(self, *args: Any, variant: str, **kwargs: Any) -> None:
        if variant not in _REPRESENTATION_VARIANTS:
            raise ValueError(f"Unsupported representation variant: {variant!r}")
        projection_rng = torch.get_rng_state()
        super().__init__(*args, **kwargs)
        self._representation_variant = variant
        # Construct the complete canonical model first, so surviving backbone
        # and SRATR tensors keep their original initialization and ordering.
        # A replacement starts from the caller's incoming RNG state without
        # reseeding or advancing the canonical construction's outgoing state.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(projection_rng)
            if variant == "without_history_multiscale":
                del self.lag_history, self.patch_history, self.history_query
                self.basic_history_projection = BasicHistoryProjection(
                    d_model=self.config.d_model, lag_dim=self.config.lag_hidden
                )
                self._loaded_history_keys = ()
            else:
                del self.scalar_provider, self.provider_fusion
                self.aligned_fusion = FixedAlignedFeatureFusion(self.aligned_fusion)
                self.basic_weather_projection = nn.Linear(N_CHANNELS, self.config.d_model)

    @property
    def requires_history_initializer(self) -> bool:
        return self._representation_variant != "without_history_multiscale"

    def _history_states(
        self, batch: Mapping[str, Tensor], condition: Tensor
    ) -> tuple[Tensor, Tensor]:
        if self._representation_variant == "without_history_multiscale":
            return self.basic_history_projection(batch["hist_power"], batch["hist_time"])
        return super()._history_states(batch, condition)

    def _weather_states(
        self,
        batch: Mapping[str, Tensor],
        scalar: Tensor,
        raw_scalar: Tensor,
        source_mask: Tensor,
        condition: Tensor,
    ) -> dict[str, Tensor]:
        if self._representation_variant != "without_source_aware_nwp_fusion":
            return super()._weather_states(batch, scalar, raw_scalar, source_mask, condition)
        mask = source_mask[:, None, :, None].to(scalar.dtype)
        count = mask.sum(dim=2).clamp_min(1.0)
        equal_mean = (scalar * mask).sum(dim=2) / count
        weather = self.basic_weather_projection(equal_mean)
        has_source = source_mask.bool().any(dim=1).to(weather.dtype)
        return {"fused_weather": weather * has_source[:, None, None]}


class _RepresentationBackbone(_RepresentationMixin, MSWPFuseBackbone):
    """First-stage model for one of the two representation replacements."""


class _RepresentationSelected(_RepresentationMixin, MSWPFuse):
    """The same replacement backbone with the unchanged SRATR correction."""


def build_backbone(
    config: MSWPFuseConfig | Mapping[str, Any] | None,
    *,
    nwp_mean: Tensor,
    nwp_std: Tensor,
    history_state_dict: Mapping[str, Tensor] | None = None,
    variant: str = "full",
) -> MSWPFuseBackbone:
    """Build Full or a reported replacement for its own backbone training."""
    if variant not in BACKBONE_VARIANTS:
        raise ValueError(f"Unsupported backbone variant: {variant!r}; expected {BACKBONE_VARIANTS}")
    kwargs = dict(nwp_mean=nwp_mean, nwp_std=nwp_std, history_state_dict=history_state_dict)
    if variant == "full":
        return MSWPFuseBackbone(config, **kwargs)
    return _RepresentationBackbone(config, variant=variant, **kwargs)


def build_selected(
    config: MSWPFuseConfig | Mapping[str, Any] | None,
    *,
    nwp_mean: Tensor,
    nwp_std: Tensor,
    variant: str = "full",
) -> MSWPFuseBackbone:
    """Build an output stage; load its matching completed backbone before fit.

    Full and Head-only share the full parent. Head-only has exactly the original
    backbone topology and prediction path, with no SRATR modules or auxiliaries.
    Its training settings must use ``trainable_scope="head_only"``.
    """
    if variant not in SELECTED_VARIANTS:
        raise ValueError(f"Unsupported selected variant: {variant!r}; expected {SELECTED_VARIANTS}")
    kwargs = dict(nwp_mean=nwp_mean, nwp_std=nwp_std)
    if variant == "full":
        return MSWPFuse(config, **kwargs)
    if variant == "head_only":
        return MSWPFuseBackbone(config, **kwargs)
    return _RepresentationSelected(config, variant=variant, **kwargs)


__all__ = ["BACKBONE_VARIANTS", "SELECTED_VARIANTS", "build_backbone", "build_selected"]
