"""Masked losses and training-only gradient balancing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

EPSILON = 1.0e-8
METRIC_FLOOR = 0.2


def valid_mask(target: Tensor, supplied: Tensor | None = None) -> Tensor:
    if supplied is None:
        return torch.isfinite(target) & (target >= 0.0)
    if supplied.shape != target.shape:
        raise ValueError("Target mask must match the target shape")
    return supplied.bool() & torch.isfinite(target) & (target >= 0.0)


def masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    denominator = mask.sum().to(values.dtype).clamp_min(1.0)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / denominator


def _check_shapes(prediction: Tensor, target: Tensor) -> None:
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.size(1) != 96:
        raise ValueError("Predictions and targets must share shape (B, 96)")


class StationMacroCLoss(nn.Module):
    """A differentiable surrogate for the station-macro C score."""

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        station_ids: Tensor,
        supplied_mask: Tensor | None = None,
    ) -> Tensor:
        _check_shapes(prediction, target)
        mask = valid_mask(target, supplied_mask)
        valid_samples = mask.any(dim=1)
        if not bool(valid_samples.any()):
            return prediction.sum() * 0.0
        denominator = target.clamp_min(METRIC_FLOOR).detach()
        normalized_error = ((prediction - target) / denominator).square()
        daily = torch.sqrt(
            (normalized_error * mask.to(prediction.dtype)).sum(dim=1)
            / mask.sum(dim=1).clamp_min(1).to(prediction.dtype)
            + EPSILON
        )
        station_values = []
        for station in torch.unique(station_ids[valid_samples].long(), sorted=True):
            selected = valid_samples & (station_ids.long() == station)
            station_values.append(daily[selected].mean())
        return torch.stack(station_values).mean() if station_values else prediction.sum() * 0.0


class CharbonnierMAELoss(nn.Module):
    def __init__(self, delta: float = 0.01):
        super().__init__()
        self.delta = float(delta)

    def forward(self, prediction: Tensor, target: Tensor, supplied_mask: Tensor | None = None) -> Tensor:
        _check_shapes(prediction, target)
        value = torch.sqrt((prediction - target).square() + self.delta**2) - self.delta
        return masked_mean(value, valid_mask(target, supplied_mask))


class PooledRMSELoss(nn.Module):
    def forward(self, prediction: Tensor, target: Tensor, supplied_mask: Tensor | None = None) -> Tensor:
        _check_shapes(prediction, target)
        return torch.sqrt(masked_mean((prediction - target).square(), valid_mask(target, supplied_mask)) + EPSILON)


class CompositeForecastLoss(nn.Module):
    component_names = ("station_macro_c", "charbonnier_mae", "pooled_rmse")

    def __init__(self, weights: Mapping[str, float], station_penalty_weight: float = 1.0e-4):
        super().__init__()
        if set(weights) != set(self.component_names):
            raise ValueError("Loss weights must cover all three components")
        vector = torch.tensor([float(weights[name]) for name in self.component_names])
        if not torch.isfinite(vector).all() or bool((vector <= 0).any()):
            raise ValueError("Loss weights must be finite and positive")
        self.register_buffer("weights", vector / vector.sum())
        self.station_c = StationMacroCLoss()
        self.charbonnier = CharbonnierMAELoss()
        self.rmse = PooledRMSELoss()
        self.station_penalty_weight = float(station_penalty_weight)

    def components(
        self,
        prediction: Tensor,
        target: Tensor,
        station_ids: Tensor,
        supplied_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        return {
            "station_macro_c": self.station_c(prediction, target, station_ids, supplied_mask),
            "charbonnier_mae": self.charbonnier(prediction, target, supplied_mask),
            "pooled_rmse": self.rmse(prediction, target, supplied_mask),
        }

    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        station_ids: Tensor,
        supplied_mask: Tensor | None = None,
        station_penalty: Tensor | None = None,
    ) -> Tensor:
        values = self.components(prediction, target, station_ids, supplied_mask)
        vector = torch.stack([values[name] for name in self.component_names])
        result = (vector * self.weights.to(vector)).sum()
        if station_penalty is not None:
            result = result + self.station_penalty_weight * station_penalty
        return result


@dataclass(frozen=True)
class GradientBalance:
    weights: dict[str, float]
    gradient_rms: dict[str, float]


def calibrate_gradient_weights(
    target: Tensor,
    supplied_mask: Tensor,
    station_ids: Tensor,
    reference_prediction: Tensor,
    desired_shares: Mapping[str, float],
) -> GradientBalance:
    """Set fixed component weights from training data only."""

    _check_shapes(reference_prediction, target)
    if set(desired_shares) != set(CompositeForecastLoss.component_names):
        raise ValueError("Desired shares must cover all three loss components")
    desired = torch.tensor(
        [float(desired_shares[name]) for name in CompositeForecastLoss.component_names],
        dtype=torch.float64,
    )
    if not torch.isfinite(desired).all() or bool((desired <= 0).any()):
        raise ValueError("Desired shares must be finite and positive")
    desired = desired / desired.sum()
    mask = valid_mask(target, supplied_mask)
    prediction = reference_prediction.detach().clone().to(torch.float64).requires_grad_(True)
    target64 = target.detach().to(torch.float64)
    components = {
        "station_macro_c": StationMacroCLoss()(prediction, target64, station_ids, mask),
        "charbonnier_mae": CharbonnierMAELoss()(prediction, target64, mask),
        "pooled_rmse": PooledRMSELoss()(prediction, target64, mask),
    }
    gradients: dict[str, Tensor] = {}
    names = CompositeForecastLoss.component_names
    for index, name in enumerate(names):
        gradient = torch.autograd.grad(components[name], prediction, retain_graph=index < len(names) - 1)[0]
        gradients[name] = torch.where(mask, gradient, torch.zeros_like(gradient))
    count = mask.sum().to(torch.float64).clamp_min(1.0)
    norms = {name: torch.sqrt(gradient.square().sum() / count) for name, gradient in gradients.items()}
    if any(not bool(torch.isfinite(value)) or bool(value <= 0) for value in norms.values()):
        raise FloatingPointError("Gradient balancing produced an invalid norm")
    unnormalized = {name: desired[index] / norms[name].clamp_min(1.0e-12) for index, name in enumerate(names)}
    total = torch.stack(list(unnormalized.values())).sum()
    return GradientBalance(
        weights={name: float(value / total) for name, value in unnormalized.items()},
        gradient_rms={name: float(value) for name, value in norms.items()},
    )
