"""Shared neural network primitives and numerical operations for MSWP-Fuse."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HORIZON = 96
FORECAST_STEPS = 96
HISTORY_STEPS = 672
HISTORY_DAYS = 7
NATIVE_STEPS = 96
HOURLY_STEPS = 24
N_SOURCES = 3
N_CHANNELS = 9
N_STATIONS = 10
N_RESOURCES = 2
METRIC_FLOOR = 0.2

WIND_RESOURCE_ID = 0
SOLAR_RESOURCE_ID = 1
POAI_INDEX = 1
PRESSURE_INDEX = 2
TEMPERATURE_INDEX = 3
CLOUD_INDEX = 4
PRECIPITATION_INDEX = 5
U_WIND_INDEX = 6
V_WIND_INDEX = 7
SPEED_INDEX = 8

AMENDMENT_PREFIXES = (
    "temporal_weather.",
    "station_regime_interaction.",
    "inactive_head.",
    "correction_head.",
    "scale_logit",
)


class InputShapeError(ValueError):
    """Raised when batch tensor shapes or value domains violate the contract."""


InputContractError = InputShapeError


def fp32_cuda_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def check_scalar_inputs(batch: Mapping[str, Tensor]) -> None:
    required = (
        "hist_power",
        "hist_time",
        "future_time",
        "nwp",
        "nwp_availability",
        "station_id",
        "resource_id",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise InputShapeError(f"Missing required inputs: {missing}")
    shapes = {
        "hist_power": (HISTORY_STEPS, 2),
        "hist_time": (HISTORY_STEPS, 5),
        "future_time": (HORIZON, 5),
        "nwp": (HORIZON, N_SOURCES * N_CHANNELS),
        "nwp_availability": (N_SOURCES,),
    }
    for key, suffix in shapes.items():
        value = batch[key]
        if value.ndim != len(suffix) + 1 or tuple(value.shape[1:]) != suffix:
            raise InputShapeError(
                f"{key} must have shape (B, {', '.join(map(str, suffix))}), got {tuple(value.shape)}"
            )
    batch_size = batch["hist_power"].size(0)
    for key in ("station_id", "resource_id"):
        value = batch[key]
        if value.ndim != 1 or value.size(0) != batch_size:
            raise InputShapeError(f"{key} must have shape (B,), got {tuple(value.shape)}")
    if not torch.isfinite(batch["hist_power"]).all():
        raise InputShapeError("hist_power contains non-finite values")
    if not torch.isfinite(batch["hist_time"]).all() or not torch.isfinite(batch["future_time"]).all():
        raise InputShapeError("time features contain non-finite values")
    if not torch.isfinite(batch["nwp"]).all():
        raise InputShapeError("nwp contains non-finite values")
    availability = batch["nwp_availability"]
    if not torch.isfinite(availability.to(torch.float32)).all():
        raise InputShapeError("nwp_availability contains non-finite values")
    if bool(((availability < 0) | (availability > 1)).any()):
        raise InputShapeError("nwp_availability must contain binary values")


def make_joint_source_mask(
    availability: Tensor,
    *,
    probability: float,
    training: bool,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one source mask that is reused by scalar, disagreement, and physics paths."""
    if availability.ndim != 2 or availability.size(1) != N_SOURCES:
        raise InputShapeError("availability must have shape (B, 3)")
    original = availability.to(dtype=torch.bool)
    if not training or probability <= 0.0:
        return original
    if probability >= 1.0:
        raise ValueError("Source dropout probability must be below one")
    random = torch.rand(
        availability.shape,
        device=availability.device,
        generator=generator,
    )
    kept = original & (random >= probability)
    has_original = original.any(dim=1)
    no_kept = has_original & ~kept.any(dim=1)
    if no_kept.any():
        first = original.to(torch.int64).argmax(dim=1)
        kept[no_kept] = False
        kept[no_kept, first[no_kept]] = True
    return kept


def inverse_standardize_scalar(
    standardized: Tensor,
    mean: Tensor,
    std: Tensor,
    source_mask: Tensor,
) -> Tensor:
    if standardized.ndim < 3 or tuple(standardized.shape[-2:]) != (N_SOURCES, N_CHANNELS):
        raise InputShapeError("Unexpected standardized scalar shape")
    raw = standardized * std.view(1, 1, N_SOURCES, N_CHANNELS)
    raw = raw + mean.view(1, 1, N_SOURCES, N_CHANNELS)
    return raw * source_mask[:, None, :, None].to(raw.dtype)


def source_disagreement(values: Tensor, source_mask: Tensor) -> Tensor:
    mask = source_mask[:, None, :, None].to(values.dtype)
    count = mask.sum(dim=2, keepdim=True)
    mean = (values * mask).sum(dim=2, keepdim=True) / count.clamp_min(1.0)
    residual = (values - mean) * mask
    return torch.where(count >= 2.0, residual, torch.zeros_like(residual))


def source_disagreement_profiles(
    values: Tensor,
    availability: Tensor,
) -> Tensor:
    mask = availability[:, None, :, None].to(values.dtype)
    count = mask.sum(dim=2, keepdim=True).clamp_min(1.0)
    mean = (values * mask).sum(dim=2, keepdim=True) / count
    disagreement = (values - mean) * mask
    enough_sources = availability.sum(dim=1) >= 2
    return torch.where(
        enough_sources[:, None, None, None],
        disagreement,
        torch.zeros_like(disagreement),
    )


def scalar_physics_features(raw: Tensor) -> Tensor:
    pressure = raw[..., PRESSURE_INDEX].clamp(50_000.0, 120_000.0)
    temperature = raw[..., TEMPERATURE_INDEX].clamp(200.0, 350.0)
    speed = raw[..., SPEED_INDEX].clamp_min(0.0)
    u_wind = raw[..., U_WIND_INDEX]
    v_wind = raw[..., V_WIND_INDEX]
    density = pressure / (287.05 * temperature)
    density_ratio = (density / 1.225).clamp_min(1.0e-4)
    effective_speed = speed * density_ratio.pow(1.0 / 3.0)
    speed_norm = effective_speed / 12.0
    direction_norm = speed.clamp_min(1.0e-4)
    poai = raw[..., POAI_INDEX].clamp_min(0.0)
    ghi = raw[..., 0].clamp_min(0.0)
    cloud = raw[..., CLOUD_INDEX].clamp(0.0, 1.5)
    precipitation = raw[..., PRECIPITATION_INDEX].clamp_min(0.0)
    temperature_c = temperature - 273.15
    cell_temperature = temperature_c + 25.0 * poai / 1000.0
    temperature_factor = (1.0 - 0.004 * (cell_temperature - 25.0)).clamp(0.5, 1.5)
    adjusted_poai = poai * temperature_factor
    return torch.stack(
        [
            speed_norm,
            speed_norm.square(),
            speed_norm.square() * speed_norm,
            v_wind / direction_norm,
            u_wind / direction_norm,
            density_ratio,
            poai / 1000.0,
            ghi / 1000.0,
            cloud,
            precipitation,
            temperature_c / 40.0,
            adjusted_poai / 1000.0,
        ],
        dim=-1,
    )


def smooth_nonnegative(raw: Tensor) -> Tensor:
    values = raw.float()
    return 0.5 * (values + torch.sqrt(values.square() + 1.0e-8))


def apply_solar_darkness(
    prediction: Tensor,
    raw_scalar: Tensor,
    source_mask: Tensor,
    resource_id: Tensor,
    *,
    threshold: float = 1.0e-6,
) -> Tensor:
    available = source_mask.bool()[:, None, :].expand(-1, HORIZON, -1)
    any_available = available.any(dim=-1)
    poai = raw_scalar[..., POAI_INDEX]
    available_low = torch.where(
        available,
        poai <= threshold,
        torch.ones_like(poai, dtype=torch.bool),
    )
    dark = any_available & available_low.all(dim=-1)
    solar = resource_id.long()[:, None] == SOLAR_RESOURCE_ID
    return torch.where(solar & dark, torch.zeros_like(prediction), prediction)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        encoding = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding[0, :, 0::2] = torch.sin(position * divisor)
        encoding[0, :, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.encoding[:, : values.size(1)].to(values)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm1(values)
        values = values + self.attention(normalized, normalized, normalized)[0]
        return values + self.feed_forward(self.norm2(values))


class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
    ):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
    ) -> Tensor:
        normalized_query = self.query_norm(query)
        normalized_memory = self.memory_norm(memory)
        query = query + self.attention(
            normalized_query,
            normalized_memory,
            normalized_memory,
        )[0]
        return query + self.feed_forward(self.output_norm(query))


class ConditionalLayerNorm(nn.Module):
    def __init__(self, d_model: int, condition_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.film = nn.Linear(condition_dim, 2 * d_model)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.film(condition).chunk(2, dim=-1)
        return self.norm(values) * (1.0 + scale) + shift
