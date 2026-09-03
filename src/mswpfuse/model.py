"""The standalone MSWP-Fuse SRATR forecasting model."""

from __future__ import annotations

import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

HISTORY_STEPS = 672
FORECAST_STEPS = 96
N_SOURCES = 3
N_CHANNELS = 9
N_STATIONS = 10
N_RESOURCES = 2
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


@dataclass(frozen=True)
class MSWPFuseConfig:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    history_layers: int = 2
    history_patch_length: int = 8
    history_patch_stride: int = 8
    lag_hidden: int = 64
    provider_heads: int = 4
    horizon_head_rank: int = 16
    dropout: float = 0.10
    use_refiner: bool = True
    temporal_hidden: int = 24
    interaction_rank: int = 12
    bounded_scale_maximum: float = 0.15
    initial_scale_fraction: float = 0.4125854203825821
    initial_inactive_bias: float = -6.0
    initial_correction_gate_bias: float = -4.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> "MSWPFuseConfig":
        if value is None:
            return cls()
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: value[key] for key in value if key in allowed})

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.history_patch_length != 8 or self.history_patch_stride != 8:
            raise ValueError("The public model uses non-overlapping 8-step history patches")
        if self.d_model != 128 or self.lag_hidden != 64:
            raise ValueError("The public model uses the paper dimensions")
        if not 0.0 < self.initial_scale_fraction < 1.0:
            raise ValueError("initial_scale_fraction must lie between zero and one")
        if self.temporal_hidden != 24 or self.interaction_rank != 12:
            raise ValueError("The public model uses the selected temporal dimensions")
        if self.bounded_scale_maximum != 0.15:
            raise ValueError("The public model uses a maximum residual scale of 0.15")


class InputShapeError(ValueError):
    pass


def _fp32_context(device: torch.device):
    return torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext()


def _check_inputs(batch: Mapping[str, Tensor]) -> None:
    required = (
        "hist_power",
        "hist_time",
        "future_time",
        "nwp",
        "nwp_availability",
        "station_id",
        "resource_id",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise InputShapeError(f"Missing model inputs: {missing}")
    expected = {
        "hist_power": (HISTORY_STEPS, 2),
        "hist_time": (HISTORY_STEPS, 5),
        "future_time": (FORECAST_STEPS, 5),
        "nwp": (FORECAST_STEPS, N_SOURCES * N_CHANNELS),
        "nwp_availability": (N_SOURCES,),
    }
    for name, suffix in expected.items():
        value = batch[name]
        if value.ndim != len(suffix) + 1 or tuple(value.shape[1:]) != suffix:
            raise InputShapeError(f"{name} must have shape (B, {', '.join(map(str, suffix))})")
    batch_size = batch["hist_power"].size(0)
    for name in ("station_id", "resource_id"):
        value = batch[name]
        if value.ndim != 1 or value.size(0) != batch_size:
            raise InputShapeError(f"{name} must have shape (B,)")
    if not all(bool(torch.isfinite(batch[name]).all()) for name in ("hist_power", "hist_time", "future_time", "nwp")):
        raise InputShapeError("Model inputs contain non-finite values")
    availability = batch["nwp_availability"]
    if not bool(torch.isfinite(availability.float()).all()) or bool(((availability < 0) | (availability > 1)).any()):
        raise InputShapeError("nwp_availability must contain binary values")


def make_source_mask(availability: Tensor, probability: float, training: bool) -> Tensor:
    if availability.ndim != 2 or availability.size(1) != N_SOURCES:
        raise InputShapeError("nwp_availability must have shape (B, 3)")
    original = availability.bool()
    if not training or probability <= 0.0:
        return original
    if probability >= 1.0:
        raise ValueError("Source dropout must be below one")
    kept = original & (torch.rand_like(availability.float()) >= probability)
    restore = original.any(dim=1) & ~kept.any(dim=1)
    if bool(restore.any()):
        first_available = original.long().argmax(dim=1)
        kept[restore] = False
        kept[restore, first_available[restore]] = True
    return kept


def inverse_standardize_scalar(values: Tensor, mean: Tensor, std: Tensor, source_mask: Tensor) -> Tensor:
    reshaped_mean = mean.view(1, 1, N_SOURCES, N_CHANNELS).to(values)
    reshaped_std = std.view(1, 1, N_SOURCES, N_CHANNELS).to(values)
    raw = values * reshaped_std + reshaped_mean
    return raw * source_mask[:, None, :, None].to(raw.dtype)


def source_disagreement(values: Tensor, source_mask: Tensor) -> Tensor:
    mask = source_mask[:, None, :, None].to(values.dtype)
    count = mask.sum(dim=2, keepdim=True)
    mean = (values * mask).sum(dim=2, keepdim=True) / count.clamp_min(1.0)
    residual = (values - mean) * mask
    return torch.where(count >= 2.0, residual, torch.zeros_like(residual))


def scalar_physics_features(raw: Tensor) -> Tensor:
    pressure = raw[..., PRESSURE_INDEX].clamp(50_000.0, 120_000.0)
    temperature = raw[..., TEMPERATURE_INDEX].clamp(200.0, 350.0)
    speed = raw[..., SPEED_INDEX].clamp_min(0.0)
    density = pressure / (287.05 * temperature)
    density_ratio = (density / 1.225).clamp_min(1.0e-4)
    speed_norm = speed * density_ratio.pow(1.0 / 3.0) / 12.0
    direction = speed.clamp_min(1.0e-4)
    poai = raw[..., POAI_INDEX].clamp_min(0.0)
    ghi = raw[..., 0].clamp_min(0.0)
    cloud = raw[..., CLOUD_INDEX].clamp(0.0, 1.5)
    precipitation = raw[..., PRECIPITATION_INDEX].clamp_min(0.0)
    temperature_c = temperature - 273.15
    cell_temperature = temperature_c + 25.0 * poai / 1000.0
    temperature_factor = (1.0 - 0.004 * (cell_temperature - 25.0)).clamp(0.5, 1.5)
    return torch.stack(
        [
            speed_norm,
            speed_norm.square(),
            speed_norm.square() * speed_norm,
            raw[..., V_WIND_INDEX] / direction,
            raw[..., U_WIND_INDEX] / direction,
            density_ratio,
            poai / 1000.0,
            ghi / 1000.0,
            cloud,
            precipitation,
            temperature_c / 40.0,
            poai * temperature_factor / 1000.0,
        ],
        dim=-1,
    )


def smooth_nonnegative(values: Tensor) -> Tensor:
    values = values.float()
    return 0.5 * (values + torch.sqrt(values.square() + 1.0e-8))


def apply_solar_darkness(prediction: Tensor, raw_scalar: Tensor, source_mask: Tensor, resource_id: Tensor) -> Tensor:
    available = source_mask.bool()[:, None, :].expand(-1, FORECAST_STEPS, -1)
    any_available = available.any(dim=-1)
    poai = raw_scalar[..., POAI_INDEX]
    every_available_source_dark = torch.where(available, poai <= 1.0e-6, torch.ones_like(poai, dtype=torch.bool)).all(dim=-1)
    is_solar = resource_id.long()[:, None] == 1
    return torch.where(is_solar & any_available & every_available_source_dark, torch.zeros_like(prediction), prediction)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 1024):
        super().__init__()
        encoding = torch.zeros(1, max_length, d_model)
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        encoding[0, :, 0::2] = torch.sin(position * divisor)
        encoding[0, :, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding)

    def forward(self, values: Tensor) -> Tensor:
        return values + self.encoding[:, : values.size(1)].to(values)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, values: Tensor) -> Tensor:
        normalized = self.norm1(values)
        values = values + self.attention(normalized, normalized, normalized, need_weights=False)[0]
        return values + self.feed_forward(self.norm2(values))


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(d_model)
        self.memory_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.output_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model), nn.Dropout(dropout)
        )

    def forward(self, query: Tensor, memory: Tensor) -> Tensor:
        query = query + self.attention(self.query_norm(query), self.memory_norm(memory), self.memory_norm(memory), need_weights=False)[0]
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


class StationResourceConditioner(nn.Module):
    def __init__(self, d_model: int, condition_dim: int = 64):
        super().__init__()
        self.resource_embedding = nn.Embedding(N_RESOURCES, condition_dim)
        self.station_delta = nn.Embedding(N_STATIONS, condition_dim)
        self.time_projection = nn.Linear(5, condition_dim)
        self.horizon_embedding = nn.Parameter(torch.randn(1, FORECAST_STEPS, condition_dim) * 0.02)
        self.output_projection = nn.Sequential(nn.Linear(condition_dim, d_model), nn.LayerNorm(d_model), nn.GELU())

    def forward(self, station_id: Tensor, resource_id: Tensor, future_time: Tensor) -> Tensor:
        station = station_id.long()
        resource = resource_id.long()
        if bool((station < 0).any()) or bool((station >= N_STATIONS).any()):
            raise InputShapeError("station_id must be in 0..9")
        if bool((resource < 0).any()) or bool((resource >= N_RESOURCES).any()):
            raise InputShapeError("resource_id must be in 0..1")
        condition = (self.resource_embedding(resource) + self.station_delta(station)).unsqueeze(1)
        condition = condition + self.time_projection(future_time) + self.horizon_embedding
        return self.output_projection(condition)

    def station_delta_penalty(self) -> Tensor:
        return self.station_delta.weight.square().mean()


class LagBankHistoryEncoder(nn.Module):
    def __init__(self, hidden_size: int = 64, condition_dim: int = 128):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(7, hidden_size, batch_first=True)
        self.condition_projection = nn.Linear(condition_dim, hidden_size)

    def forward(self, hist_power: Tensor, hist_time: Tensor, condition: Tensor) -> Tensor:
        values = torch.cat([hist_power, hist_time], dim=-1)
        values = values.reshape(values.size(0), 7, FORECAST_STEPS, 7).transpose(1, 2).contiguous()
        values = values.reshape(values.size(0) * FORECAST_STEPS, 7, 7)
        _, state = self.gru(values)
        return state[-1].reshape(-1, FORECAST_STEPS, self.hidden_size) + self.condition_projection(condition)


class HistoryPatchEncoder(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        self.patch_length = config.history_patch_length
        self.patch_stride = config.history_patch_stride
        self.history_patch_projection = nn.Linear(self.patch_length, config.d_model)
        self.history_time_projection = nn.Linear(5, config.d_model, bias=False)
        self.history_channel_embedding = nn.Embedding(2, config.d_model)
        self.position = PositionalEncoding(config.d_model)
        self.encoder = nn.ModuleList(
            [TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout) for _ in range(config.history_layers)]
        )

    def forward(self, hist_power: Tensor, hist_time: Tensor) -> Tensor:
        history = hist_power.transpose(1, 2)
        patches = history.unfold(2, self.patch_length, self.patch_stride)
        time_patches = hist_time.transpose(1, 2).unfold(2, self.patch_length, self.patch_stride).mean(dim=-1).transpose(1, 2)
        values = self.history_patch_projection(patches) + self.history_time_projection(time_patches).unsqueeze(1)
        channels = self.history_channel_embedding(torch.arange(2, device=history.device))
        values = values + channels.view(1, 2, 1, -1)
        values = self.position(values.flatten(1, 2))
        for layer in self.encoder:
            values = layer(values)
        return values


class HorizonHistoryQuery(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, FORECAST_STEPS, config.d_model) * 0.02)
        self.attention = CrossAttentionBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)

    def forward(self, condition: Tensor, memory: Tensor) -> Tensor:
        return self.attention(self.query.expand(condition.size(0), -1, -1) + condition, memory)


class ScalarProviderEncoder(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        self.input_projection = nn.Linear(36, config.d_model)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.condition = ConditionalLayerNorm(config.d_model, config.d_model)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        with _fp32_context(values.device):
            encoded = F.gelu(self.norm(self.input_projection(values.float())))
            encoded = self.output_projection(self.dropout(encoded))
            return self.condition(encoded, condition.float().unsqueeze(2))


class MaskedProviderSetFusion(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.d_model, config.provider_heads, batch_first=True)
        self.projection = nn.Sequential(nn.Linear(2 * config.d_model, config.d_model), nn.LayerNorm(config.d_model), nn.GELU())
        self.condition = ConditionalLayerNorm(config.d_model, config.d_model)

    def forward(self, values: Tensor, source_mask: Tensor, condition: Tensor) -> Tensor:
        batch_size = values.size(0)
        valid = source_mask.bool()
        safe_valid = valid.clone()
        no_source = ~safe_valid.any(dim=1)
        if bool(no_source.any()):
            safe_valid[no_source, 0] = True
        flattened = values.float().reshape(batch_size * FORECAST_STEPS, N_SOURCES, -1)
        padding = ~safe_valid[:, None, :].expand(batch_size, FORECAST_STEPS, N_SOURCES).reshape(batch_size * FORECAST_STEPS, N_SOURCES)
        attended = self.attention(flattened, flattened, flattened, key_padding_mask=padding, need_weights=False)[0]
        attended = attended.reshape(batch_size, FORECAST_STEPS, N_SOURCES, -1)
        mask = valid[:, None, :, None].to(attended.dtype)
        count = mask.sum(dim=2).clamp_min(1.0)
        mean = (attended * mask).sum(dim=2) / count
        deviation = torch.sqrt(((attended - mean.unsqueeze(2)).square() * mask).sum(dim=2) / count + 1.0e-8)
        output = self.condition(self.projection(torch.cat([mean, deviation], dim=-1)), condition.float())
        return output * valid.any(dim=1).to(output.dtype)[:, None, None]


class AlignedFeatureFusion(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model), nn.Linear(config.lag_hidden, config.d_model), nn.Linear(config.d_model, config.d_model), nn.Linear(config.d_model, config.d_model)]
        )
        self.gate = nn.Linear(4 * config.d_model, 4)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, history: Tensor, lag: Tensor, weather: Tensor, condition: Tensor) -> Tensor:
        branches = [
            self.projections[0](history.float()),
            self.projections[1](lag.float()),
            self.projections[2](weather.float()),
            self.projections[3](condition.float()),
        ]
        weights = torch.softmax(self.gate(torch.cat(branches, dim=-1)), dim=-1)
        return self.norm((torch.stack(branches, dim=2) * weights.unsqueeze(-1)).sum(dim=2))


class HorizonRefiner(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        self.input_norm = nn.LayerNorm(config.d_model)
        self.condition = ConditionalLayerNorm(config.d_model, config.d_model)
        self.gru = nn.GRU(config.d_model, config.d_model // 2, bidirectional=True, batch_first=True)
        self.output_projection = nn.Linear(config.d_model, config.d_model)
        self.layer_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        refined, _ = self.gru(self.condition(self.input_norm(values.float()), condition.float()))
        return values.float() + self.layer_scale.float() * self.output_projection(refined)


class FactorizedHorizonHead(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        hidden_size = 64
        rank = config.horizon_head_rank
        self.shared_projection = nn.Sequential(nn.Linear(config.d_model, hidden_size), nn.GELU())
        self.resource_weight = nn.Parameter(torch.empty(N_RESOURCES, hidden_size))
        self.horizon_factor = nn.Parameter(torch.empty(FORECAST_STEPS, rank))
        self.resource_factor = nn.Parameter(torch.empty(N_RESOURCES, rank, hidden_size))
        self.bias = nn.Parameter(torch.zeros(N_RESOURCES, FORECAST_STEPS))
        nn.init.xavier_uniform_(self.resource_weight)
        nn.init.xavier_uniform_(self.horizon_factor)
        nn.init.xavier_uniform_(self.resource_factor)

    def forward(self, values: Tensor, resource_id: Tensor) -> Tensor:
        hidden = self.shared_projection(values.float())
        resource = resource_id.long()
        base = self.resource_weight[resource][:, None, :]
        factor = torch.einsum("hr,brd->bhd", self.horizon_factor, self.resource_factor[resource])
        return (hidden * (base + factor)).sum(dim=-1) + self.bias[resource]


class DualResolutionWeatherEncoder(nn.Module):
    """A high-resolution and a four-step pooled weather encoder."""

    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        hidden = config.temporal_hidden
        self.input_projection = nn.Linear(N_SOURCES * N_CHANNELS + N_SOURCES + 5, 2 * hidden)
        self.high_resolution_gru = nn.GRU(2 * hidden, hidden, bidirectional=True, batch_first=True)
        self.low_resolution_gru = nn.GRU(2 * hidden, hidden, bidirectional=True, batch_first=True)
        self.output_projection = nn.Linear(4 * hidden, config.d_model)

    @staticmethod
    def _pool_four(values: Tensor) -> Tensor:
        if values.ndim != 3 or values.size(1) != FORECAST_STEPS:
            raise InputShapeError("Temporal encoder expects 96 forecast steps")
        return values.reshape(values.size(0), 24, 4, values.size(2)).mean(dim=2)

    def forward(self, nwp: Tensor, availability: Tensor, future_time: Tensor) -> Tensor:
        scalar = nwp.float().reshape(-1, FORECAST_STEPS, N_SOURCES, N_CHANNELS)
        mask = availability.bool()
        scalar = scalar * mask[:, None, :, None].to(scalar.dtype)
        inputs = torch.cat(
            [
                scalar.reshape(-1, FORECAST_STEPS, N_SOURCES * N_CHANNELS),
                mask[:, None, :].expand(-1, FORECAST_STEPS, -1).float(),
                future_time.float(),
            ],
            dim=-1,
        )
        projected = self.input_projection(inputs)
        high, _ = self.high_resolution_gru(projected)
        low, _ = self.low_resolution_gru(self._pool_four(projected))
        return self.output_projection(torch.cat([high, low.repeat_interleave(4, dim=1)], dim=-1)).float()


class StationRegimeHorizonInteraction(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        rank = config.interaction_rank
        self.station_factor = nn.Parameter(torch.empty(N_STATIONS, rank))
        self.resource_factor = nn.Parameter(torch.empty(N_RESOURCES, rank))
        self.horizon_factor = nn.Parameter(torch.empty(FORECAST_STEPS, rank))
        self.regime = nn.Sequential(nn.Linear(2 * config.d_model, 64), nn.GELU(), nn.Linear(64, rank))
        self.output_projection = nn.Linear(rank, config.d_model)
        nn.init.xavier_uniform_(self.station_factor)
        nn.init.xavier_uniform_(self.resource_factor)
        nn.init.xavier_uniform_(self.horizon_factor)
        nn.init.xavier_uniform_(self.regime[0].weight)
        nn.init.zeros_(self.regime[0].bias)
        nn.init.xavier_uniform_(self.regime[2].weight)
        nn.init.zeros_(self.regime[2].bias)

    def forward(self, temporal: Tensor, condition: Tensor, station_id: Tensor, resource_id: Tensor) -> Tensor:
        station = station_id.long()
        resource = resource_id.long()
        regime = torch.sigmoid(self.regime(torch.cat([temporal.float(), condition.float()], dim=-1)))
        product = self.station_factor[station][:, None, :] * self.resource_factor[resource][:, None, :] * self.horizon_factor[None, :, :] * regime
        return self.output_projection(product).float()


class CorrectionHead(nn.Module):
    def __init__(self, config: MSWPFuseConfig):
        super().__init__()
        width = 4 * config.d_model
        self.temporal_delta = nn.Linear(width, 1)
        self.correction_gate = nn.Linear(width, 1)
        nn.init.xavier_uniform_(self.temporal_delta.weight)
        nn.init.zeros_(self.temporal_delta.bias)
        nn.init.zeros_(self.correction_gate.weight)
        nn.init.constant_(self.correction_gate.bias, config.initial_correction_gate_bias)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        return torch.tanh(self.temporal_delta(values.float()).squeeze(-1)), torch.sigmoid(self.correction_gate(values.float()).squeeze(-1))


class MSWPFuse(nn.Module):
    """MSWP-Fuse with the selected station-regime adaptive residual."""

    def __init__(
        self,
        config: MSWPFuseConfig | Mapping[str, Any] | None = None,
        *,
        nwp_mean: Tensor | None = None,
        nwp_std: Tensor | None = None,
    ):
        super().__init__()
        self.config = config if isinstance(config, MSWPFuseConfig) else MSWPFuseConfig.from_mapping(config)
        self.config.validate()
        mean = torch.zeros(N_SOURCES * N_CHANNELS) if nwp_mean is None else torch.as_tensor(nwp_mean, dtype=torch.float32).flatten()
        std = torch.ones(N_SOURCES * N_CHANNELS) if nwp_std is None else torch.as_tensor(nwp_std, dtype=torch.float32).flatten()
        if mean.shape != (N_SOURCES * N_CHANNELS,) or std.shape != mean.shape or not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
            raise ValueError("NWP scaler statistics must contain 27 finite values with positive standard deviations")
        self.register_buffer("nwp_mean", mean)
        self.register_buffer("nwp_std", std)
        self.conditioner = StationResourceConditioner(self.config.d_model)
        self.lag_history = LagBankHistoryEncoder(self.config.lag_hidden, self.config.d_model)
        self.patch_history = HistoryPatchEncoder(self.config)
        self.history_query = HorizonHistoryQuery(self.config)
        self.scalar_provider = ScalarProviderEncoder(self.config)
        self.provider_fusion = MaskedProviderSetFusion(self.config)
        self.aligned_fusion = AlignedFeatureFusion(self.config)
        self.refiner = HorizonRefiner(self.config) if self.config.use_refiner else None
        self.head = FactorizedHorizonHead(self.config)
        self.temporal_weather = DualResolutionWeatherEncoder(self.config)
        self.station_regime_interaction = StationRegimeHorizonInteraction(self.config)
        self.inactive_head = nn.Linear(4 * self.config.d_model, 1)
        nn.init.zeros_(self.inactive_head.weight)
        nn.init.constant_(self.inactive_head.bias, self.config.initial_inactive_bias)
        self.correction_head = CorrectionHead(self.config)
        initial_logit = math.log(self.config.initial_scale_fraction / (1.0 - self.config.initial_scale_fraction))
        self.scale_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
        self._source_dropout_probability = 0.05

    @property
    def bounded_scale(self) -> Tensor:
        return self.config.bounded_scale_maximum * torch.sigmoid(self.scale_logit)

    def set_source_dropout(self, probability: float) -> None:
        if not 0.0 <= probability < 1.0:
            raise ValueError("Source dropout must lie in [0, 1)")
        self._source_dropout_probability = float(probability)

    def station_delta_penalty(self) -> Tensor:
        return self.conditioner.station_delta_penalty()

    def _parent_states(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        _check_inputs(batch)
        source_mask = make_source_mask(batch["nwp_availability"], self._source_dropout_probability, self.training)
        condition = self.conditioner(batch["station_id"], batch["resource_id"], batch["future_time"].float())
        lag = self.lag_history(batch["hist_power"], batch["hist_time"], condition)
        history = self.history_query(condition, self.patch_history(batch["hist_power"], batch["hist_time"]))
        scalar = batch["nwp"].reshape(-1, FORECAST_STEPS, N_SOURCES, N_CHANNELS).float()
        scalar = scalar * source_mask[:, None, :, None].to(scalar.dtype)
        raw_scalar = inverse_standardize_scalar(scalar, self.nwp_mean, self.nwp_std, source_mask)
        provider_inputs = torch.cat(
            [
                scalar,
                source_disagreement(scalar, source_mask),
                scalar_physics_features(raw_scalar) * source_mask[:, None, :, None].to(scalar.dtype),
                source_mask[:, None, :, None].to(scalar.dtype).expand(-1, FORECAST_STEPS, -1, 1),
                batch["future_time"].unsqueeze(2).expand(-1, -1, N_SOURCES, -1),
            ],
            dim=-1,
        )
        provider = self.scalar_provider(provider_inputs, condition)
        weather = self.provider_fusion(provider, source_mask, condition)
        base_state = self.aligned_fusion(history, lag, weather, condition)
        if self.refiner is not None:
            base_state = self.refiner(base_state, condition)
        return {
            "condition": condition,
            "source_mask": source_mask,
            "raw_scalar": raw_scalar,
            "base_state": base_state,
            "raw_base": self.head(base_state, batch["resource_id"]),
        }

    def forward_parent(self, batch: Mapping[str, Tensor]) -> Tensor:
        states = self._parent_states(batch)
        return apply_solar_darkness(smooth_nonnegative(states["raw_base"]), states["raw_scalar"], states["source_mask"], batch["resource_id"])

    def forward_features(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        states = self._parent_states(batch)
        temporal = self.temporal_weather(batch["nwp"], states["source_mask"], batch["future_time"])
        interaction = self.station_regime_interaction(temporal, states["condition"], batch["station_id"], batch["resource_id"])
        correction_features = torch.cat([states["base_state"].float(), temporal.float(), interaction.float(), states["condition"].float()], dim=-1)
        inactive_logit = self.inactive_head(correction_features).squeeze(-1)
        inactive_probability = torch.sigmoid(inactive_logit)
        temporal_delta, correction_gate = self.correction_head(correction_features)
        raw = states["raw_base"].float() + self.bounded_scale.float() * correction_gate * temporal_delta
        prediction = (1.0 - inactive_probability) * smooth_nonnegative(raw)
        return {
            **states,
            "temporal_weather_state": temporal,
            "station_regime_interaction": interaction,
            "inactive_logit": inactive_logit,
            "inactive_probability": inactive_probability,
            "correction_gate": correction_gate,
            "temporal_delta": temporal_delta,
            "prediction": apply_solar_darkness(prediction, states["raw_scalar"], states["source_mask"], batch["resource_id"]),
        }

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.forward_features(batch)["prediction"]


MSWPFuseSRATR = MSWPFuse

__all__ = [
    "AMENDMENT_PREFIXES",
    "FORECAST_STEPS",
    "HISTORY_STEPS",
    "MSWPFuse",
    "MSWPFuseConfig",
    "MSWPFuseSRATR",
    "InputShapeError",
]
