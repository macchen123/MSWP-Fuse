"""The standalone MSWP-Fuse SRATR forecasting model and canonical parent backbone."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .primitives import (
    AMENDMENT_PREFIXES,
    FORECAST_STEPS,
    HISTORY_STEPS,
    HORIZON,
    N_CHANNELS,
    N_RESOURCES,
    N_SOURCES,
    N_STATIONS,
    ConditionalLayerNorm,
    CrossAttentionBlock,
    InputShapeError,
    PositionalEncoding,
    TransformerBlock,
    apply_solar_darkness,
    check_scalar_inputs,
    fp32_cuda_context,
    inverse_standardize_scalar,
    make_joint_source_mask,
    scalar_physics_features,
    smooth_nonnegative,
    source_disagreement,
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
    source_dropout: float = 0.05
    use_refiner: bool = True
    temporal_hidden: int = 24
    interaction_rank: int = 12
    bounded_scale_maximum: float = 0.15
    initial_scale_fraction: float = 0.4125854203825821
    initial_inactive_bias: float = -6.0
    initial_correction_gate_bias: float = -4.0
    n_stations: int = 10
    n_resources: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None = None) -> MSWPFuseConfig:
        if value is None:
            return cls()
        data = dict(value)
        if "history_patch_len" in data and "history_patch_length" not in data:
            data["history_patch_length"] = data["history_patch_len"]
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: data[key] for key in data if key in allowed})

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.history_patch_length != 8 or self.history_patch_stride != 8:
            raise ValueError("The public model uses non-overlapping 8-step history patches")
        if self.d_model != 128 or self.lag_hidden != 64:
            raise ValueError("The public model uses the paper dimensions")
        if not 0.0 < self.initial_scale_fraction < 1.0:
            raise ValueError("initial_scale_fraction must lie between zero and one")
        if self.bounded_scale_maximum <= 0.0:
            raise ValueError("bounded_scale_maximum must be positive")
        if self.source_dropout < 0.0 or self.source_dropout >= 1.0:
            raise ValueError("source_dropout must lie in [0, 1)")


class StationResourceConditioner(nn.Module):
    def __init__(
        self,
        n_stations: int = 10,
        d_model: int = 128,
        n_resources: int = 2,
        condition_dim: int = 64,
    ) -> None:
        super().__init__()
        self.n_stations = n_stations
        self.n_resources = n_resources
        self.resource_embedding = nn.Embedding(n_resources, condition_dim)
        self.station_delta = nn.Embedding(n_stations, condition_dim)
        self.time_projection = nn.Linear(5, condition_dim)
        self.horizon_embedding = nn.Parameter(torch.randn(1, HORIZON, condition_dim) * 0.02)
        self.output_projection = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(
        self,
        station_id: Tensor,
        resource_id: Tensor,
        future_time: Tensor,
    ) -> Tensor:
        station = station_id.long()
        resource = resource_id.long()
        if station.min().item() < 0 or station.max().item() >= self.n_stations:
            raise InputShapeError("station_id is outside the configured range")
        if resource.min().item() < 0 or resource.max().item() >= self.n_resources:
            raise InputShapeError("resource_id is outside the configured range")
        values = (
            self.resource_embedding(resource)
            + self.station_delta(station)
        ).unsqueeze(1)
        values = values + self.time_projection(future_time)
        values = values + self.horizon_embedding
        return self.output_projection(values)

    def station_delta_penalty(self) -> Tensor:
        return self.station_delta.weight.square().mean()


class LagBankHistoryEncoder(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, condition_dim: int = 128) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)

    def forward(
        self,
        hist_power: Tensor,
        hist_time: Tensor,
        condition: Tensor,
    ) -> Tensor:
        if hist_power.shape[-2:] != (HISTORY_STEPS, 2):
            raise InputShapeError("LagBank received an unexpected hist_power shape")
        if hist_time.shape[-2:] != (HISTORY_STEPS, 5):
            raise InputShapeError("LagBank received an unexpected hist_time shape")
        values = torch.cat([hist_power, hist_time], dim=-1)
        values = values.reshape(values.size(0), 7, HORIZON, self.input_dim)
        values = values.transpose(1, 2).contiguous()
        values = values.reshape(values.size(0) * HORIZON, 7, self.input_dim)
        _, state = self.gru(values)
        output = state[-1].reshape(-1, HORIZON, self.hidden_dim)
        return output + self.condition_projection(condition)


class HistoryPatchEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        patch_len: int = 8,
        patch_stride: int = 8,
        layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 512,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        if patch_len != 8 or patch_stride != 8:
            raise ValueError("History transfer requires patch_len=8 and patch_stride=8")
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.history_patch_projection = nn.Linear(patch_len, d_model)
        self.history_time_projection = nn.Linear(5, d_model, bias=False)
        self.history_channel_embedding = nn.Embedding(2, d_model)
        self.history_position = PositionalEncoding(d_model)
        self.history_encoder = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(layers)]
        )

    def forward(self, hist_power: Tensor, hist_time: Tensor) -> Tensor:
        history = hist_power.transpose(1, 2)
        patches = history.unfold(2, self.patch_len, self.patch_stride)
        time_patches = (
            hist_time.transpose(1, 2)
            .unfold(2, self.patch_len, self.patch_stride)
            .mean(dim=-1)
            .transpose(1, 2)
        )
        tokens = self.history_patch_projection(patches)
        tokens = tokens + self.history_time_projection(time_patches).unsqueeze(1)
        channels = self.history_channel_embedding(
            torch.arange(2, device=history.device)
        )
        tokens = tokens + channels.view(1, 2, 1, -1)
        tokens = tokens.flatten(1, 2)
        tokens = self.history_position(tokens)
        for layer in self.history_encoder:
            tokens = layer(tokens)
        return tokens

    def load_history_subset(self, state_dict: Mapping[str, Tensor]) -> tuple[str, ...]:
        allowed = (
            "history_patch_projection",
            "history_time_projection",
            "history_channel_embedding",
            "history_position",
            "history_encoder",
        )
        cleaned = {
            name[len("patch_history."):] if name.startswith("patch_history.") else name: value
            for name, value in state_dict.items()
        }
        selected = {
            name: value
            for name, value in cleaned.items()
            if name.startswith(allowed)
        }
        expected = {
            name: value
            for name, value in self.state_dict().items()
            if name.startswith(allowed)
        }
        if set(selected) != set(expected):
            missing = sorted(set(expected) - set(selected))
            extra = sorted(set(selected) - set(expected))
            raise ValueError(f"History transfer key mismatch; missing={missing}, extra={extra}")
        for name, value in expected.items():
            if tuple(selected[name].shape) != tuple(value.shape):
                raise ValueError(
                    f"History transfer shape mismatch for {name}: "
                    f"{tuple(selected[name].shape)} != {tuple(value.shape)}"
                )
        self.load_state_dict(selected, strict=False)
        return tuple(sorted(selected))


class HorizonHistoryQuery(nn.Module):
    def __init__(self, d_model: int = 128, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.10) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, HORIZON, d_model) * 0.02)
        self.attention = CrossAttentionBlock(d_model, n_heads, d_ff, dropout)

    def forward(self, condition: Tensor, memory: Tensor) -> Tensor:
        query = self.query.expand(condition.size(0), -1, -1) + condition
        return self.attention(query, memory)


class ScalarProviderEncoder(nn.Module):
    def __init__(self, input_dim: int = 36, d_model: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, d_model)
        self.condition = ConditionalLayerNorm(d_model, d_model)

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        with fp32_cuda_context(values.device):
            encoded = self.input_projection(values.float())
            encoded = F.gelu(self.norm(encoded))
            encoded = self.dropout(encoded)
            encoded = self.output_projection(encoded)
            return self.condition(encoded, condition.float().unsqueeze(2))


class MaskedProviderSetFusion(nn.Module):
    def __init__(self, d_model: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.projection = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.condition = ConditionalLayerNorm(d_model, d_model)

    def forward(self, values: Tensor, source_mask: Tensor, condition: Tensor) -> Tensor:
        if values.ndim != 4 or values.shape[2] != N_SOURCES:
            raise InputShapeError("Provider values must have shape (B, 96, 3, D)")
        with fp32_cuda_context(values.device):
            values = values.float()
            condition = condition.float()
            batch_size = values.size(0)
            valid = source_mask.bool()
            safe_valid = valid.clone()
            no_source = ~safe_valid.any(dim=1)
            if no_source.any():
                safe_valid[no_source, 0] = True
            flattened = values.reshape(batch_size * HORIZON, N_SOURCES, -1)
            key_padding_mask = ~safe_valid[:, None, :].expand(batch_size, HORIZON, N_SOURCES)
            key_padding_mask = key_padding_mask.reshape(batch_size * HORIZON, N_SOURCES)
            attended = self.attention(
                flattened,
                flattened,
                flattened,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )[0].reshape(batch_size, HORIZON, N_SOURCES, -1)
            mask = valid[:, None, :, None].to(attended.dtype)
            count = mask.sum(dim=2).clamp_min(1.0)
            mean = (attended * mask).sum(dim=2) / count
            variance = ((attended - mean.unsqueeze(2)).square() * mask).sum(dim=2) / count
            standard_deviation = torch.sqrt(variance.clamp_min(0.0) + 1.0e-8)
            output = self.projection(torch.cat([mean, standard_deviation], dim=-1))
            output = self.condition(output, condition)
            return output * valid.any(dim=1).to(output.dtype)[:, None, None]


class AlignedFeatureFusion(nn.Module):
    def __init__(self, d_model: int = 128, lag_dim: int = 64) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [nn.Linear(d_model, d_model), nn.Linear(lag_dim, d_model)]
            + [nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)]
        )
        self.gate = nn.Linear(4 * d_model, 4)
        self.output_norm = nn.LayerNorm(d_model)

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
            stacked = torch.stack(branches, dim=2)
            weights = torch.softmax(self.gate(torch.cat(branches, dim=-1)), dim=-1)
            return self.output_norm((stacked * weights.unsqueeze(-1)).sum(dim=2))


class HorizonRefiner(nn.Module):
    def __init__(self, d_model: int = 128, hidden_size: int = 64, layer_scale_init: float = 0.10) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(d_model)
        self.condition = ConditionalLayerNorm(d_model, d_model)
        self.gru = nn.GRU(
            d_model,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.output_projection = nn.Linear(2 * hidden_size, d_model)
        self.layer_scale = nn.Parameter(torch.tensor(float(layer_scale_init)))

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        with fp32_cuda_context(values.device):
            refined, _ = self.gru(
                self.condition(self.input_norm(values.float()), condition.float())
            )
            refined = self.output_projection(refined)
            return values.float() + self.layer_scale.float() * refined


class FactorizedHorizonHead(nn.Module):
    def __init__(self, d_model: int = 128, hidden_dim: int = 64, rank: int = 16, resources: int = 2) -> None:
        super().__init__()
        self.shared_projection = nn.Sequential(nn.Linear(d_model, hidden_dim), nn.GELU())
        self.resource_weight = nn.Parameter(torch.empty(resources, hidden_dim))
        self.horizon_factor = nn.Parameter(torch.empty(HORIZON, rank))
        self.resource_factor = nn.Parameter(torch.empty(resources, rank, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(resources, HORIZON))
        nn.init.xavier_uniform_(self.resource_weight)
        nn.init.xavier_uniform_(self.horizon_factor)
        nn.init.xavier_uniform_(self.resource_factor)

    def forward(self, values: Tensor, resource_id: Tensor) -> Tensor:
        with fp32_cuda_context(values.device):
            hidden = self.shared_projection(values.float())
            resource = resource_id.long()
            base = self.resource_weight[resource][:, None, :]
            factor = torch.einsum(
                "hr,brd->bhd",
                self.horizon_factor,
                self.resource_factor[resource],
            )
            weights = base + factor
            bias = self.bias[resource]
            return (hidden * weights).sum(dim=-1) + bias


class DualResolutionWeatherTemporalEncoder(nn.Module):
    """Deterministic 96/24 scalar-weather sequence encoder."""

    def __init__(
        self,
        *,
        input_dimension: int = N_SOURCES * N_CHANNELS + N_SOURCES + 5,
        hidden_size: int = 32,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dimension, 2 * hidden_size)
        self.high_resolution_gru = nn.GRU(
            2 * hidden_size,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.low_resolution_gru = nn.GRU(
            2 * hidden_size,
            hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        self.output_projection = nn.Linear(4 * hidden_size, d_model)

    @staticmethod
    def pool_four(values: Tensor) -> Tensor:
        if values.ndim != 3 or values.size(1) != HORIZON:
            raise InputShapeError(
                f"Temporal pooling expects (B, 96, C), got {tuple(values.shape)}"
            )
        return values.reshape(values.size(0), 24, 4, values.size(2)).mean(dim=2)

    @staticmethod
    def interpolate_to_96(values: Tensor) -> Tensor:
        if values.ndim != 3 or values.size(1) != 24:
            raise InputShapeError(
                f"Temporal interpolation expects (B, 24, C), got {tuple(values.shape)}"
            )
        return values.repeat_interleave(4, dim=1)

    def forward(
        self,
        nwp: Tensor,
        provider_availability: Tensor,
        future_time: Tensor,
    ) -> Tensor:
        if nwp.ndim != 3 or tuple(nwp.shape[1:]) != (HORIZON, N_SOURCES * N_CHANNELS):
            raise InputShapeError(f"Temporal nwp shape is invalid: {tuple(nwp.shape)}")
        if provider_availability.ndim != 2 or provider_availability.size(1) != N_SOURCES:
            raise InputShapeError(
                f"Temporal provider availability shape is invalid: {tuple(provider_availability.shape)}"
            )
        if future_time.ndim != 3 or tuple(future_time.shape[1:]) != (HORIZON, 5):
            raise InputShapeError(
                f"Temporal future_time shape is invalid: {tuple(future_time.shape)}"
            )
        if nwp.size(0) != provider_availability.size(0) or nwp.size(0) != future_time.size(0):
            raise InputShapeError("Temporal weather batch dimensions do not agree")
        with fp32_cuda_context(nwp.device):
            scalar = nwp.float().reshape(-1, HORIZON, N_SOURCES, N_CHANNELS)
            availability = provider_availability.bool()
            scalar = scalar * availability[:, None, :, None].to(scalar.dtype)
            masked = scalar.reshape(-1, HORIZON, N_SOURCES * N_CHANNELS)
            available_steps = availability[:, None, :].to(torch.float32).expand(-1, HORIZON, -1)
            sequence = torch.cat([masked, available_steps, future_time.float()], dim=-1)
            projected = self.input_projection(sequence)
            low_input = self.pool_four(projected)
            high_state, _ = self.high_resolution_gru(projected)
            low_state, _ = self.low_resolution_gru(low_input)
            low_state = self.interpolate_to_96(low_state)
            joined = torch.cat([high_state, low_state], dim=-1)
            return self.output_projection(joined).float()


class StationRegimeHorizonInteraction(nn.Module):
    """Rank-r station/resource/horizon factors modulated by a weather regime."""

    def __init__(
        self,
        *,
        n_stations: int = 10,
        n_resources: int = 2,
        rank: int = 16,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.n_stations = int(n_stations)
        self.n_resources = int(n_resources)
        self.rank = int(rank)
        self.station_factor = nn.Parameter(torch.empty(n_stations, rank))
        self.resource_factor = nn.Parameter(torch.empty(n_resources, rank))
        self.horizon_factor = nn.Parameter(torch.empty(HORIZON, rank))
        self.regime_network = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.GELU(),
            nn.Linear(64, rank),
        )
        self.output_projection = nn.Linear(rank, d_model)
        nn.init.xavier_uniform_(self.station_factor)
        nn.init.xavier_uniform_(self.resource_factor)
        nn.init.xavier_uniform_(self.horizon_factor)
        nn.init.xavier_uniform_(self.regime_network[0].weight)
        nn.init.zeros_(self.regime_network[0].bias)
        nn.init.xavier_uniform_(self.regime_network[2].weight)
        nn.init.zeros_(self.regime_network[2].bias)

    def forward(
        self,
        temporal_weather_state: Tensor,
        condition: Tensor,
        station_id: Tensor,
        resource_id: Tensor,
    ) -> Tensor:
        if temporal_weather_state.ndim != 3 or tuple(temporal_weather_state.shape[1:]) != (
            HORIZON,
            128,
        ):
            raise InputShapeError("Temporal state shape is invalid for interaction")
        if condition.shape != temporal_weather_state.shape:
            raise InputShapeError("Condition and temporal state shapes must agree")
        station = station_id.long()
        resource = resource_id.long()
        if station.ndim != 1 or station.size(0) != temporal_weather_state.size(0):
            raise InputShapeError("station_id must have shape (B,)")
        if resource.ndim != 1 or resource.size(0) != temporal_weather_state.size(0):
            raise InputShapeError("resource_id must have shape (B,)")
        if bool((station < 0).any()) or bool((station >= self.n_stations).any()):
            raise InputShapeError("station_id is outside the canonical 0..9 range")
        if bool((resource < 0).any()) or bool((resource >= self.n_resources).any()):
            raise InputShapeError("resource_id is outside the configured range")

        with fp32_cuda_context(temporal_weather_state.device):
            regime = torch.sigmoid(
                self.regime_network(
                    torch.cat([temporal_weather_state.float(), condition.float()], dim=-1)
                )
            )
            station_factor = self.station_factor[station][:, None, :]
            resource_factor = self.resource_factor[resource][:, None, :]
            horizon_factor = self.horizon_factor[None, :, :]
            product = station_factor * resource_factor * horizon_factor * regime
            return self.output_projection(product).float()


class M5CorrectionHead(nn.Module):
    """Magnitude residual and reliability gate over the four-state feature tuple."""

    def __init__(self, input_dimension: int = 512) -> None:
        super().__init__()
        self.temporal_delta = nn.Sequential(nn.Linear(input_dimension, 1))
        self.correction_gate = nn.Sequential(nn.Linear(input_dimension, 1))
        gate = self.correction_gate[-1]
        nn.init.xavier_uniform_(self.temporal_delta[-1].weight)
        nn.init.zeros_(self.temporal_delta[-1].bias)
        nn.init.zeros_(gate.weight)
        nn.init.constant_(gate.bias, -3.0)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        if values.ndim != 3 or values.size(-1) != 512:
            raise InputShapeError("Correction features must have shape (B, 96, 512)")
        delta = torch.tanh(self.temporal_delta(values.float()).squeeze(-1))
        gate = torch.sigmoid(self.correction_gate(values.float()).squeeze(-1))
        return delta, gate


class MSWPFuseBackbone(nn.Module):
    """The canonical 119-tensor parent MSWP-Fuse backbone model."""

    def __init__(
        self,
        config: MSWPFuseConfig | Mapping[str, Any] | None = None,
        *,
        nwp_mean: Tensor | None = None,
        nwp_std: Tensor | None = None,
        history_state_dict: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__()
        self.config = config if isinstance(config, MSWPFuseConfig) else MSWPFuseConfig.from_mapping(config)
        self.config.validate()
        if nwp_mean is None or nwp_std is None:
            raise ValueError("Fold-specific scalar NWP scaler statistics are required")
        mean = torch.as_tensor(nwp_mean, dtype=torch.float32).flatten()
        std = torch.as_tensor(nwp_std, dtype=torch.float32).flatten()
        if mean.shape != (N_SOURCES * N_CHANNELS,) or std.shape != mean.shape:
            raise ValueError("NWP scaler statistics must have shape (27,)")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()) or bool((std <= 0).any()):
            raise ValueError("NWP scaler statistics must be finite and positive")
        self._source_dropout_probability = float(self.config.source_dropout)
        self.register_buffer("nwp_mean", mean)
        self.register_buffer("nwp_std", std)
        self.register_buffer("daylight_threshold", torch.tensor(1.0e-6))
        self.conditioner = StationResourceConditioner(
            n_stations=self.config.n_stations,
            d_model=self.config.d_model,
            n_resources=self.config.n_resources,
        )
        self.lag_history = LagBankHistoryEncoder(hidden_dim=self.config.lag_hidden)
        self.patch_history = HistoryPatchEncoder(
            d_model=self.config.d_model,
            patch_len=self.config.history_patch_length,
            patch_stride=self.config.history_patch_stride,
            layers=self.config.history_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
        )
        self.history_query = HorizonHistoryQuery(
            self.config.d_model,
            self.config.n_heads,
            self.config.d_ff,
            self.config.dropout,
        )
        self.scalar_provider = ScalarProviderEncoder(
            input_dim=36,
            d_model=self.config.d_model,
            dropout=self.config.dropout,
        )
        self.field_tokenizer = None
        self.local_field_fusion = None
        self.provider_fusion = MaskedProviderSetFusion(
            d_model=self.config.d_model,
            heads=self.config.provider_heads,
        )
        self.aligned_fusion = AlignedFeatureFusion(
            d_model=self.config.d_model,
            lag_dim=self.config.lag_hidden,
        )
        self.refiner = (
            HorizonRefiner(
                d_model=self.config.d_model,
                hidden_size=self.config.d_model // 2,
                layer_scale_init=0.10,
            )
            if self.config.use_refiner
            else None
        )
        self.head = FactorizedHorizonHead(
            d_model=self.config.d_model,
            hidden_dim=64,
            rank=self.config.horizon_head_rank,
            resources=self.config.n_resources,
        )
        self._loaded_history_keys: tuple[str, ...] = ()
        if history_state_dict is not None:
            self._loaded_history_keys = self.patch_history.load_history_subset(history_state_dict)

    def set_source_dropout(self, probability: float) -> None:
        if probability < 0.0 or probability >= 1.0:
            raise ValueError("source dropout probability must be in [0, 1)")
        self._source_dropout_probability = float(probability)

    def station_delta_penalty(self) -> Tensor:
        return self.conditioner.station_delta_penalty()

    @property
    def loaded_history_keys(self) -> tuple[str, ...]:
        return self._loaded_history_keys

    @property
    def requires_history_initializer(self) -> bool:
        return True

    def _history_states(
        self, batch: Mapping[str, Tensor], condition: Tensor
    ) -> tuple[Tensor, Tensor]:
        lag = self.lag_history(
            batch["hist_power"],
            batch["hist_time"],
            condition,
        )
        memory = self.patch_history(batch["hist_power"], batch["hist_time"])
        return self.history_query(condition, memory), lag

    def _weather_states(
        self,
        batch: Mapping[str, Tensor],
        scalar: Tensor,
        raw_scalar: Tensor,
        source_mask: Tensor,
        condition: Tensor,
    ) -> dict[str, Tensor]:
        disagreement = source_disagreement(scalar, source_mask)
        physics = scalar_physics_features(raw_scalar)
        physics = physics * source_mask[:, None, :, None].to(physics.dtype)
        availability = source_mask[:, None, :, None].to(scalar.dtype).expand(
            -1, HORIZON, -1, 1
        )
        future_time = batch["future_time"].unsqueeze(2).expand(-1, -1, N_SOURCES, -1)
        provider_inputs = torch.cat(
            [scalar, disagreement, physics, availability, future_time],
            dim=-1,
        )
        provider = self.scalar_provider(provider_inputs, condition)
        weather = self.provider_fusion(provider, source_mask, condition)
        return {"provider_state": provider, "fused_weather": weather}

    def _pre_output(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        check_scalar_inputs(batch)
        source_mask = make_joint_source_mask(
            batch["nwp_availability"],
            probability=self._source_dropout_probability,
            training=self.training,
        )
        with fp32_cuda_context(batch["future_time"].device):
            condition = self.conditioner(
                batch["station_id"],
                batch["resource_id"],
                batch["future_time"].float(),
            )
        history, lag = self._history_states(batch, condition)
        scalar = batch["nwp"].reshape(-1, HORIZON, N_SOURCES, N_CHANNELS).float()
        scalar = scalar * source_mask[:, None, :, None].to(scalar.dtype)
        raw_scalar = inverse_standardize_scalar(
            scalar,
            self.nwp_mean.to(scalar),
            self.nwp_std.to(scalar),
            source_mask,
        )
        weather_states = self._weather_states(batch, scalar, raw_scalar, source_mask, condition)
        aligned_base_state = self.aligned_fusion(history, lag, weather_states["fused_weather"], condition)
        if self.refiner is not None:
            refiner_state = self.refiner(aligned_base_state, condition)
        else:
            refiner_state = aligned_base_state
        raw_base = self.head(refiner_state, batch["resource_id"])
        return {
            "condition": condition,
            "source_mask": source_mask,
            "masked_scalar_nwp": scalar,
            "raw_scalar": raw_scalar,
            **weather_states,
            "aligned_base_state": aligned_base_state,
            "refiner_state": refiner_state,
            "base_state": refiner_state,
            "raw_base": raw_base,
        }

    def forward_features(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        states = self._pre_output(batch)
        prediction = smooth_nonnegative(states["raw_base"])
        prediction = apply_solar_darkness(
            prediction,
            states["raw_scalar"],
            states["source_mask"],
            batch["resource_id"],
            threshold=float(self.daylight_threshold.item()),
        )
        return {**states, "prediction": prediction}

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.forward_features(batch)["prediction"]


class MSWPFuse(MSWPFuseBackbone):
    """Station- and regime-adaptive temporal residual model with six configurations."""

    def __init__(
        self,
        config: MSWPFuseConfig | Mapping[str, Any] | None = None,
        *,
        nwp_mean: Tensor | None = None,
        nwp_std: Tensor | None = None,
        history_state_dict: Mapping[str, Tensor] | None = None,
    ) -> None:
        super().__init__(
            config,
            nwp_mean=nwp_mean,
            nwp_std=nwp_std,
            history_state_dict=history_state_dict,
        )
        # 1. Base M5 / R15 topology
        self.temporal_weather = DualResolutionWeatherTemporalEncoder(hidden_size=32)
        self.station_regime_interaction = StationRegimeHorizonInteraction(
            n_stations=self.config.n_stations,
            n_resources=self.config.n_resources,
            rank=16,
        )
        self.inactive_head = nn.Sequential(nn.Linear(512, 1))
        nn.init.zeros_(self.inactive_head[-1].weight)
        nn.init.constant_(self.inactive_head[-1].bias, -4.0)
        self.correction_head = M5CorrectionHead(512)
        self.scale_logit = nn.Parameter(
            torch.tensor(-2.1972245773362196, dtype=torch.float32)
        )

        # 2. Selected stage2 c00 configuration (FullTunableSRATR)
        self.temporal_weather = DualResolutionWeatherTemporalEncoder(hidden_size=24)
        self.station_regime_interaction = StationRegimeHorizonInteraction(
            n_stations=self.config.n_stations,
            n_resources=self.config.n_resources,
            rank=12,
        )
        nn.init.constant_(self.inactive_head[-1].bias, -6.0)
        nn.init.constant_(self.correction_head.correction_gate[-1].bias, -4.0)
        with torch.no_grad():
            self.scale_logit.fill_(math.log(0.4125854203825821 / (1.0 - 0.4125854203825821)))
        self._tuning_scale_maximum = 0.15

        # 3. CandidateFull candidate overrides
        if self.config.temporal_hidden != 24:
            self.temporal_weather = DualResolutionWeatherTemporalEncoder(
                hidden_size=int(self.config.temporal_hidden)
            )
        if self.config.interaction_rank != 12:
            self.station_regime_interaction = StationRegimeHorizonInteraction(
                n_stations=self.config.n_stations,
                n_resources=self.config.n_resources,
                rank=int(self.config.interaction_rank),
            )
        nn.init.constant_(self.inactive_head[-1].bias, float(self.config.initial_inactive_bias))
        nn.init.constant_(self.correction_head.correction_gate[-1].bias, float(self.config.initial_correction_gate_bias))
        with torch.no_grad():
            fraction = float(self.config.initial_scale_fraction)
            self.scale_logit.fill_(math.log(fraction / (1.0 - fraction)))
        self._tuning_scale_maximum = float(self.config.bounded_scale_maximum)

        self._last_inactive_logit_for_loss: Tensor | None = None
        self._last_inactive_probability_for_loss: Tensor | None = None
        self._last_active_gate_for_loss: Tensor | None = None
        self._last_correction_gate_for_loss: Tensor | None = None
        self._last_temporal_delta_for_loss: Tensor | None = None

    @property
    def bounded_scale(self) -> Tensor:
        return self._tuning_scale_maximum * torch.sigmoid(self.scale_logit)

    @property
    def inactive_probability_for_loss(self) -> Tensor | None:
        return self._last_inactive_probability_for_loss

    @property
    def active_gate_for_loss(self) -> Tensor | None:
        return self._last_active_gate_for_loss

    @property
    def inactive_logit_for_loss(self) -> Tensor | None:
        return self._last_inactive_logit_for_loss

    @property
    def correction_gate_for_loss(self) -> Tensor | None:
        return self._last_correction_gate_for_loss

    @property
    def temporal_delta_for_loss(self) -> Tensor | None:
        return self._last_temporal_delta_for_loss

    def canonical_m2_state_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                name
                for name in self.state_dict()
                if not any(name.startswith(prefix) for prefix in AMENDMENT_PREFIXES)
            )
        )

    def forward_parent(self, batch: Mapping[str, Tensor]) -> Tensor:
        return super().forward_features(batch)["prediction"]

    def forward_features(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        states = self._pre_output(batch)
        temporal = self.temporal_weather(
            batch["nwp"], states["source_mask"], batch["future_time"]
        )
        interaction = self.station_regime_interaction(
            temporal,
            states["condition"],
            batch["station_id"],
            batch["resource_id"],
        )
        correction_features = torch.cat(
            [
                states["base_state"].float(),
                temporal.float(),
                interaction.float(),
                states["condition"].float(),
            ],
            dim=-1,
        )
        inactive_logit = self.inactive_head(correction_features).squeeze(-1)
        inactive_probability = torch.sigmoid(inactive_logit)
        active_gate = 1.0 - inactive_probability
        temporal_delta, correction_gate = self.correction_head(correction_features)
        raw = states["raw_base"].float() + self.bounded_scale.float() * correction_gate * temporal_delta
        prediction = active_gate * smooth_nonnegative(raw)
        prediction = apply_solar_darkness(
            prediction,
            states["raw_scalar"],
            states["source_mask"],
            batch["resource_id"],
            threshold=float(self.daylight_threshold.item()),
        )
        self._last_inactive_logit_for_loss = inactive_logit
        self._last_inactive_probability_for_loss = inactive_probability
        self._last_active_gate_for_loss = active_gate
        self._last_correction_gate_for_loss = correction_gate
        self._last_temporal_delta_for_loss = temporal_delta
        return {
            **states,
            "temporal_weather_state": temporal,
            "station_regime_interaction": interaction,
            "correction_features": correction_features,
            "inactive_logit": inactive_logit,
            "inactive_probability": inactive_probability,
            "active_gate": active_gate,
            "temporal_delta": temporal_delta,
            "correction_gate": correction_gate,
            "bounded_scale": self.bounded_scale,
            "raw": raw,
            "prediction": prediction,
        }

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return self.forward_features(batch)["prediction"]


MSWPFuseSRATR = MSWPFuse

__all__ = [
    "AMENDMENT_PREFIXES",
    "FORECAST_STEPS",
    "HISTORY_STEPS",
    "HORIZON",
    "InputShapeError",
    "MSWPFuse",
    "MSWPFuseBackbone",
    "MSWPFuseConfig",
    "MSWPFuseSRATR",
]
