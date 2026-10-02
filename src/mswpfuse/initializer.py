"""Standalone history initializer model matching the canonical MinimalV3MSWPFormer (v3-median)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from .primitives import (
    CrossAttentionBlock,
    PositionalEncoding,
    TransformerBlock,
    source_disagreement_profiles,
)

NWP_SOURCE_COUNT = 3
NWP_CHANNELS_PER_SOURCE = 9
HISTORY_CHANNELS = 2
NATIVE_STEPS = 96
HOURLY_STEPS = 24


class HistoryInitializer(nn.Module):
    """The complete 80-tensor minimal-v3 history model used for pretraining."""

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        n_stations: int = 10,
    ) -> None:
        super().__init__()
        del n_stations
        cfg = dict(config or {})
        d_model = int(cfg.get("d_model", 128))
        n_heads = int(cfg.get("n_heads", 4))
        d_ff = int(cfg.get("d_ff", 512))
        dropout = float(cfg.get("dropout", 0.10))
        self.horizon = int(cfg.get("horizon", NATIVE_STEPS))
        self.patch_len = int(cfg.get("patch_len", 8))
        self.patch_stride = int(cfg.get("patch_stride", 8))
        self.fusion_mode = str(cfg.get("fusion_mode", "cross_attention"))
        self.resource_gate_enabled = bool(cfg.get("resource_gate_enabled", True))
        encoder_layers = int(cfg.get("encoder_layers", 2))

        self.history_patch_projection = nn.Linear(self.patch_len, d_model)
        self.history_time_projection = nn.Linear(5, d_model, bias=False)
        self.history_channel_embedding = nn.Embedding(HISTORY_CHANNELS, d_model)
        self.history_position = PositionalEncoding(d_model)
        self.history_encoder = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(encoder_layers)]
        )

        self.nwp_native_projection = nn.Linear(3 * NATIVE_STEPS, d_model)
        self.nwp_hourly_projection = nn.Linear(3 * HOURLY_STEPS, d_model)
        self.nwp_provider_embedding = nn.Embedding(NWP_SOURCE_COUNT, d_model)
        self.nwp_channel_embedding = nn.Embedding(NWP_CHANNELS_PER_SOURCE, d_model)
        self.nwp_scale_embedding = nn.Embedding(2, d_model)
        self.nwp_valid_start_projection = nn.Linear(5, d_model, bias=False)
        self.nwp_availability_projection = nn.Linear(1, d_model, bias=False)
        self.nwp_encoder = TransformerBlock(d_model, n_heads, d_ff, dropout)
        self.nwp_norm = nn.LayerNorm(d_model)
        self.register_buffer(
            "native_lead_profile",
            torch.linspace(0.0, 1.0, NATIVE_STEPS),
        )
        self.register_buffer(
            "hourly_lead_profile",
            torch.linspace(0.0, 1.0, HOURLY_STEPS),
        )

        self.query_embedding = nn.Parameter(
            torch.randn(1, self.horizon, d_model) * 0.02
        )
        self.future_time_projection = nn.Linear(5, d_model, bias=False)
        self.fusion_attention = CrossAttentionBlock(
            d_model,
            n_heads,
            d_ff,
            dropout,
        )
        if self.fusion_mode == "naive_concatenation":
            self.naive_nwp_projection = nn.Linear(2 * d_model, d_model)
            self.naive_fusion_projection = nn.Linear(2 * d_model, d_model)
        else:
            self.naive_nwp_projection = None
            self.naive_fusion_projection = None

        if self.resource_gate_enabled:
            self.resource_embedding = nn.Embedding(2, d_model)
            self.resource_adapter = nn.Linear(d_model, d_model, bias=False)
            self.resource_gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.resource_adapter.weight)
        else:
            self.resource_embedding = None
            self.resource_adapter = None
            self.resource_gate = None
        self.output_norm = nn.LayerNorm(d_model)
        self.shared_head = nn.Linear(d_model, 1)

    def _history_tokens(self, batch: Mapping[str, Tensor]) -> Tensor:
        history = batch["hist_power"].transpose(1, 2)
        if history.size(1) != HISTORY_CHANNELS:
            raise ValueError(f"Unexpected history channels: {history.size(1)}")
        patches = history.unfold(2, self.patch_len, self.patch_stride)
        time_patches = (
            batch["hist_time"]
            .transpose(1, 2)
            .unfold(2, self.patch_len, self.patch_stride)
            .mean(dim=-1)
            .transpose(1, 2)
        )
        tokens = self.history_patch_projection(patches)
        tokens = tokens + self.history_time_projection(time_patches).unsqueeze(1)
        channels = self.history_channel_embedding(
            torch.arange(HISTORY_CHANNELS, device=history.device)
        )
        tokens = tokens + channels.view(1, HISTORY_CHANNELS, 1, -1)
        tokens = tokens.flatten(1, 2)
        tokens = self.history_position(tokens)
        for layer in self.history_encoder:
            tokens = layer(tokens)
        return tokens

    def _nwp_tokens(self, batch: Mapping[str, Tensor]) -> Tensor:
        values = batch["nwp"].reshape(
            -1,
            self.horizon,
            NWP_SOURCE_COUNT,
            NWP_CHANNELS_PER_SOURCE,
        )
        availability = batch["nwp_availability"].to(values.dtype)
        profiles = values.permute(0, 2, 3, 1)
        disagreement = source_disagreement_profiles(values, availability).permute(
            0, 2, 3, 1
        )
        native_lead = self.native_lead_profile.to(values.dtype).view(1, 1, 1, -1)
        native = self.nwp_native_projection(
            torch.cat(
                [
                    profiles,
                    disagreement,
                    native_lead.expand_as(profiles),
                ],
                dim=-1,
            )
        )
        hourly_profiles = profiles.reshape(
            profiles.size(0),
            NWP_SOURCE_COUNT,
            NWP_CHANNELS_PER_SOURCE,
            HOURLY_STEPS,
            4,
        ).mean(dim=-1)
        hourly_disagreement = disagreement.reshape(
            disagreement.size(0),
            NWP_SOURCE_COUNT,
            NWP_CHANNELS_PER_SOURCE,
            HOURLY_STEPS,
            4,
        ).mean(dim=-1)
        hourly_lead = self.hourly_lead_profile.to(values.dtype).view(1, 1, 1, -1)
        hourly = self.nwp_hourly_projection(
            torch.cat(
                [
                    hourly_profiles,
                    hourly_disagreement,
                    hourly_lead.expand_as(hourly_profiles),
                ],
                dim=-1,
            )
        )

        providers = self.nwp_provider_embedding(
            torch.arange(NWP_SOURCE_COUNT, device=values.device)
        ).view(1, NWP_SOURCE_COUNT, 1, -1)
        channels = self.nwp_channel_embedding(
            torch.arange(NWP_CHANNELS_PER_SOURCE, device=values.device)
        ).view(1, 1, NWP_CHANNELS_PER_SOURCE, -1)
        valid_start = self.nwp_valid_start_projection(batch["future_time"][:, 0])
        available = self.nwp_availability_projection(
            availability[:, :, None, None].expand(
                -1,
                -1,
                NWP_CHANNELS_PER_SOURCE,
                -1,
            )
        )
        shared = providers + channels + valid_start[:, None, None, :] + available
        native = native + shared + self.nwp_scale_embedding.weight[0]
        hourly = hourly + shared + self.nwp_scale_embedding.weight[1]
        tokens = torch.cat(
            [native.flatten(1, 2), hourly.flatten(1, 2)],
            dim=1,
        )
        return self.nwp_norm(self.nwp_encoder(tokens))

    def _resource_control(
        self,
        query: Tensor,
        resource_id: Tensor,
    ) -> Tensor:
        if not self.resource_gate_enabled:
            return query
        if (
            self.resource_embedding is None
            or self.resource_adapter is None
            or self.resource_gate is None
        ):
            raise RuntimeError("Resource-gate modules are unavailable")
        resource = self.resource_embedding(resource_id.long())
        repeated = resource.unsqueeze(1).expand(-1, self.horizon, -1)
        gate = self.resource_gate(torch.cat([query, repeated], dim=-1))
        return query + gate * self.resource_adapter(repeated)

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        history = self._history_tokens(batch)
        nwp = self._nwp_tokens(batch)
        query = self.query_embedding.expand(history.size(0), -1, -1)
        query = query + self.future_time_projection(batch["future_time"])
        if self.fusion_mode == "cross_attention":
            query = self.fusion_attention(query, torch.cat([history, nwp], dim=1))
        else:
            if (
                self.naive_nwp_projection is None
                or self.naive_fusion_projection is None
            ):
                raise RuntimeError("Naive fusion modules are unavailable")
            query = self.fusion_attention(query, history)
            native, hourly = nwp.split(
                NWP_SOURCE_COUNT * NWP_CHANNELS_PER_SOURCE,
                dim=1,
            )
            summary = self.naive_nwp_projection(
                torch.cat([native.mean(dim=1), hourly.mean(dim=1)], dim=-1)
            )
            query = self.naive_fusion_projection(
                torch.cat(
                    [query, summary.unsqueeze(1).expand(-1, self.horizon, -1)],
                    dim=-1,
                )
            )
        query = self._resource_control(query, batch["resource_id"])
        features = self.output_norm(query)
        return self.shared_head(features).squeeze(-1)


__all__ = ["HistoryInitializer"]
