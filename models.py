import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x):
        return self.norm(x)


@dataclass
class MambaConfig:
    d_model: int
    n_layers: int
    dt_rank: Union[int, str] = "auto"
    d_state: int = 16
    expand_factor: int = 2
    d_conv: int = 4

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.in_proj = nn.Linear(config.d_model, config.d_inner)
        self.out_proj = nn.Linear(config.d_inner, config.d_model)

    def forward(self, x):
        return self.out_proj(F.silu(self.in_proj(x)))


class ResidualBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = MambaBlock(config)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class MambaBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidualBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm_f(x)


class ScaleBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return self.block(x)


class MAGIVul(nn.Module):
    """
    Implementation aligned with Section III (Method) of MAGI‑Vul.
    Components:
    1. Input Encoding Layer
    2. Mamba Backbone
    3. Multi‑Scale Representation Pyramid (1,2,4)
    4. Adaptive Gated Fusion
    5. Multi‑Level Attention Interaction Enhancement
    6. Scaled Residual Prediction Head
    """

    def __init__(self,
                 input_size,
                 output_size=1,
                 hidden_size=128,
                 num_layers=4,
                 dropout=0.3):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        config = MambaConfig(
            d_model=hidden_size,
            n_layers=num_layers
        )
        self.mamba = MambaBackbone(config)

        self.scale_factors = [1, 2, 4]

        self.down_layers = nn.ModuleList()
        self.scale_blocks = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        for s in self.scale_factors:
            scale_dim = hidden_size // s

            self.down_layers.append(
                nn.Linear(hidden_size, scale_dim)
            )

            self.scale_blocks.append(
                ScaleBlock(scale_dim, dropout)
            )

            self.up_layers.append(
                nn.Linear(scale_dim, hidden_size)
            )

        self.scale_importance = nn.Parameter(torch.zeros(len(self.scale_factors)))

        self.scale_gates = nn.ModuleList([
            nn.Linear(hidden_size * 2, hidden_size)
            for _ in self.scale_factors
        ])

        self.global_gate = nn.Linear(hidden_size * 2, hidden_size)

        self.interaction_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            batch_first=True
        )

        self.interaction_ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

        self.interaction_norm = nn.LayerNorm(hidden_size)

        self.gamma = nn.Parameter(torch.tensor(0.1))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):

        h = self.input_layer(x)

        H = self.mamba(h)
        B = H

        scale_features = []

        for down, block, up in zip(
                self.down_layers,
                self.scale_blocks,
                self.up_layers):

            ds = F.relu(down(h))
            us = block(ds)
            Ss = up(us)

            scale_features.append(Ss)

        alpha = torch.softmax(self.scale_importance, dim=0)

        fused_feature = 0

        for idx, Ss in enumerate(scale_features):
            gate = torch.sigmoid(
                self.scale_gates[idx](torch.cat([B, Ss], dim=-1))
            )
            fused_feature = fused_feature + alpha[idx] * (gate * Ss)

        balance_gate = torch.sigmoid(
            self.global_gate(torch.cat([B, fused_feature], dim=-1))
        )

        Q = balance_gate * B + (1 - balance_gate) * fused_feature

        scale_stack = torch.stack(scale_features, dim=1)
        base_token = B.unsqueeze(1)

        attention_output, _ = self.interaction_attention(
            query=scale_stack,
            key=base_token,
            value=base_token
        )

        E = attention_output.mean(dim=1)
        E = self.interaction_norm(self.interaction_ffn(E))

        Z = Q + self.gamma * E

        return self.classifier(Z)


def get_model_class(model_name):
    model_name = model_name.lower()

    if model_name in ["magi-vul", "magivul"]:
        return MAGIVul

    raise ValueError(f"Unsupported model: {model_name}")
