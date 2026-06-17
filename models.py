import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Union


def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_normal_(module.weight)
    elif isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:
                nn.init.orthogonal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)


class LayerNorm(nn.Module):
    """增强型层归一化，添加随机噪声"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.noise_scale = 0.05

    def forward(self, x):
        x = self.norm(x)
        if self.training:
            noise = torch.randn_like(x) * self.noise_scale
            return x + noise
        return x


class Transformer(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=256, num_layers=6, dropout=0.2):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Transformer层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层 - 二分类
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.apply(init_weights)

    def forward(self, x):
        # 输入形状: [batch_size, input_size]
        x = self.input_projection(x)  # [batch_size, hidden_size]

        # 添加虚拟序列维度
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 通过Transformer
        x = self.transformer_encoder(x)  # [batch_size, 1, hidden_size]

        # 移除虚拟序列维度
        x = x.squeeze(1)  # [batch_size, hidden_size]

        # 输出预测
        return self.output_layer(x)  # [batch_size, output_size]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class iTransformer(nn.Module):
    def __init__(self, input_size, output_size=1,
                 hidden_size=128, num_layers=4, dropout=0.2):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # 1. 输入投影：将输入特征映射到隐藏空间
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # 3. Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=4 * hidden_size,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        # 输入投影: [batch_size, input_size] -> [batch_size, hidden_size]
        x = self.input_projection(x)  # [batch_size, hidden_size]

        # 添加序列维度: [batch_size, hidden_size] -> [batch_size, 1, hidden_size]
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 位置编码（可选）
        x = self.pos_encoder(x)  # [batch_size, 1, hidden_size]

        # Transformer处理
        encoded = self.transformer_encoder(x)  # [batch_size, 1, hidden_size]

        # 移除序列维度
        encoded = encoded.squeeze(1)  # [batch_size, hidden_size]

        # 输出预测
        return self.output_layer(encoded)


@dataclass
class MambaConfig:
    d_model: int  # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16  # N in paper/comments
    expand_factor: int = 2  # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random"  # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    bias: bool = False
    conv_bias: bool = True

    pscan: bool = True  # use parallel scan mode or sequential mode when training

    def __post_init__(self):
        self.d_inner = self.expand_factor * self.d_model  # E*D = ED in comments
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MambaBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config

        # projects block input from D to 2*ED (two branches)
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner, out_channels=config.d_inner,
            kernel_size=config.d_conv, bias=config.conv_bias,
            groups=config.d_inner,
            padding=config.d_conv - 1
        )

        # projects x to input-dependent Δ, B, C
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects Δ from dt_rank to d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        dt_init_std = config.dt_rank ** -0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # dt bias
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D real initialization
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))

        self.D = nn.Parameter(torch.ones(config.d_inner))

        # projects block output from ED back to D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x):
        # x: (B, D)
        B, D = x.shape[:2]
        L = 1  # 对于表格数据，序列长度为1

        xz = self.in_proj(x)  # (B, 2*ED)
        x, z = xz.chunk(2, dim=-1)  # (B, ED), (B, ED)

        # x分支 - 由于序列长度为1，简化卷积操作
        x = x.unsqueeze(-1)  # (B, ED, 1)
        x = F.silu(x)

        # 简化SSM操作
        # 对于长度为1的序列，SSM退化为简单的线性变换
        A = -torch.exp(self.A_log.float())  # (ED, N)
        B_val = self.x_proj(x.squeeze(-1))[:, :self.config.d_state]  # (B, N)
        C_val = self.x_proj(x.squeeze(-1))[:, self.config.d_state:self.config.d_state * 2]  # (B, N)

        # 计算h和y
        h = torch.zeros(B, self.config.d_inner, self.config.d_state, device=x.device)
        h = B_val.unsqueeze(1) * x  # (B, ED, N)
        y = (h @ C_val.unsqueeze(-1)).squeeze(-1)  # (B, ED)

        # z分支
        z = F.silu(z)

        output = y * z
        output = self.out_proj(output)  # (B, D)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.mixer = MambaBlock(config)
        self.norm = RMSNorm(config.d_model)

    def forward(self, x):
        # x: (B, D)
        output = self.mixer(self.norm(x)) + x
        return output


class mamba(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model)

    def forward(self, x):
        # x: (B, D)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        return x


class Mamba(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=256, num_layers=6, dropout=0.2):
        super().__init__()
        # 配置Mamba模型
        mamba_config = MambaConfig(
            d_model=hidden_size,
            n_layers=num_layers,
            dt_rank='auto',
            d_state=16,
            expand_factor=2,
            d_conv=4,
            bias=False,
            conv_bias=True,
            pscan=False
        )

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            LayerNorm(hidden_size),
            nn.Dropout(dropout)
        )

        # Mamba核心层
        self.mamba = mamba(mamba_config)

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        self.apply(init_weights)

    def forward(self, x):
        # 输入形状: [batch_size, input_size]
        x = self.input_layer(x)  # [batch_size, hidden_size]
        x = self.mamba(x)  # [batch_size, hidden_size]
        return self.output_layer(x)  # [batch_size, output_size]


class MambaPlus(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=256, num_layers=6, dropout=0.2):
        super().__init__()

        # 配置Mamba模型
        mamba_config = MambaConfig(
            d_model=hidden_size,
            n_layers=num_layers,
            dt_rank='auto',
            d_state=16,
            expand_factor=2,
            d_conv=4,
            bias=False,
            conv_bias=True,
            pscan=False
        )

        # 输入层
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Mamba核心层
        self.mamba = mamba(mamba_config)

        # ========== 多尺度特征金字塔 ==========
        self.scale_factors = [1, 2, 4]  # 不同尺度的下采样因子

        # 下采样层，将输入投影到不同尺度
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // factor),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for factor in self.scale_factors
        ])

        # 多尺度特征提取块
        self.feature_blocks = nn.ModuleList([
            self._create_scale_specific_block(hidden_size // factor, dropout, f"scale_{factor}")
            for factor in self.scale_factors
        ])

        # 上采样层，将不同尺度特征统一到相同维度
        self.upsample_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size // factor, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for factor in self.scale_factors
        ])

        # ========== 自适应门控机制 ==========
        self.importance_weights = nn.Parameter(torch.ones(len(self.scale_factors)))
        self.scale_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Sigmoid()
            ) for _ in range(len(self.scale_factors))
        ])

        # 全局门控，平衡基干特征和多尺度特征
        self.global_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid()
        )

        # ========== 多级注意力机制 ==========
        # 用交叉注意力替代自注意力，避免序列长度为1的问题
        self.cross_scale_attention = nn.MultiheadAttention(
            hidden_size, num_heads=4, dropout=dropout, batch_first=True
        )

        # 特征增强前馈网络
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )

        self.enhancer_norm = LayerNorm(hidden_size)

        # 残差缩放因子
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )

        self.apply(init_weights)

    def _create_scale_specific_block(self, feature_dim, dropout, scale_name):
        """创建针对特定尺度的特征提取块"""
        if "scale_1" in scale_name:  # 原始尺度：深层复杂网络
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 2, feature_dim),
                nn.BatchNorm1d(feature_dim)
            )
        elif "scale_2" in scale_name:  # 中等尺度：平衡复杂度
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(feature_dim)
            )
        else:  # 小尺度：轻量级
            return nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        # 输入处理
        x = self.input_layer(x)  # [batch_size, hidden_size]
        base_feature = self.mamba(x)  # [batch_size, hidden_size]

        # ===== 多尺度特征提取 =====
        scale_features = []

        for i, (downsample, feature_block, upsample) in enumerate(
                zip(self.downsample_layers, self.feature_blocks, self.upsample_layers)):
            # 下采样到对应尺度
            downsampled = downsample(x)  # [batch_size, hidden_size//factor]

            # 尺度特定特征提取
            scale_feature = feature_block(downsampled)  # [batch_size, hidden_size//factor]

            # 上采样到统一维度
            upsampled_feature = upsample(scale_feature)  # [batch_size, hidden_size]

            scale_features.append(upsampled_feature)

        # ===== 自适应门控 =====
        # 1. 计算每个尺度的重要性权重
        softmax_weights = F.softmax(self.importance_weights, dim=0)

        # 2. 应用尺度特定门控
        gated_scale_features = []
        for i, (scale_feat, scale_gate) in enumerate(zip(scale_features, self.scale_gates)):
            gate_weights = scale_gate(scale_feat)  # [batch_size, hidden_size]
            gated_feat = scale_feat * gate_weights * softmax_weights[i]
            gated_scale_features.append(gated_feat)

        # 3. 融合多尺度特征
        fused_features = torch.stack(gated_scale_features, dim=1)  # [batch_size, num_scales, hidden_size]
        fused_features = torch.sum(fused_features, dim=1)  # [batch_size, hidden_size]

        # 4. 全局门控平衡基干特征和多尺度特征
        gate_input = torch.cat([base_feature, fused_features], dim=1)  # [batch_size, hidden_size*2]
        global_gate_weights = self.global_gate(gate_input)  # [batch_size, hidden_size]

        # 最终特征融合
        final_feature = (global_gate_weights * base_feature +
                         (1 - global_gate_weights) * fused_features)  # [batch_size, hidden_size]

        # ===== 多级注意力机制 =====
        # 将多尺度特征作为查询，基干特征作为键值进行交叉注意力
        query = torch.stack(scale_features, dim=1)  # [batch_size, num_scales, hidden_size]
        key_value = base_feature.unsqueeze(1)  # [batch_size, 1, hidden_size]

        # 交叉尺度注意力
        attended_features, _ = self.cross_scale_attention(
            query, key_value, key_value
        )  # [batch_size, num_scales, hidden_size]

        # 合并注意力后的特征
        attended_pooled = torch.mean(attended_features, dim=1)  # [batch_size, hidden_size]

        # 特征增强前馈网络
        enhanced_features = self.feature_enhancer(attended_pooled)  # [batch_size, hidden_size]
        enhanced_features = self.enhancer_norm(enhanced_features)

        # 带缩放因子的残差连接
        final_feature = final_feature + self.residual_scale * enhanced_features  # [batch_size, hidden_size]

        # 输出处理
        output = self.output_layer(final_feature)  # [batch_size, output_size]

        return output


def get_model_class(model_name):
    model_map = {
        'transformer': Transformer,
        'itransformer': iTransformer,
        'mamba': Mamba,
        'mambaplus': MambaPlus
    }
    return model_map[model_name]