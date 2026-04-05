from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .encoding import BOARD_SIZE, NUM_CHANNELS


@dataclass(frozen=True)
class ValueTransformerConfig:
    d_model: int = 256
    depth: int = 8
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * mlp_ratio, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x


class BoardValueTransformer(nn.Module):
    def __init__(self, config: ValueTransformerConfig | None = None) -> None:
        super().__init__()
        self.config = config or ValueTransformerConfig()
        num_tokens = BOARD_SIZE * BOARD_SIZE

        self.input_proj = nn.Linear(NUM_CHANNELS, self.config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_tokens + 1, self.config.d_model))
        self.dropout = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            TransformerBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                dropout=self.config.dropout,
            )
            for _ in range(self.config.depth)
        )
        self.norm = nn.LayerNorm(self.config.d_model)
        self.value_head = nn.Linear(self.config.d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input of shape (batch, {NUM_CHANNELS}, 8, 8), got {tuple(x.shape)}.")

        batch_size = x.shape[0]
        tokens = x.permute(0, 2, 3, 1).reshape(batch_size, BOARD_SIZE * BOARD_SIZE, NUM_CHANNELS)
        tokens = self.input_proj(tokens)
        cls = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = self.dropout(tokens + self.pos_embedding)

        for block in self.blocks:
            tokens = block(tokens)

        cls_output = self.norm(tokens)[:, 0]
        return self.value_head(cls_output).squeeze(-1)
