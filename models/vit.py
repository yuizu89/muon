from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def _init_transformer_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        attention_dropout: float,
        projection_dropout: float,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(dim, dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        attention = self.attention_dropout(attention)

        x = attention @ value
        x = x.transpose(1, 2).reshape(batch_size, num_tokens, dim)
        x = self.projection(x)
        x = self.projection_dropout(x)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attention = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            projection_dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEmbedding(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.projection = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_normal_(self.projection.weight, mode="fan_out")
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embedding[:, : x.size(1)]
        x = self.dropout(x)
        return x


class ViTBody(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    attention_dropout=attention_dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed = ViTEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            dim=dim,
            dropout=embedding_dropout,
        )
        self.body = ViTBody(
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )
        self.head = nn.Linear(dim, num_classes)

        self.apply(_init_transformer_weights)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        x = self.body(x)
        return x[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        return self.head(features)
