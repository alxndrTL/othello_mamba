import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class TransformerConfig:
    d_model: int # D or d_model in comments
    n_layers: int
    n_heads: int
    max_len: int # maximum sequence length (for positional embedding)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be a multiple of n_heads"

        self.d_head = self.d_model // self.n_heads


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()

        self.config = config

        self.PE = nn.Parameter(torch.randn(config.max_len, config.d_model)/10)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.n_layers)])

    def forward(self, X):
        # X : (B, L, D)

        # Y : (B, L, D)

        _, T, _ = X.size()

        X = X + self.PE[:T]

        for layer in self.layers:
            X = layer(X) # (B, L, d_model)

        return X
    
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.sa = SelfAttentionMultiHead(config)
        self.l1 = nn.LayerNorm(config.d_model)
        self.fc1 = nn.Linear(config.d_model, 4*config.d_model)
        self.act = F.selu # F.gelu # selu used in GPT-2/3, gelu is new
        self.fc2 = nn.Linear(4 * config.d_model, config.d_model)
        self.l2 = nn.LayerNorm(config.d_model)

    def forward(self, X):
        # X : (B, L, D)

        # Y : (B, L, D)

        X = self.l1(X + self.sa(X)) #sublayer 1 = SA
        X = self.l2(X + self.fc2(self.act(self.fc1(X)))) #sublayer 2 = FC

        return X

class SelfAttentionMultiHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False) # d_query = d_head as in the Transformer paper
        self.key_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, X):
        # X : (B, T, d_model)

        B, L, _ = X.size()

        Q = self.query_proj(X).view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_query)
        K = self.key_proj(X).view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_key)
        V = self.value_proj(X).view(B, L, self.config.n_heads, self.config.d_head).transpose(1, 2) # (B, n_heads, L, d_head=d_value)

        QK_T = Q @ torch.transpose(K, 2, 3) # (B, n_heads, L, L)

        mask = torch.tril(torch.ones((L, L), dtype=torch.int32)).bool()
        QK_T[:, :, ~mask] = -float("inf")

        attention_scores = torch.softmax(QK_T / math.sqrt(self.config.d_head), dim=3) # (B, n_heafs, L, L)
        attention = attention_scores @ V # (B, n_h, L, d_value=d_head)

        attention = attention.transpose(1, 2) # (B, L, n_heafs, d_head)
        attention = attention.contiguous().view(B, L, self.config.d_model) # n_heads * d_head = d_model

        return attention
