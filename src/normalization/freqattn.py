"""
FreqAttn — Frequency Attention Normalizer

Treats 128 mel-frequency bins as tokens, uses self-attention to learn
which frequency bands to correct based on all other bands.
Physically motivated: device mismatch is a per-frequency response shift,
and adjacent/related frequency bands carry evidence about each other.

~15K parameters. Residual output like DevNormNet.
"""

import torch
import torch.nn as nn
import math


class FreqAttentionBlock(nn.Module):
    def __init__(self, n_mels=128, d_model=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x):
        # x: (B, n_mels, d_model)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class FreqAttn(nn.Module):
    """
    Frequency Attention Normalizer (~15K params).

    Input:  (B, 1, n_mels, T)
    Output: (B, 1, n_mels, T)  — residual correction in frequency domain

    Each time frame is processed independently; self-attention runs
    over the 128 frequency bins so each bin attends to all others.
    """

    def __init__(self, n_mels=128, d_model=32, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj  = nn.Linear(1, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, n_mels, d_model) * 0.02)
        self.layers      = nn.ModuleList([
            FreqAttentionBlock(n_mels, d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, 1)
        self.norm_out    = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, 1, F, T)
        B, C, F, T = x.shape
        # Process each time frame: reshape to (B*T, F, 1)
        xt = x.squeeze(1).permute(0, 2, 1).reshape(B * T, F, 1)  # (B*T, F, 1)
        h  = self.input_proj(xt) + self.pos_embed                  # (B*T, F, d_model)
        for layer in self.layers:
            h = layer(h)
        h = self.output_proj(self.norm_out(h))                      # (B*T, F, 1)
        correction = h.reshape(B, T, F, 1).squeeze(-1).permute(0, 2, 1).unsqueeze(1)  # (B,1,F,T)
        return x + correction

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
