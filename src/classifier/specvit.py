"""
SpecViT — Spectrogram Patch Transformer

Splits the mel spectrogram into small patches (like ViT) and processes
them with a lightweight transformer. Adapted for the DCASE <128K param limit.

Patch size: 16×16 on a 128×98 spectrogram → 8×6 = 48 patches.
d_model=64, n_heads=4, n_layers=3  →  ~50K params.
"""

import torch
import torch.nn as nn
import math


class PatchEmbed(nn.Module):
    def __init__(self, patch_h=16, patch_w=16, d_model=64):
        super().__init__()
        self.proj = nn.Conv2d(1, d_model, kernel_size=(patch_h, patch_w),
                              stride=(patch_h, patch_w))

    def forward(self, x):
        # x: (B, 1, F, T) → (B, d_model, nH, nW) → (B, n_patches, d_model)
        x = self.proj(x)
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # (B, H*W, d_model)


class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, mlp_ratio=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        dim_ff = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class SpecViT(nn.Module):
    """
    Spectrogram Patch Transformer for scene classification.

    Input:  (B, 1, 128, T)
    Output: (B, n_classes)
    ~50K parameters — stays well within 128K DCASE limit.
    """

    def __init__(self, n_classes=10, patch_h=16, patch_w=16,
                 d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(patch_h, patch_w, d_model)
        # learnable CLS token + positional embedding (max 64 patches)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed  = nn.Parameter(torch.randn(1, 65, d_model) * 0.02)
        self.dropout    = nn.Dropout(dropout)
        self.layers     = nn.ModuleList([
            TransformerBlock(d_model, n_heads, mlp_ratio=2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                                    # (B, n_patches, d)
        cls = self.cls_token.expand(B, -1, -1)                    # (B, 1, d)
        x   = torch.cat([cls, x], dim=1)                          # (B, 1+n_patches, d)
        n   = x.size(1)
        x   = x + self.pos_embed[:, :n, :]
        x   = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
        cls_out = self.norm(x[:, 0])                               # CLS token
        return self.head(cls_out)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
