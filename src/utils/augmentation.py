"""
Data augmentation for device-robust acoustic scene classification.

Freq-MixStyle (FMS): mixes per-frequency-band statistics between samples,
forcing the model to learn device-invariant features. The single highest-ROI
augmentation in DCASE 2022-2024 — used by every top-10 team.

Mixup: convex combination of two samples and their labels.

SpecAugment: random frequency and time masking.
"""

import torch
import numpy as np
import random


def freq_mixstyle(x, p=0.5, alpha=0.1):
    """
    Freq-MixStyle: mix per-frequency-band (mean, std) statistics between samples.
    x: (B, 1, F, T)
    """
    if random.random() > p:
        return x
    B, C, F, T = x.shape
    mu  = x.mean(dim=-1, keepdim=True)              # (B, 1, F, 1)
    std = x.var(dim=-1, keepdim=True).add(1e-6).sqrt()
    x_norm = (x - mu) / std

    perm   = torch.randperm(B, device=x.device)
    mu2, std2 = mu[perm], std[perm]

    lam = torch.distributions.Beta(alpha, alpha).sample([B, 1, 1, 1]).to(x.device)
    return x_norm * (lam * std + (1 - lam) * std2) + (lam * mu + (1 - lam) * mu2)


def mixup_batch(x, y, alpha=0.2):
    """
    Mixup augmentation.
    Returns: mixed_x, ya, yb, lam  — loss = lam*ce(logits,ya) + (1-lam)*ce(logits,yb)
    """
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[perm], y, y[perm], lam


def mixup_loss(ce, logits, ya, yb, lam):
    return lam * ce(logits, ya) + (1 - lam) * ce(logits, yb)


def spec_augment(x, freq_mask=20, time_mask=20, num_freq=2, num_time=2):
    """
    SpecAugment: random frequency and time masking.
    x: (B, 1, F, T)
    """
    x = x.clone()
    _, _, F, T = x.shape
    for _ in range(num_freq):
        f  = random.randint(0, freq_mask)
        f0 = random.randint(0, max(F - f, 0))
        x[:, :, f0:f0 + f, :] = 0.0
    for _ in range(num_time):
        t  = random.randint(0, time_mask)
        t0 = random.randint(0, max(T - t, 0))
        x[:, :, :, t0:t0 + t] = 0.0
    return x
