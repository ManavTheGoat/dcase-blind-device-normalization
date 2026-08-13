"""
Adversarial Domain Adaptation for Device-Invariant ASC

No paired data required. A gradient reversal layer forces the shared
encoder to produce device-agnostic features.

Architecture:
  - SharedEncoder  (CNN feature extractor, ~8K params)
  - SceneClassifier (linear head, ~10K params)
  - DeviceDiscriminator (small MLP, ~2K params)

Training: scene loss (forward) + device loss through reversed gradient.
The encoder learns features that maximise scene accuracy and simultaneously
confuse the device discriminator — i.e., device-invariant features.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return GradientReversal.apply(x, alpha)


class SharedEncoder(nn.Module):
    """Lightweight CNN encoder shared between scene and device heads."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,  16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.net(x).flatten(1)  # (B, 64)


class SceneHead(nn.Module):
    def __init__(self, in_dim=64, n_classes=10):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.fc(x)


class DeviceHead(nn.Module):
    """Device discriminator — tries to predict which device; encoder tries to fool it."""
    def __init__(self, in_dim=64, n_devices=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(True),
            nn.Linear(32, n_devices),
        )

    def forward(self, x):
        return self.net(x)


class AdversarialASC(nn.Module):
    """
    Full adversarial model (~20K params total).

    At inference: only encoder + scene_head used — no device label needed.
    """
    def __init__(self, n_classes=10, n_devices=9):
        super().__init__()
        self.encoder   = SharedEncoder()
        self.scene_head  = SceneHead(64, n_classes)
        self.device_head = DeviceHead(64, n_devices)

    def forward(self, x, alpha=1.0):
        feat = self.encoder(x)
        scene_logits  = self.scene_head(feat)
        device_logits = self.device_head(grad_reverse(feat, alpha))
        return scene_logits, device_logits

    def predict(self, x):
        """Inference: scene prediction only."""
        return self.scene_head(self.encoder(x))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
