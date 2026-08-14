import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution — parameter-efficient building block."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.pw(self.dw(x))))


class CPMobile(nn.Module):
    """
    Lightweight CNN classifier (<128K parameters) for acoustic scene classification.
    Optionally accepts device-level signal features concatenated before final layer.

    Based on CP-Mobile architecture used in DCASE 2025 Task 1 baseline.
    Novel addition: signal_feat_dim parameter to incorporate blind device features.
    """

    def __init__(self, n_classes=10, signal_feat_dim=0):
        super().__init__()
        self.signal_feat_dim = signal_feat_dim

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16, 32, stride=2),
            DepthwiseSeparableConv(32, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        classifier_in = 64 + signal_feat_dim
        self.classifier = nn.Linear(classifier_in, n_classes)

    def forward(self, mel, signal_feats=None):
        x = self.features(mel)
        x = self.pool(x).squeeze(-1).squeeze(-1)  # (batch, 64)

        if self.signal_feat_dim > 0 and signal_feats is not None:
            x = torch.cat([x, signal_feats], dim=1)  # (batch, 64 + signal_feat_dim)

        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CPMobileWide(nn.Module):
    """
    Wider CPMobile using the full 128K parameter budget.

    With bc=40: ~93K params. Combined with frozen FreqAttn (21K) = 115K total.
    Adds one extra DS block and a bottleneck FC head with dropout.
    """

    def __init__(self, n_classes=10, bc=40):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, bc, 3, padding=1, bias=False),
            nn.BatchNorm2d(bc),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(bc,   bc*2, stride=2),
            DepthwiseSeparableConv(bc*2, bc*2, stride=2),
            DepthwiseSeparableConv(bc*2, bc*4, stride=2),
            DepthwiseSeparableConv(bc*4, bc*4, stride=2),
            DepthwiseSeparableConv(bc*4, bc*4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(bc*4, bc*2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(bc*2, n_classes),
        )

    def forward(self, x):
        x = self.pool(self.features(x)).squeeze(-1).squeeze(-1)
        return self.classifier(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
