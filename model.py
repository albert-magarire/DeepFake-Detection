"""EfficientNet spatial features fused with FFT features; B4 by default."""

import torch
from torch import nn
from torchvision import models


class FrequencyCNN(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.fc(self.net(x).flatten(1))


class DualBranchModel(nn.Module):
    def __init__(self, pretrained=False, spatial_weight=0.8, frequency_weight=0.2,
                 backbone="b4"):
        super().__init__()
        constructors = {
            "b4": (models.efficientnet_b4, models.EfficientNet_B4_Weights.DEFAULT),
            "b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.DEFAULT),
        }
        if backbone not in constructors:
            raise ValueError("Supported backbones: b4 and b0.")
        constructor, weights = constructors[backbone]
        backbone = constructor(weights=weights if pretrained else None)
        self.spatial = backbone.features
        self.spatial_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.spatial_dim = backbone.classifier[1].in_features
        self.freq_branch = FrequencyCNN(256)
        self.spatial_weight = spatial_weight
        self.freq_weight = frequency_weight
        self.classifier = nn.Sequential(
            nn.Linear(self.spatial_dim + 256, 512),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    @staticmethod
    def fft_transform(x):
        spectrum = torch.fft.fft2(x, dim=(-2, -1))
        spectrum = torch.fft.fftshift(spectrum, dim=(-2, -1))
        return torch.log1p(torch.abs(spectrum))

    def forward(self, x):
        spatial = self.spatial_pool(self.spatial(x)).flatten(1)
        frequency = self.freq_branch(self.fft_transform(x))
        fused = torch.cat([
            self.spatial_weight * spatial,
            self.freq_weight * frequency,
        ], dim=1)
        return self.classifier(fused).squeeze(1)
