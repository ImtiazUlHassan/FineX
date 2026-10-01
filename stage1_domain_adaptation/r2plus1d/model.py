"""
model.py — Stage 1 RGB stream: R(2+1)D-34 pretrained on IG-65M.
"""

import torch
import torch.nn as nn


class R2Plus1DRGB(nn.Module):
    def __init__(self, num_classes: int = 288, dropout: float = 0.5):
        super().__init__()
        backbone_full = torch.hub.load(
            "moabitcoin/ig65m-pytorch", "r2plus1d_34_32_ig65m",
            num_classes=359, pretrained=True,
        )
        self.backbone   = nn.Sequential(*(list(backbone_full.children())[:-2]))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(video))
