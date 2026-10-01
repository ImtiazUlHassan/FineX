"""
model.py — SlowOnly-R50 on GPU-rendered Gaussian keypoint heatmap volumes.

Architecture exactly matches pyskl joint.py / limb.py configs:
  in_channels   : 17  (joint) | 34  (joint+limb)
  base_channels : 32
  num_stages    : 3
  stage_blocks  : (4, 6, 3)
  inflate       : (0, 1, 1)   — stage 0 is 2-D, stages 1-2 are 3-D (3x1x1 style)
  spatial_strides  : (2, 2, 2)
  temporal_strides : (1, 1, 2)
  conv1_stride  : (1, 1)
  pool1_stride  : (1, 1)

Input/output flow (T=48, H=56, W=56):
  (B, C, 48, 56, 56)           C = 17 or 34
  → stem (1×7×7 conv, no stride, maxpool no stride)
  → (B, 32, 48, 56, 56)
  → layer1  (spatial×2, temporal×1, inflate=False) → (B, 128, 48, 28, 28)
  → layer2  (spatial×2, temporal×1, inflate=True)  → (B, 256, 48, 14, 14)
  → layer3  (spatial×2, temporal×2, inflate=True)  → (B, 512, 24,  7,  7)
  → AdaptiveAvgPool3d(1) → Dropout(0.5) → Linear(512, num_classes)
"""

import torch
import torch.nn as nn


# ════════════════════════════════════════════════════════════════════════════
# Skeleton / limb definitions  (same as pyskl limb.py)
# ════════════════════════════════════════════════════════════════════════════

SKELETONS = [
    (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
    (11, 13), (13, 15), (6, 12), (12, 14), (14, 16), (0, 1), (0, 2),
    (1, 3), (2, 4), (11, 12),
]


# ════════════════════════════════════════════════════════════════════════════
# Bottleneck block  (inflate_style='3x1x1', matching pyskl default)
# ════════════════════════════════════════════════════════════════════════════

class Bottleneck3d(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=(1, 1),
                 inflate=True, downsample=None):
        super().__init__()
        t_stride, s_stride = stride

        self.conv1 = nn.Conv3d(inplanes, planes,
                               kernel_size=(3, 1, 1) if inflate else 1,
                               padding=(1, 0, 0) if inflate else 0,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes,
                               kernel_size=(1, 3, 3),
                               stride=(t_stride, s_stride, s_stride),
                               padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm3d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ════════════════════════════════════════════════════════════════════════════
# SlowOnly-R50 backbone
# ════════════════════════════════════════════════════════════════════════════

def _make_layer(inplanes, planes, blocks, temporal_stride, spatial_stride, inflate):
    downsample = nn.Sequential(
        nn.Conv3d(inplanes, planes * 4,
                  kernel_size=1,
                  stride=(temporal_stride, spatial_stride, spatial_stride),
                  bias=False),
        nn.BatchNorm3d(planes * 4),
    )
    layers = [Bottleneck3d(inplanes, planes,
                           stride=(temporal_stride, spatial_stride),
                           inflate=inflate,
                           downsample=downsample)]
    for _ in range(1, blocks):
        layers.append(Bottleneck3d(planes * 4, planes, inflate=inflate))
    return nn.Sequential(*layers)


class SlowOnlyR50(nn.Module):
    """ResNet3dSlowOnly-R50 backbone. Outputs (B, 512, T', H', W')."""
    def __init__(self, in_channels=17, base_channels=32):
        super().__init__()
        C = base_channels

        self.conv1   = nn.Conv3d(in_channels, C,
                                 kernel_size=(1, 7, 7),
                                 stride=(1, 1, 1),
                                 padding=(0, 3, 3), bias=False)
        self.bn1     = nn.BatchNorm3d(C)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                   stride=(1, 1, 1),
                                   padding=(0, 1, 1))

        self.layer1 = _make_layer(C,     C,   4, temporal_stride=1, spatial_stride=2, inflate=False)
        self.layer2 = _make_layer(C*4,   C*2, 6, temporal_stride=1, spatial_stride=2, inflate=True)
        self.layer3 = _make_layer(C*4*2, C*4, 3, temporal_stride=2, spatial_stride=2, inflate=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for m in self.modules():
            if isinstance(m, Bottleneck3d):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# ════════════════════════════════════════════════════════════════════════════
# Full model: GPU heatmap rendering + SlowOnly + classifier
# ════════════════════════════════════════════════════════════════════════════

class SlowOnlyHeatmap(nn.Module):
    """
    SlowOnly-R50 on GPU-rendered Gaussian keypoint heatmap volumes.

    forward(keypoints) where keypoints : (B, T, J, 2) in [0, 1]
    Returns logits : (B, num_classes)

    with_limb=True  → 17 joint + 17 limb channels (34 total).
    with_limb=False → joint only (17 channels).

    _render_limbs is fully vectorised — 3 GPU ops, no Python loop.
    """
    def __init__(self, num_classes=288, J=17, H=56, W=56,
                 sigma=2.0, with_limb=True):
        super().__init__()
        self.J         = J
        self.H         = H
        self.W         = W
        self.sigma     = sigma
        self.with_limb = with_limb

        in_channels = J * 2 if with_limb else J

        self.backbone   = SlowOnlyR50(in_channels=in_channels)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.register_buffer("_j1", torch.tensor([s[0] for s in SKELETONS], dtype=torch.long))
        self.register_buffer("_j2", torch.tensor([s[1] for s in SKELETONS], dtype=torch.long))

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mode  = "joint+limb" if with_limb else "joint only"
        print(f"[SlowOnlyHeatmap]  mode={mode}  in_ch={in_channels}  "
              f"H={H}  W={W}  σ={sigma}  Trainable: {total/1e6:.1f}M")

    def _render_joints(self, keypoints: torch.Tensor,
                       H: int = None, W: int = None) -> torch.Tensor:
        """keypoints: (B, T, J, 2) in [0,1] → (B, J, T, H, W)"""
        B, T, J = keypoints.shape[:3]
        H = H if H is not None else self.H
        W = W if W is not None else self.W
        device = keypoints.device

        gy, gx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )

        kx = (keypoints[..., 0] * (W - 1)).permute(0, 2, 1)
        ky = (keypoints[..., 1] * (H - 1)).permute(0, 2, 1)

        dx = gx[None, None, None] - kx[..., None, None]
        dy = gy[None, None, None] - ky[..., None, None]

        return torch.exp(-(dx ** 2 + dy ** 2) / (2 * self.sigma ** 2))

    def _render_limbs(self, keypoints: torch.Tensor,
                      H: int = None, W: int = None) -> torch.Tensor:
        """Vectorised limb rendering — 3 GPU ops. keypoints: (B,T,J,2) → (B,L,T,H,W)"""
        all_hm = self._render_joints(keypoints, H, W)
        return torch.maximum(all_hm[:, self._j1], all_hm[:, self._j2])

    def forward(self, keypoints: torch.Tensor,
                H: int = None, W: int = None) -> torch.Tensor:
        """
        keypoints : (B, T, J, 2) in [0, 1]
        H, W      : heatmap size override — pass H=W=64 at val/test to match pyskl.
        """
        joint_hm = self._render_joints(keypoints, H, W)

        if self.with_limb:
            limb_hm = self._render_limbs(keypoints, H, W)
            x = torch.cat([joint_hm, limb_hm], dim=1)
        else:
            x = joint_hm

        x = self.backbone(x)
        return self.classifier(x)
