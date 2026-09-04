"""Two small pose-regression heads on one conv trunk.

The interesting part of this file is not the trunk, it is the head.

A pose regressor must report *where* the cube is. That means the head cannot be
translation invariant. Global average pooling -- the default head in every
classification network -- averages over space and throws the answer away. So:

  head="flat"        flatten the 8x8x128 feature map and let a linear layer
                     figure out the geometry from scratch. Generic, no prior.

  head="softargmax"  spatial softmax over each channel, then take the expected
                     (u, v) of that distribution. The layer *is* a coordinate,
                     so position is handed to the network instead of learned.
                     This is the keypoint front-end used by visuomotor policies
                     (Levine et al., deep spatial autoencoders).

Same data, same budget, ~5x fewer parameters in the softargmax version. If the
prior is worth anything it should show up as lower error, faster convergence,
or both. If it does not, that is also a result.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout, stride=2):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SpatialSoftArgmax(nn.Module):
    """(B,K,H,W) -> (B,2K) expected pixel coords in [-1,1], per channel."""

    def __init__(self, temperature=1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, x):
        b, k, h, w = x.shape
        p = F.softmax(x.reshape(b, k, h * w) / self.log_t.exp(), dim=-1).reshape(b, k, h, w)
        # normalised grid, matches the image convention: u right, v down
        u = torch.linspace(-1, 1, w, device=x.device, dtype=x.dtype).view(1, 1, 1, w)
        v = torch.linspace(-1, 1, h, device=x.device, dtype=x.dtype).view(1, 1, h, 1)
        eu = (p * u).sum((2, 3))
        ev = (p * v).sum((2, 3))
        return torch.cat([eu, ev], dim=1)


class PoseNet(nn.Module):
    """RGB 128x128 -> (x_norm, y_norm, sin4yaw, cos4yaw)"""

    def __init__(self, head="flat", keypoints=16):
        super().__init__()
        self.head_kind = head
        if head == "flat":
            self.trunk = nn.Sequential(
                block(3, 16), block(16, 32), block(32, 64), block(64, 128),
            )                                             # 128x128 -> 8x8
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(8 * 8 * 128, 4),
            )
        elif head == "softargmax":
            self.trunk = nn.Sequential(
                block(3, 16), block(16, 32), block(32, 64),
            )                                             # 128x128 -> 16x16
            self.kp = nn.Conv2d(64, keypoints, 1)
            self.head = nn.Sequential(
                SpatialSoftArgmax(),
                nn.Linear(2 * keypoints, 64), nn.ReLU(inplace=True),
                nn.Linear(64, 4),
            )
        else:
            raise ValueError(head)

    def forward(self, x):
        z = self.trunk(x)
        if self.head_kind == "softargmax":
            z = self.kp(z)
        return self.head(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())
