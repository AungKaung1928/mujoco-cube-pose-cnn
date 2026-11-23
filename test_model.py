"""Hand-computable checks on model.py: parameter counts and the soft-argmax convention.

The README compares a 130k-parameter flatten head with a 27k-parameter soft-argmax
head; the counts below are those two numbers to the unit. The soft-argmax must
return the pixel position of a sharp peak in [-1, 1] with u to the right and v
down, the same convention common.py uses for labels.

Run:  python test_model.py
"""
import sys

import numpy as np
import torch

from model import PoseNet, SpatialSoftArgmax

FAILED = []


def check(name, got, want, tol=1e-9):
    got, want = np.asarray(got, float), np.asarray(want, float)
    ok = np.all(np.abs(got - want) <= tol)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:46s} got {np.round(got, 6)}  want {np.round(want, 6)}")
    if not ok:
        FAILED.append(name)


torch.manual_seed(0)
print("\nparameter counts (README: 130k flatten, 27k soft-argmax)")
check("flatten head", PoseNet("flat").n_params(), 130452)
check("soft-argmax head", PoseNet("softargmax").n_params(), 27109)

print("\noutput shape: 4 = x, y, sin 4yaw, cos 4yaw")
for head in ("flat", "softargmax"):
    with torch.no_grad():
        check(f"{head} on 2 images", tuple(PoseNet(head)(torch.zeros(2, 3, 128, 128)).shape), (2, 4))

print("\nsoft-argmax: a sharp peak returns its own pixel position")
h, w = 16, 16
x = torch.zeros(1, 2, h, w)
x[0, 0, 0, w - 1] = 100.0      # top-right corner -> u = +1, v = -1
x[0, 1, h - 1, 0] = 100.0      # bottom-left corner -> u = -1, v = +1
out = SpatialSoftArgmax()(x)[0].detach().numpy()   # (u0, u1, v0, v1)
check("channel 0 at top-right (u, v)", [out[0], out[2]], [1.0, -1.0], 1e-6)
check("channel 1 at bottom-left (u, v)", [out[1], out[3]], [-1.0, 1.0], 1e-6)
check("flat map -> centre", SpatialSoftArgmax()(torch.zeros(1, 1, h, w))[0].detach().numpy(), [0.0, 0.0], 1e-6)
# pixel (row 4, col 12) on a 16-grid: u = -1 + 2*12/15, v = -1 + 2*4/15
x = torch.zeros(1, 1, h, w)
x[0, 0, 4, 12] = 100.0
check("interior peak (row 4, col 12)", SpatialSoftArgmax()(x)[0].detach().numpy(), [-1 + 24 / 15, -1 + 8 / 15], 1e-6)

print("\ntraining targets and the ONNX decode are inverses")
import common as C
from export_onnx import decode
from train import targets
yaw = np.radians([5.0, 40.0, 89.0])
labels = np.c_[[0.07, -0.03, 0.0], [-0.05, 0.02, 0.074], np.sin(4 * yaw), np.cos(4 * yaw), yaw].astype(np.float32)
t = targets(labels, np.arange(3)).numpy()
check("xy target is metres / XY_RANGE", t[:, :2], labels[:, :2] / C.XY_RANGE, 1e-6)
xy, y2 = decode(t.astype(np.float64))
check("decode(targets) xy, metres", xy, labels[:, :2], 1e-6)
check("decode(targets) yaw, radians", y2, yaw, 1e-5)

print(f"\n{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
