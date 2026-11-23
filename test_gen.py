"""gen_dataset.py writes labels that describe its own images, and val is not train.

A dozen easy-regime images are rendered into a temporary ROOT (data/ is never
touched). Each label must agree with its picture: the red silhouette's centre sits
within 3 px of world_to_pixel(x, y) (the side faces pull it outward by up to ~2.3 px
here), and the box around it is s * (|cos yaw| + |sin yaw|) wide, which a cube left
unrotated or a yaw label from another sample would break. The sin/cos columns must
be sin 4yaw, cos 4yaw. main() must give val a different seed from train.

Run:  MUJOCO_GL=egl python test_gen.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np

import common as C
import gen_dataset as G

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:48s} {detail}")
    if not ok:
        FAILED.append(name)


HERE = os.path.dirname(os.path.abspath(__file__))
with tempfile.TemporaryDirectory() as d:
    shutil.copy(os.path.join(HERE, "scene.xml"), d)
    G.ROOT = d
    G.generate("easy", 12, 128, 0, "train")
    imgs = np.load(os.path.join(d, "data", "easy", "train_images.npy"))
    lab = np.load(os.path.join(d, "data", "easy", "train_labels.npy"))
G.ROOT = HERE

print("\nlabel columns")
check("sin/cos columns are sin 4yaw, cos 4yaw",
      np.allclose(lab[:, 2], np.sin(4 * lab[:, 4]), atol=1e-6) and np.allclose(lab[:, 3], np.cos(4 * lab[:, 4]), atol=1e-6))
check("yaw sampled in [0, 90 deg)", bool(((lab[:, 4] >= 0) & (lab[:, 4] < np.pi / 2)).all()))

print("\nlabels agree with their images")
side = 2 * C.CUBE_HALF * C.px_per_m(128) * (C.CAM_HEIGHT - C.CUBE_HALF) / (C.CAM_HEIGHT - 2 * C.CUBE_HALF)
cerr, werr = [], []
for im, (x, y, _, _, yaw) in zip(imgs, lab):
    vs, us = np.nonzero((im[..., 0].astype(int) - im[..., 1]) > 60)
    u, v = C.world_to_pixel(x, y, 128)
    cerr.append(np.hypot(us.mean() + 0.5 - u, vs.mean() + 0.5 - v))
    werr.append(abs((us.max() + 1 - us.min()) - side * (abs(np.cos(yaw)) + abs(np.sin(yaw)))))
check("silhouette centre within 3 px of the xy label", max(cerr) < 3.0, f"max {max(cerr):.2f} px")
check("silhouette width matches the yaw label (1.5 px)", max(werr) < 1.5, f"max {max(werr):.2f} px")

print("\nval split is not the train split")
seeds = {}
G.generate, real = (lambda regime, n, size, seed, split: seeds.__setitem__(split, seed)), G.generate
try:
    G.main(["--regime", "easy", "--n", "6"])
finally:
    G.generate = real
check("train and val drawn from different seeds", set(seeds) == {"train", "val"} and seeds["train"] != seeds["val"],
      str(seeds))

print(f"\n{'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
sys.exit(1 if FAILED else 0)
