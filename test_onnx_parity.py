"""The tracked ONNX graph must still be the tracked checkpoint (no dataset needed).

The README's ONNX rows (0.59 mm, 0.23 ms, max abs diff 6.6e-7) describe
runs/hard_softargmax/model.onnx; its accuracy rows describe final.pt in the same
folder. If one is retrained or re-exported without the other, both rows still
print but describe different networks. This feeds the same seeded batch through
both, with train.py's own preprocessing, and requires export_onnx.py's tolerance.

Run:  python test_onnx_parity.py
"""
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

from model import PoseNet
from train import to_batch

RUN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "hard_softargmax")
TOL = 1e-4                                       # export_onnx.py's OK threshold

torch.set_num_threads(1)
net = PoseNet("softargmax")
net.load_state_dict(torch.load(os.path.join(RUN, "final.pt"), map_location="cpu"))
net.eval()
imgs = np.random.default_rng(0).integers(0, 256, (16, 128, 128, 3), dtype=np.uint8)
x, _ = to_batch(imgs, np.arange(len(imgs)))
with torch.no_grad():
    ref = net(x).numpy()
sess = ort.InferenceSession(os.path.join(RUN, "model.onnx"), providers=["CPUExecutionProvider"])
got = sess.run(["pose"], {"image": x.numpy()})[0]
d = float(np.abs(ref - got).max())
ok = got.shape == ref.shape and d < TOL
print(f"  {'ok  ' if ok else 'FAIL'}  tracked model.onnx vs final.pt on 16 seeded images: max |diff| {d:.2e} (< {TOL})")
sys.exit(0 if ok else 1)
