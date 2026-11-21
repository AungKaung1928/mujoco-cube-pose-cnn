"""Where does INT8 hurt the soft-argmax? Simulate uint8 only inside the head, in PyTorch.

    python int8_mechanism.py      # -> runs/hard_softargmax/int8_mechanism.json

Fake-quantises (round to 256 levels, MinMax range from the same 200 TRAIN images
quantize_int8.py calibrates on) the softmax logits, the probability map, or both,
with everything else in fp32, then scores val with the training metric. Compare
with quantize_int8.py's whole-graph INT8 row to see how much of the damage the
head explains on its own. One thread.
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import common as C
import model as M
from train import evaluate, to_batch

ROOT = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(ROOT, "runs", "hard_softargmax")


def fake_q(t, lo, hi):
    s = (hi - lo) / 255.0
    return torch.clamp(torch.round((t - lo) / s), 0, 255) * s + lo


def main():
    torch.set_num_threads(1)
    net = M.PoseNet("softargmax")
    net.load_state_dict(torch.load(os.path.join(RUN, "final.pt"), map_location="cpu"))
    net.eval()
    st = {"mode": "calib", "xmin": np.inf, "xmax": -np.inf, "pmax": 0.0}

    def forward(self, x):
        b, k, h, w = x.shape
        if st["mode"] == "calib":
            st["xmin"], st["xmax"] = min(st["xmin"], float(x.min())), max(st["xmax"], float(x.max()))
        elif st["mode"] in ("logits", "both"):
            x = fake_q(x, min(st["xmin"], 0.0), st["xmax"])
        p = F.softmax(x.reshape(b, k, h * w) / self.log_t.exp(), dim=-1).reshape(b, k, h, w)
        if st["mode"] == "calib":
            st["pmax"] = max(st["pmax"], float(p.max()))
        elif st["mode"] in ("probability_map", "both"):
            p = fake_q(p, 0.0, st["pmax"])
        u = torch.linspace(-1, 1, w).view(1, 1, 1, w)
        v = torch.linspace(-1, 1, h).view(1, 1, h, 1)
        return torch.cat([(p * u).sum((2, 3)), (p * v).sum((2, 3))], dim=1)

    M.SpatialSoftArgmax.forward = forward
    tr = C.load_split(ROOT, "hard", "train")[0]
    va_i, va_l = C.load_split(ROOT, "hard", "val")[:2]
    with torch.no_grad():
        net(to_batch(tr, np.arange(200))[0])
    out = {"calib_images": 200, "logit_range": [st["xmin"], st["xmax"]], "pmax": st["pmax"], "rows": {}}
    for mode in ("fp32", "logits", "probability_map", "both"):
        st["mode"] = mode
        m = evaluate(net, va_i, va_l)
        out["rows"][mode] = m
        print(f"{mode:16s} xy median {m['xy_median_mm']:.4f} mm  radial bias {m['radial_bias_mm']:.4f} mm")
    with open(os.path.join(RUN, "int8_mechanism.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
