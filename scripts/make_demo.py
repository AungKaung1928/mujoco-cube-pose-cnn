"""docs/media/predictions.png: the tracked soft-argmax ONNX model on the first 8 `hard` val
images (no selection). White = ground-truth top face, black = predicted, error in the label.

Needs data/ (python gen_dataset.py --regime hard --n 12000, ~30 s). One thread, seconds.

    python scripts/make_demo.py
"""
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
import common as C                       # noqa: E402
from export_onnx import decode           # noqa: E402

N, SCALE, BAR = 8, 2, 18


def square(x, y, yaw, size):
    """Top-face corners in pixels: centre (x, y) m, yaw rad, image `size` px."""
    h = C.CUBE_HALF
    c, s = np.cos(yaw), np.sin(yaw)
    pts = [(x + c * dx - s * dy, y + s * dx + c * dy) for dx, dy in ((h, h), (-h, h), (-h, -h), (h, -h))]
    return [tuple(float(p) * SCALE for p in C.world_to_pixel(px, py, size)) for px, py in pts]


def main():
    imgs, labels, meta = C.load_split(ROOT, "hard", "val")
    size = meta["img_size"]
    x = np.ascontiguousarray(imgs[:N]).transpose(0, 3, 1, 2).astype(np.float32) / 127.5 - 1.0
    so = ort.SessionOptions()
    so.intra_op_num_threads = so.inter_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(ROOT, "runs/hard_softargmax/model.onnx"), so,
                                providers=["CPUExecutionProvider"])
    pxy, pyaw = decode(sess.run(["pose"], {"image": x})[0].astype(np.float64))
    err_mm = np.linalg.norm(pxy - labels[:N, :2], axis=1) * 1000
    cell = size * SCALE
    sheet = Image.new("RGB", (cell * 4, (cell + BAR) * 2), "white")
    for i in range(N):
        im = Image.fromarray(np.asarray(imgs[i])).resize((cell, cell), Image.NEAREST)
        d = ImageDraw.Draw(im)
        d.polygon(square(labels[i, 0], labels[i, 1], labels[i, 4], size), outline="white", width=3)
        d.polygon(square(pxy[i, 0], pxy[i, 1], pyaw[i], size), outline="black", width=1)
        r, c = divmod(i, 4)
        sheet.paste(im, (c * cell, r * (cell + BAR) + BAR))
        ImageDraw.Draw(sheet).text((c * cell + 4, r * (cell + BAR) + 3),
                                   f"xy err {err_mm[i]:.2f} mm, yaw err {C.yaw_err_deg(pyaw[i], labels[i, 4]):.2f} deg",
                                   fill="black")
    path = os.path.join(ROOT, "docs", "media", "predictions.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sheet.save(path, optimize=True)
    print(path, os.path.getsize(path) // 1024, "KB; median xy err of these", N, f"{np.median(err_mm):.2f} mm")


if __name__ == "__main__":
    main()
