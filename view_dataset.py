"""Eyeball the dataset. Always look at your data before training on it.

Draws the ground-truth centre and the yaw axis on a grid of samples, using the
analytic overhead projection in common.py. If the markers do not land on the
cubes, the label convention is wrong and every metric downstream is garbage.

    python view_dataset.py --regime easy
    -> out/easy_train_grid.png
"""
import argparse
import os

import cv2
import numpy as np

import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))


def annotate(img, x, y, yaw, size):
    img = np.ascontiguousarray(img[:, :, ::-1])   # RGB -> BGR for cv2
    u, v = C.world_to_pixel(x, y, size)
    cv2.circle(img, (int(round(u)), int(round(v))), 2, (0, 255, 0), -1)
    L = 0.045 * C.px_per_m(size)                  # half-length of the drawn axis
    du, dv = L * np.cos(yaw), -L * np.sin(yaw)
    cv2.line(img, (int(u - du), int(v - dv)), (int(u + du), int(v + dv)), (0, 255, 0), 1)
    return img


def main(regime, split, cols, rows):
    imgs, labels, meta = C.load_split(ROOT, regime, split)
    size = meta["img_size"]
    n = min(cols * rows, len(imgs))
    tiles = [annotate(imgs[i], *labels[i, [0, 1, 4]], size) for i in range(n)]
    while len(tiles) < cols * rows:
        tiles.append(np.zeros((size, size, 3), np.uint8))
    sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    out = os.path.join(ROOT, "out", f"{regime}_{split}_grid.png")
    cv2.imwrite(out, sheet)
    print(f"{out}  ({len(imgs)} imgs in split, showing {n})")
    print(f"px_per_m = {meta['px_per_m']:.1f}  -> 1 px = {1000/meta['px_per_m']:.2f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=C.REGIMES, default="easy")
    ap.add_argument("--split", default="train")
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--rows", type=int, default=3)
    a = ap.parse_args()
    main(a.regime, a.split, a.cols, a.rows)
