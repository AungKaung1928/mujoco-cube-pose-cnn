"""Classical CV baseline: threshold -> largest contour -> centroid + minAreaRect.

No learning. This is the number the CNN has to beat, and on the `easy` regime it
may well not. Two hand-written thresholds are provided on purpose:

  --method red   fixed hue window. The naive choice. Encodes "the cube is red".
  --method sat   saturation only. Encodes the weaker, more honest prior
                 "the cube is the most colourful thing on a near-grey table".

Comparing the two shows that "classical CV" is not one method, and that choosing
a better hand-written feature often beats reaching for a network.

    python baseline_cv.py --regime easy --method red
    python baseline_cv.py --regime hard --method sat
"""
import argparse
import os
import time

import cv2
import numpy as np

import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))


def mask_red(hsv):
    lo1 = cv2.inRange(hsv, (0, 100, 50), (10, 255, 255))
    lo2 = cv2.inRange(hsv, (170, 100, 50), (180, 255, 255))
    return cv2.bitwise_or(lo1, lo2)


def mask_sat(hsv):
    return cv2.inRange(hsv, (0, 90, 40), (180, 255, 255))


MASKS = {"red": mask_red, "sat": mask_sat}


def predict(img_rgb, method, size):
    """-> (x, y, yaw) in world units, or None if nothing was detected."""
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    m = MASKS[method](hsv)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:          # cube is ~22x22 px, so ~480 px^2; 30 is a floor
        return None
    (u, v), (w, h), ang = cv2.minAreaRect(c)
    x, y = C.pixel_to_world(u, v, size)
    # image v grows downward, so a rotation that is CCW in the image is CW in world
    yaw = np.mod(np.radians(-ang), np.pi / 2)
    return float(x), float(y), float(yaw)


def main(regime, split, method, grid):
    imgs, labels, meta = C.load_split(ROOT, regime, split)
    size = meta["img_size"]
    n = len(imgs)

    preds = np.full((n, 3), np.nan, np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        p = predict(np.ascontiguousarray(imgs[i]), method, size)
        if p is not None:
            preds[i] = p
    lat_ms = (time.perf_counter() - t0) / n * 1e3

    ok = ~np.isnan(preds[:, 0])
    gt_xy, gt_yaw = labels[:, :2], labels[:, 4]
    m = C.pose_metrics(preds[ok, :2], preds[ok, 2], gt_xy[ok], gt_yaw[ok])
    C.print_metrics(m, int(ok.sum()), n, meta["px_per_m"], lat_ms,
                    header=f"{regime}/{split}  method={method}  n={n}")

    if grid:
        tiles = []
        for i in range(min(15, n)):
            im = np.ascontiguousarray(imgs[i][:, :, ::-1])
            gu, gv = C.world_to_pixel(*gt_xy[i], size)
            cv2.drawMarker(im, (int(gu), int(gv)), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
            if ok[i]:
                pu, pv = C.world_to_pixel(*preds[i, :2], size)
                cv2.drawMarker(im, (int(pu), int(pv)), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)
            tiles.append(im)
        sheet = np.vstack([np.hstack(tiles[r*5:(r+1)*5]) for r in range(3)])
        out = os.path.join(ROOT, "out", f"baseline_{regime}_{method}.png")
        cv2.imwrite(out, sheet)
        print(f"  wrote {out}   (green = ground truth, red = baseline)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=C.REGIMES, default="easy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--method", choices=list(MASKS), default="sat")
    ap.add_argument("--grid", action="store_true")
    a = ap.parse_args()
    main(a.regime, a.split, a.method, a.grid)
