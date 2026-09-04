"""Classical baseline, but calibrated -- the fair comparison.

baseline_cv.py loses to the CNN by ~4.5x on position, and essentially all of that
gap is one systematic effect: the silhouette centroid of a 3D box sits radially
outward of the projected cube centre, because an overhead camera also sees the
outward-facing side walls. The CNN removes it because it is trained on labels.

That is not a fair fight. The CNN saw 12000 labelled images; v1 saw none. So give
the classical method the same privilege and no more: fit ONE scalar -- a radial
scale correction -- on the TRAIN split, then apply it unchanged to VAL.

    r_corrected = k * r_measured,   k = argmin || k*r_meas - r_true ||

If that single number closes most of the gap, then the CNN was not solving a
perception problem, it was doing calibration the expensive way. Knowing which of
the two it is, is the entire point of building a baseline.
"""
import argparse, os, time
import numpy as np

import common as C
from baseline_cv import predict

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(regime, split, method):
    imgs, labels, meta = C.load_split(ROOT, regime, split)
    size, n = meta["img_size"], len(imgs)
    preds = np.full((n, 3), np.nan, np.float64)
    t0 = time.perf_counter()
    for i in range(n):
        p = predict(np.ascontiguousarray(imgs[i]), method, size)
        if p is not None:
            preds[i] = p
    return preds, labels, meta, (time.perf_counter() - t0) / n * 1e3


def main(a):
    tr_p, tr_l, meta, _ = run(a.regime, "train", a.method)
    ok = ~np.isnan(tr_p[:, 0])
    r_meas = np.linalg.norm(tr_p[ok, :2], axis=1)
    r_true = np.linalg.norm(tr_l[ok, :2], axis=1)
    k = float((r_meas * r_true).sum() / (r_meas ** 2).sum())
    print(f"calibrated on {a.regime}/train  n={ok.sum()}  radial scale k = {k:.5f} "
          f"({100*(k-1):+.2f}%)")

    va_p, va_l, meta, lat = run(a.regime, "val", a.method)
    ok = ~np.isnan(va_p[:, 0])
    xy = va_p[ok, :2] * k                       # one scalar, that is the whole model
    m = C.pose_metrics(xy, va_p[ok, 2], va_l[ok, :2], va_l[ok, 4])
    C.print_metrics(m, int(ok.sum()), int(ok.size), meta["px_per_m"], lat,
                    header=f"baseline v2 (calibrated)  {a.regime}/val  method={a.method}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="hard", choices=C.REGIMES)
    p.add_argument("--method", default="sat", choices=["red", "sat"])
    main(p.parse_args())
