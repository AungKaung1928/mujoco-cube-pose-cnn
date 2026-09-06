"""Hand-computable checks on common.py -- the conventions every number depends on.

Every expected value here is derived on paper. If the yaw fold or the projection is
wrong, the baseline and the CNN are still scored against each other consistently,
so nothing downstream would complain; only the numbers would be meaningless.

Run:  python test_common.py
"""
import numpy as np

import common as C

FAILED = []


def check(name, got, want, tol=1e-9):
    got, want = np.asarray(got, float), np.asarray(want, float)
    ok = np.all(np.abs(got - want) <= tol)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:46s} got {np.round(got, 6)}  want {np.round(want, 6)}")
    if not ok:
        FAILED.append(name)


print("\nyaw fold: a square top face repeats every 90 deg")
d = np.radians
# 5 deg and 95 deg must produce the same label vector
check("yaw_to_vec(5) == yaw_to_vec(95)", C.yaw_to_vec(d(5)), C.yaw_to_vec(d(95)))
# and the same folded angle back
check("vec_to_yaw(vec(95)) == 5 deg", np.degrees(C.vec_to_yaw(C.yaw_to_vec(d(95)))), 5.0, 1e-6)
# round trip inside the fold
for t in (0.0, 10.0, 44.9, 45.0, 89.9):
    check(f"round trip {t} deg", np.degrees(C.vec_to_yaw(C.yaw_to_vec(d(t)))), t, 1e-6)
# the label has no discontinuity at the fold boundary: 89.9 and 0.1 are neighbours
v_a, v_b = C.yaw_to_vec(d(89.9)), C.yaw_to_vec(d(0.1))
check("label continuous across 90 deg (|dv| < 0.03)", np.linalg.norm(v_a - v_b) < 0.03, True)
# error metric respects the fold: 89 deg vs 1 deg is a 2 deg error, not 88
check("yaw_err(89, 1) == 2 deg", C.yaw_err_deg(d(89), d(1)), 2.0, 1e-9)
check("yaw_err(1, 89) is symmetric", C.yaw_err_deg(d(1), d(89)), 2.0, 1e-9)
check("yaw_err max is 45 deg", C.yaw_err_deg(d(45), d(0)), 45.0, 1e-9)
check("yaw_err(t, t+90) == 0", C.yaw_err_deg(d(30), d(120)), 0.0, 1e-9)

print("\noverhead projection: exactly affine, so pixel<->world must invert exactly")
size = 128
s = C.px_per_m(size)
# half the image spans d*tan(fovy/2) metres, d = 0.45 - 0.03 = 0.42
check("px_per_m(128)", s, 64 / (0.42 * np.tan(np.radians(22.5))), 1e-9)
check("1 px == 2.72 mm (README)", 1000 / s, 2.718, 2e-3)
check("world origin -> image centre", C.world_to_pixel(0.0, 0.0, size), (64.0, 64.0))
u, v = C.world_to_pixel(0.05, -0.02, size)
check("+x is right, -y is down", (u > 64, v > 64), (True, True))
check("pixel_to_world(world_to_pixel(p)) == p", C.pixel_to_world(u, v, size), (0.05, -0.02), 1e-12)
check("scale is linear in image size", C.px_per_m(256) / C.px_per_m(128), 2.0, 1e-12)

print("\npose_metrics: units and the radial-bias sign")
gt = np.array([[0.05, 0.0], [0.0, -0.05]])
pr = gt * 1.02                                   # 2% radially outward
m = C.pose_metrics(pr, np.zeros(2), gt, np.zeros(2))
check("xy err in mm (0.001 m -> 1.0 mm)", m["xy_median_mm"], 1.0, 1e-9)
check("outward prediction -> positive radial bias (mm)", m["radial_bias_mm"], 1.0, 1e-9)
m = C.pose_metrics(gt, np.full(2, d(1.0)), gt, np.full(2, d(89.0)))
check("yaw err reported in deg, folded", m["yaw_median_deg"], 2.0, 1e-9)

print()
if FAILED:
    print(f"{len(FAILED)} FAILED: {FAILED}")
    raise SystemExit(1)
print("all hand-computed cases pass")
