"""Shared constants and geometry for the cube-pose task.

Everything here is deliberately explicit. No hidden magic, because every number
in this file shows up later in the error metrics.
"""
import json
import numpy as np

# --- scene constants, must match scene.xml ---
CAM_HEIGHT = 0.45        # m, camera z
CUBE_HALF = 0.03         # m, box half-size -> cube top face is at z = 0.06
FOVY_DEG = 45.0
TABLE_HALF = 0.40        # m

# cube centre is sampled inside this square (m)
XY_RANGE = 0.075

# Yaw of a cube with a square top face is only observable modulo 90 degrees.
# A camera cannot tell 5deg from 95deg. So the label is not the angle itself but
# (sin 4t, cos 4t), which is identical for t and t+90deg and has no wrap-around
# discontinuity. Regressing raw degrees would put a cliff at the range boundary
# and the model would learn to sit in the middle of it.
YAW_FOLD = 4


def yaw_to_vec(yaw):
    """radians -> (sin 4t, cos 4t)"""
    return np.stack([np.sin(YAW_FOLD * yaw), np.cos(YAW_FOLD * yaw)], -1)


def vec_to_yaw(v):
    """(sin 4t, cos 4t) -> radians folded into [0, 90deg)"""
    t = np.arctan2(v[..., 0], v[..., 1]) / YAW_FOLD
    return np.mod(t, np.pi / 2)


def yaw_err_deg(a, b):
    """smallest angular distance between two yaws, respecting the 90deg fold"""
    d = np.mod(a - b, np.pi / 2)
    d = np.minimum(d, np.pi / 2 - d)
    return np.degrees(d)


def px_per_m(img_size):
    """Overhead pinhole camera. The cube's *centre* sits at z = CUBE_HALF, so the
    object plane is CAM_HEIGHT - CUBE_HALF below the camera."""
    d = CAM_HEIGHT - CUBE_HALF
    half_extent = d * np.tan(np.radians(FOVY_DEG) / 2)   # m visible from centre to edge
    return (img_size / 2) / half_extent


def world_to_pixel(x, y, img_size):
    """MuJoCo cam x-axis = world +x, cam y-axis = world +y, image v grows downward."""
    s = px_per_m(img_size)
    u = img_size / 2 + np.asarray(x) * s
    v = img_size / 2 - np.asarray(y) * s
    return u, v


def pixel_to_world(u, v, img_size):
    s = px_per_m(img_size)
    x = (np.asarray(u) - img_size / 2) / s
    y = (img_size / 2 - np.asarray(v)) / s
    return x, y


# --- dataset regimes ---
# easy: fixed appearance. A colour-threshold baseline should nail this, and that
#       is the point: it shows where classical CV is simply better.
# hard: appearance randomised. The threshold baseline degrades; a CNN should not.
REGIMES = ("easy", "hard")


def load_split(root, regime, split):
    import os
    p = os.path.join(root, "data", regime, split)
    imgs = np.load(p + "_images.npy", mmap_mode="r")
    labels = np.load(p + "_labels.npy")
    with open(os.path.join(root, "data", regime, "meta.json")) as f:
        meta = json.load(f)
    return imgs, labels, meta


# --- scoring: one implementation, used by BOTH the classical baseline and the CNN ---
# If these were computed in two places they would drift, and the comparison that
# this whole project exists to make would be worthless.

def pose_metrics(pred_xy, pred_yaw, gt_xy, gt_yaw):
    """All errors in mm / degrees. Inputs are metres / radians, already filtered
    to the detected subset."""
    dxy = np.linalg.norm(np.asarray(pred_xy) - np.asarray(gt_xy), axis=1) * 1e3
    dyaw = yaw_err_deg(np.asarray(pred_yaw), np.asarray(gt_yaw))
    # Systematic radial bias: the silhouette centroid of a 3D box sits outward of
    # the projected centre, because the camera sees the side faces. A hand-written
    # centroid cannot know that; a network can absorb it from data.
    r_gt = np.linalg.norm(np.asarray(gt_xy), axis=1)
    r_pr = np.linalg.norm(np.asarray(pred_xy), axis=1)
    return {
        "xy_median_mm": float(np.median(dxy)),
        "xy_p95_mm": float(np.percentile(dxy, 95)),
        "yaw_median_deg": float(np.median(dyaw)),
        "yaw_p95_deg": float(np.percentile(dyaw, 95)),
        "radial_bias_mm": float(np.mean(r_pr - r_gt) * 1e3),
    }


def print_metrics(m, n_ok, n, px_per_m, lat_ms=None, header=""):
    if header:
        print(f"\n{header}")
    print(f"  detected            {n_ok}/{n}  ({100*(1-n_ok/n):.1f}% miss)")
    print(f"  xy  err  median     {m['xy_median_mm']:7.2f} mm      p95 {m['xy_p95_mm']:7.2f} mm")
    print(f"  yaw err  median     {m['yaw_median_deg']:7.2f} deg     p95 {m['yaw_p95_deg']:7.2f} deg")
    print(f"  radial bias (mean)  {m['radial_bias_mm']:+7.2f} mm")
    if lat_ms is not None:
        print(f"  latency             {lat_ms:7.2f} ms/img")
    print(f"  pixel floor         {1000/px_per_m:7.2f} mm/px")
