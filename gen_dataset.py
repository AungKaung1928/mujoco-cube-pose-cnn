"""Render a labelled cube-pose dataset from MuJoCo.

The labels are free: they are the numbers we typed into the simulator before
rendering. That is the whole reason robot-learning work starts in sim.

Usage (from the repo root, with the virtualenv active):
    python gen_dataset.py --regime easy --n 500          # smoke test
    python gen_dataset.py --regime easy --n 12000
    python gen_dataset.py --regime hard --n 12000
"""
import argparse
import json
import os
import time

import numpy as np
import mujoco

import common as C

ROOT = os.path.dirname(os.path.abspath(__file__))


def make_model(img_size):
    xml = open(os.path.join(ROOT, "scene.xml")).read()
    xml = xml.replace('offwidth="256"', f'offwidth="{img_size}"')
    xml = xml.replace('offheight="256"', f'offheight="{img_size}"')
    return mujoco.MjModel.from_xml_string(xml)


def sample_appearance(model, rng, regime, ids):
    """Mutate colours and lighting in place. Returns nothing."""
    gid_cube, gid_table, lid = ids
    if regime == "easy":
        model.geom_rgba[gid_cube] = [0.80, 0.15, 0.15, 1.0]
        model.geom_rgba[gid_table] = [0.75, 0.75, 0.72, 1.0]
        model.light_pos[lid] = [0.0, 0.0, 1.2]
        model.light_diffuse[lid] = [0.7, 0.7, 0.7]
    else:
        # cube: saturated but arbitrary hue, so a fixed red threshold cannot work
        h = rng.uniform(0, 1)
        model.geom_rgba[gid_cube] = [*hsv_to_rgb(h, rng.uniform(0.6, 1.0), rng.uniform(0.5, 1.0)), 1.0]
        g = rng.uniform(0.25, 0.85)
        model.geom_rgba[gid_table] = [g, g * rng.uniform(0.9, 1.1), g * rng.uniform(0.9, 1.1), 1.0]
        model.light_pos[lid] = [rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.9, 1.4)]
        d = rng.uniform(0.4, 1.0)
        model.light_diffuse[lid] = [d, d, d]


def hsv_to_rgb(h, s, v):
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    return [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]


def generate(regime, n, img_size, seed, split):
    model = make_model(img_size)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, img_size, img_size)
    rng = np.random.default_rng(seed)

    ids = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table"),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_LIGHT, "l0"),
    )

    outdir = os.path.join(ROOT, "data", regime)
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, split)

    imgs = np.lib.format.open_memmap(
        base + "_images.npy", mode="w+", dtype=np.uint8, shape=(n, img_size, img_size, 3)
    )
    # label columns: x, y, sin4t, cos4t, raw_yaw (raw kept only for debugging)
    labels = np.zeros((n, 5), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n):
        x = rng.uniform(-C.XY_RANGE, C.XY_RANGE)
        y = rng.uniform(-C.XY_RANGE, C.XY_RANGE)
        yaw = rng.uniform(0, np.pi / 2)          # sampling wider is pointless, 90deg fold

        # freejoint qpos = [x y z qw qx qy qz]
        data.qpos[:3] = [x, y, C.CUBE_HALF]
        data.qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        sample_appearance(model, rng, regime, ids)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera="top")
        imgs[i] = renderer.render()
        sv, cv = np.sin(C.YAW_FOLD * yaw), np.cos(C.YAW_FOLD * yaw)
        labels[i] = [x, y, sv, cv, yaw]

        if (i + 1) % 500 == 0:
            r = (i + 1) / (time.perf_counter() - t0)
            print(f"  {i+1}/{n}  {r:6.1f} img/s", flush=True)

    dt = time.perf_counter() - t0
    imgs.flush()
    np.save(base + "_labels.npy", labels)

    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(
            {
                "regime": regime,
                "img_size": img_size,
                "px_per_m": float(C.px_per_m(img_size)),
                "xy_range_m": C.XY_RANGE,
                "cube_half_m": C.CUBE_HALF,
                "yaw_fold": C.YAW_FOLD,
                "label_cols": ["x", "y", "sin4yaw", "cos4yaw", "raw_yaw_rad"],
            },
            f,
            indent=2,
        )
    print(f"{regime}/{split}: {n} imgs at {img_size}px in {dt:.1f}s "
          f"({n/dt:.1f} img/s), {imgs.nbytes/1e6:.0f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=C.REGIMES, required=True)
    ap.add_argument("--n", type=int, default=12000, help="training-split size")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # held-out split gets a different seed stream, never mixed with train
    generate(a.regime, a.n, a.img_size, a.seed, "train")
    generate(a.regime, max(200, a.n // 6), a.img_size, a.seed + 10_000, "val")
