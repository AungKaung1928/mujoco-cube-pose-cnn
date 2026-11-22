"""common.py's camera constants checked against an actual MuJoCo render of scene.xml.

test_common.py checks the projection formulas on paper; it cannot see a scene.xml
edit (camera height, fovy, cube size, axis flip). Every label and every metric
depends on the two agreeing, so here the renderer is the referee:

  - cube at the origin: silhouette = top face (z = 2*CUBE_HALF), centred on the
    image centre, width 2*CUBE_HALF * px_per_m * (CAM_HEIGHT-CUBE_HALF)/(CAM_HEIGHT-2*CUBE_HALF)
  - cube moved to +x / +y: silhouette moves to +u / -v, the direction world_to_pixel says

Run:  MUJOCO_GL=egl python test_render.py   (or osmesa / glfw)
"""
import numpy as np

import common as C
import gen_dataset as G

IMG = 128
FAILED = []


def check(name, ok, detail):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name:44s} {detail}")
    if not ok:
        FAILED.append(name)


def render_mask(model, data, renderer, x, y):
    data.qpos[:3] = [x, y, C.CUBE_HALF]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    G.mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="top")
    img = renderer.render().astype(int)
    # easy regime: cube (0.80, 0.15, 0.15) on a grey table, so red dominance is the cube
    return (img[..., 0] - img[..., 1]) > 60


def main():
    model = G.make_model(IMG)
    data = G.mujoco.MjData(model)
    renderer = G.mujoco.Renderer(model, IMG, IMG)
    ids = (G.mujoco.mj_name2id(model, G.mujoco.mjtObj.mjOBJ_GEOM, "cube_geom"),
           G.mujoco.mj_name2id(model, G.mujoco.mjtObj.mjOBJ_GEOM, "table"),
           G.mujoco.mj_name2id(model, G.mujoco.mjtObj.mjOBJ_LIGHT, "l0"))
    G.sample_appearance(model, None, "easy", ids)

    print("\nscene.xml as loaded == common.py constants")
    cam = G.mujoco.mj_name2id(model, G.mujoco.mjtObj.mjOBJ_CAMERA, "top")
    check("camera height", np.isclose(model.cam_pos[cam][2], C.CAM_HEIGHT), f"{model.cam_pos[cam][2]}")
    check("camera fovy", np.isclose(model.cam_fovy[cam], C.FOVY_DEG), f"{model.cam_fovy[cam]}")
    check("cube half-size", np.allclose(model.geom_size[ids[0]], C.CUBE_HALF), f"{model.geom_size[ids[0]]}")

    print("\nrendered cube at the origin")
    m = render_mask(model, data, renderer, 0.0, 0.0)
    vs, us = np.nonzero(m)
    cu, cv = us.mean() + 0.5, vs.mean() + 0.5          # pixel centres -> continuous coords
    check("centroid at image centre (+-0.5 px)", abs(cu - IMG / 2) < 0.5 and abs(cv - IMG / 2) < 0.5,
          f"({cu:.2f}, {cv:.2f})")
    top_scale = (C.CAM_HEIGHT - C.CUBE_HALF) / (C.CAM_HEIGHT - 2 * C.CUBE_HALF)
    want_w = 2 * C.CUBE_HALF * C.px_per_m(IMG) * top_scale
    got_w = np.sqrt(m.sum())                          # square silhouette: side = sqrt(area)
    check("top-face width (+-1 px)", abs(got_w - want_w) < 1.0, f"got {got_w:.2f} want {want_w:.2f} px")

    print("\nrendered cube off-centre: axis directions")
    for x, y in [(0.05, 0.0), (0.0, 0.05), (-0.04, -0.03)]:
        vs, us = np.nonzero(render_mask(model, data, renderer, x, y))
        u, v = C.world_to_pixel(x, y, IMG)
        du, dv = us.mean() + 0.5 - IMG / 2, vs.mean() + 0.5 - IMG / 2
        eu, ev = u - IMG / 2, v - IMG / 2
        # the silhouette centroid sits outward of the projected centre (side faces), never inward
        same = all(abs(d) < 0.5 if abs(e) < 0.5 else (np.sign(d) == np.sign(e) and abs(d) >= abs(e) - 0.5)
                   for d, e in [(du, eu), (dv, ev)])
        check(f"cube ({x:+.2f}, {y:+.2f}) m", same,
              f"centroid offset ({du:+.1f}, {dv:+.1f}) px, projection ({eu:+.1f}, {ev:+.1f}) px")

    print(f"\n{'ALL PASSED' if not FAILED else 'FAILED: ' + ', '.join(FAILED)}")
    raise SystemExit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
