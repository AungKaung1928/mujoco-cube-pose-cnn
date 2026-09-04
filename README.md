# mujoco-cube-pose-cnn

Block 1 of the CPU-only Physical AI track. **Skills: PyTorch basics · dataset creation · classical CV baseline.**

Estimate the planar pose `(x, y, yaw)` of a cube on a table from a single 128x128 RGB
image rendered in MuJoCo, and answer one question with numbers:

> When does a learned model beat hand-written geometry, and when does it not?

Two dataset regimes, one model, one classical baseline, four measurements.

| regime | appearance | expectation |
|---|---|---|
| `easy` | fixed cube colour, fixed light | colour-threshold baseline should win or tie |
| `hard` | random cube hue, random light position and intensity, random table shade | baseline should degrade, CNN should not |

## Setup

```bash
source ~/personal/ml/env.sh     # CPU torch, MUJOCO_GL=glfw, threads capped at 10
cd ~/personal/ml/mujoco-cube-pose-cnn
```

## Steps

- [x] **1. Data.** `gen_dataset.py` renders randomised scenes and writes labels straight
      from the simulator state. `view_dataset.py` draws the labels back onto the images
      through the analytic overhead projection — if the markers miss the cubes, the label
      convention is wrong and nothing downstream means anything.
- [ ] 2. Classical baseline: threshold -> largest contour -> centroid + `minAreaRect`.
- [ ] 3. Small CNN regression in PyTorch, trained on CPU.
- [ ] 4. Evaluation on the held-out split: median and 95th-percentile error.
- [ ] 5. ONNX export + `onnxruntime` latency.

Nothing moves to block 2 until the results table below is filled with real numbers.

## Task definition

- **Position** `(x, y)` in metres, cube centre, table frame. Sampled in `+-0.075 m`.
- **Yaw** is only observable **modulo 90 degrees** — a square top face looks identical at
  5 deg and 95 deg. So the network predicts `(sin 4t, cos 4t)`, not degrees. Regressing raw
  degrees puts a discontinuity at the range boundary and the model learns to hedge in the
  middle of it. `minAreaRect` has exactly the same ambiguity, so both methods are scored
  under the same convention.
- **Camera** is overhead and parallel to the table, so pixel <-> world is exactly affine:
  `367.9 px/m`, i.e. **1 px = 2.72 mm**. That is the quantisation floor of the whole task —
  no method can beat about 1 mm here, and any reported error below that is a bug.

## Results

Held-out split, generated with an independent seed stream. Fill in as steps land.

| method | regime | detect | median xy (mm) | p95 xy (mm) | median yaw (deg) | latency (ms) |
|---|---|---|---|---|---|---|
| OpenCV, fixed red hue | easy | 100% | 3.50 | 5.62 | 0.20 | 0.09 |
| OpenCV, saturation | easy | 100% | 3.54 | 5.54 | 0.18 | 0.11 |
| OpenCV, fixed red hue | hard | **10%** | 3.84 | 5.76 | 0.14 | 0.11 |
| OpenCV, saturation | hard | 100% | 3.73 | 5.69 | 0.17 | 0.11 |
| CNN | easy | | | | | |
| CNN | hard | | | | | |

`val` split, n=200, independent seed stream. Pixel floor 2.72 mm.

**What step 2 already settled, before any network existed:**

1. The naive prior ("the cube is red") does not survive appearance randomisation —
   90% detection failure on `hard`. The *accuracy* on the 10% it did find is
   unchanged, which is the classic trap: a method can look fine on its own metric
   while silently answering only the easy tenth of the data.
2. A better hand-written feature ("the cube is the most saturated thing on a
   near-grey table") survives it completely — 100% detection, same error as `easy`.
   So robustness is **not** the CNN's selling point here. Picking a stronger
   classical feature was cheaper and worked.
3. The remaining error is not noise. Mean radial bias is **+3.2 mm** against a
   3.7 mm median error, so the baseline is dominated by a *systematic* effect: the
   silhouette centroid of a 3D box sits outward of the projected centre, because
   the camera sees the cube's side faces. A hand-written centroid cannot know that.

That third point is the CNN's actual opening: it should absorb the unmodelled
perspective for free and land near the 2.72 mm pixel floor. If it does not beat
3.7 mm median, it has no reason to exist here — write that down as the result.

**Fair-baseline note:** the bias is analytically correctable (project the top-face
outline instead of assuming a point at cube centre). Do that as `baseline v2`
*after* the CNN, so the comparison stays honest in both directions.

## Measured environment

Intel Core Ultra 5 225H, 14 cores, no GPU, WSL2, software OpenGL (llvmpipe).
Rendering is **not** the bottleneck: 800-900 img/s at 128 px, so 12k images takes ~15 s.
