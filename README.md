# mujoco-cube-pose-cnn

Estimates the planar pose `(x, y, yaw)` of a cube from one 128x128 MuJoCo render: a 27k-parameter CNN with a soft-argmax head reaches 0.59 mm median error against 1.91 mm for a calibrated OpenCV baseline, in about 0.2 ms on one thread through ONNX Runtime.

<img src="docs/media/predictions.png" width="720" alt="First 8 hard val images, white outline ground-truth top face, black outline predicted, per-image error above each">

The tracked ONNX model on the first 8 `hard` val images, not selected: white = ground truth, black = prediction (`scripts/make_demo.py`, needs `data/`).

All numbers: `hard` regime (random hue, light, table shade), `val` split, n=2000, median xy error unless stated.

```mermaid
flowchart LR
    R["MuJoCo render<br/>128x128 RGB, overhead camera<br/>1 px = 2.72 mm<br/>12k train / 2k val"]
    B1["Baseline 1: OpenCV threshold<br/>red hue: detect 10.9%, 3.48 mm<br/>saturation: detect 100%, 3.41 mm<br/>radial bias +2.98 mm, 0.05 ms"]
    B2["Baseline 2: + 1 scalar fit on train<br/>k = 0.951, radial bias +0.01 mm<br/>1.91 mm median, p95 2.89 mm<br/>0.06 ms"]
    C1["CNN flatten head, 130k params<br/>0.75 mm median, p95 2.00 mm<br/>1.70 ms"]
    C2["CNN soft-argmax head, 27k params<br/>0.59 mm median, p95 1.32 mm<br/>0.65 ms PyTorch, 8 threads"]
    O["ONNX Runtime, same weights<br/>0.59 mm, max abs diff 6.6e-7<br/>0.23 ms on 1 thread, 0.12 ms on 8"]
    R --> B1
    B1 -->|"fix the -4.9% radial bias: 3.41 to 1.91 mm, 53% of the gap"| B2
    B2 -->|"the rest of the gap: 1.91 to 0.59 mm"| C2
    B2 -.-> C1
    C2 --> O
```

### Results

| method | params | regime | detect | median xy | p95 xy | median yaw | p95 yaw | latency |
|---|---|---|---|---|---|---|---|---|
| OpenCV, fixed red hue | 0 | `hard` | **10.9%** | 3.48 mm | 6.07 | 0.19 deg | 1.24 | 0.06 ms |
| OpenCV, saturation | 0 | `hard` | 100% | 3.41 mm | 5.57 | 0.19 deg | 1.66 | 0.05 ms |
| **v2** OpenCV + 1 calibrated scalar | 1 | `hard` | 100% | 1.91 mm | 2.89 | 0.19 deg | 1.66 | 0.06 ms |
| CNN, flatten head | 130k | `hard` | 100% | 0.75 mm | 2.00 | 0.28 deg | 0.94 | 1.70 ms |
| CNN, soft-argmax head | **27k** | `hard` | 100% | **0.59 mm** | **1.32** | 0.21 deg | **0.66** | 0.65 ms |
| same, ONNX Runtime, 1 thread | 27k | `hard` | 100% | 0.59 mm | 1.32 | 0.21 deg | 0.66 | **0.23 ms** |
| same, ONNX Runtime, 8 threads | 27k | `hard` | 100% | 0.59 mm | 1.32 | 0.21 deg | 0.66 | **0.12 ms** |

One-thread ONNX latency moves between runs on this laptop: 0.23 ms in the table, 0.27 ms in the committed `runs/hard_softargmax/onnx_report.json`, 0.19 to 0.23 ms in reruns on 2026-11-21. Read it as about 0.2 ms; the accuracy columns are exact across runs.

The `easy` regime, the radial-bias column, INT8 results and how each was found: [docs/RESULTS.md](docs/RESULTS.md).

## Quick start

Linux, Python 3.10+, CPU only.

```bash
git clone <this repo URL>
cd mujoco-cube-pose-cnn
python3 -m venv .venv && . .venv/bin/activate        # first time only
pip install --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple -r requirements.txt   # first time only
./verify.sh
# tier 1: yaw fold, projection, metric, head parameter counts, all "ok", under 10 s.
# Without data/ it stops there and prints the two commands that render it (~30 s, one thread);
# run them and ./verify.sh again to re-score the tracked checkpoints through PyTorch and ONNX Runtime.
python test_readme.py   # the table above against runs/*/metrics.json
```

## How it works

- `gen_dataset.py` renders a cube at a random pose under an overhead camera; `easy` keeps colours fixed, `hard` randomises hue, light and table shade.
- `baseline_cv.py`: threshold, contour, `minAreaRect`, then one scalar fitted on train removes the radial bias.
- `model.py`: small CNN with a flatten head or a spatial soft-argmax head; `export_onnx.py` and `quantize_int8.py` for ONNX Runtime fp32 and INT8.

## License

MIT, see LICENSE.
