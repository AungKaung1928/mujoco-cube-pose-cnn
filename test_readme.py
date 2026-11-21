"""README results table (CNN and classical rows) against runs/*/metrics.json (stdlib only).

Run:  python test_readme.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FAILED = []


def check(name, ok, detail=""):
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


with open(os.path.join(HERE, "README.md"), encoding="utf-8") as f:
    LINES = f.read().splitlines()

for label, run in (("CNN, flatten head", "hard_flat"), ("CNN, soft-argmax head", "hard_softargmax")):
    with open(os.path.join(HERE, "runs", run, "metrics.json")) as f:
        m = json.load(f)
    line = next(l for l in LINES if l.startswith(f"| {label} |"))
    got = [c.strip().strip("*") for c in line.strip().strip("|").split("|")][4:9]
    want = [f"{m['xy_median_mm']:.2f} mm", f"{m['xy_p95_mm']:.2f}", f"{m['yaw_median_deg']:.2f} deg",
            f"{m['yaw_p95_deg']:.2f}", f"{m['latency_ms']:.2f} ms"]
    check(label, got == want, f"README {got} vs JSON {want}")

# Classical rows, from `baseline_cv.py --regime hard --method red|sat --out runs/baseline_<m>/metrics.json`
# and `baseline_v2.py --regime hard --method sat --out runs/baseline_v2/metrics.json`:
# accuracy and detection rate only. Their 0.05-0.10 ms latency is a 2 s timing
# loop that moves between runs, so the README cell is not tied to the file.
for label, run in (("OpenCV, fixed red hue", "baseline_red"), ("OpenCV, saturation", "baseline_sat"),
                   ("**v2** OpenCV + 1 calibrated scalar", "baseline_v2")):
    with open(os.path.join(HERE, "runs", run, "metrics.json")) as f:
        m = json.load(f)
    line = next(l for l in LINES if l.startswith(f"| {label} |"))
    got = [c.strip().strip("*") for c in line.strip().strip("|").split("|")][3:8]
    tenths = (2000 * m["detected"] + m["n"]) // (2 * m["n"])   # percent x 10, half rounds up: 217/2000 = 10.85
    want = ["100%" if m["detected"] == m["n"] else f"{tenths // 10}.{tenths % 10}%",
            f"{m['xy_median_mm']:.2f} mm", f"{m['xy_p95_mm']:.2f}", f"{m['yaw_median_deg']:.2f} deg",
            f"{m['yaw_p95_deg']:.2f}"]
    check(label, got == want, f"README {got} vs JSON {want}")

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
