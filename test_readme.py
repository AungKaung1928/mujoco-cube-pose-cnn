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

# The pipeline diagram at the top repeats the headline numbers (latency left out, as above).
def load(run, name="metrics.json"):
    with open(os.path.join(HERE, "runs", run, name)) as f:
        return json.load(f)


red, sat, v2 = load("baseline_red"), load("baseline_sat"), load("baseline_v2")
flat, soft = load("hard_flat"), load("hard_softargmax")
onnx = load("hard_softargmax", "onnx_report.json")
tenths = (2000 * red["detected"] + red["n"]) // (2 * red["n"])
gap = (sat["xy_median_mm"] - v2["xy_median_mm"]) / (sat["xy_median_mm"] - soft["xy_median_mm"])
diagram = [
    f"red hue: detect {tenths // 10}.{tenths % 10}%, {red['xy_median_mm']:.2f} mm",
    f"saturation: detect 100%, {sat['xy_median_mm']:.2f} mm" if sat["detected"] == sat["n"] else "?",
    f"radial bias {sat['radial_bias_mm']:+.2f} mm",
    f"k = {v2['radial_scale_k']:.3f}, radial bias {v2['radial_bias_mm']:+.2f} mm",
    f"{v2['xy_median_mm']:.2f} mm median, p95 {v2['xy_p95_mm']:.2f} mm",
    f"{flat['params'] // 1000}k params<br/>{flat['xy_median_mm']:.2f} mm median, p95 {flat['xy_p95_mm']:.2f} mm",
    f"{soft['params'] // 1000}k params<br/>{soft['xy_median_mm']:.2f} mm median, p95 {soft['xy_p95_mm']:.2f} mm",
    f"{soft['xy_median_mm']:.2f} mm, max abs diff {onnx['max_abs_diff']:.1e}".replace("e-0", "e-"),
    f"fix the {100 * (v2['radial_scale_k'] - 1):.1f}% radial bias: {sat['xy_median_mm']:.2f} to "
    f"{v2['xy_median_mm']:.2f} mm, {100 * gap:.0f}% of the gap",
]
text = "\n".join(LINES)
check("pipeline diagram", all(d in text for d in diagram),
      f"missing: {[d for d in diagram if d not in text]}")

q0, qs, qk = (load("hard_softargmax", f"int8_report{t}.json") for t in ("", "_fp32softargmax", "_fp32kp"))
int8 = [f"from {q0['fp32']['xy_median_mm']:.3f} to {q0['int8']['xy_median_mm']:.3f} mm",
        f"radial bias from {q0['fp32']['radial_bias_mm']:.2f} to {q0['int8']['radial_bias_mm']:.2f} mm",
        f"gives {qs['int8']['xy_median_mm']:.3f} mm and bias {qs['int8']['radial_bias_mm']:.2f} mm",
        f"keypoint conv, {qk['int8']['xy_median_mm']:.3f} mm",
        f"{q0['latency_ms_p50_1t']['fp32']:.3f} → {q0['latency_ms_p50_1t']['int8']:.3f} ms"]
check("INT8 key point", all(x in text for x in int8) and q0["fp32"] == load("hard_softargmax", "onnx_report.json")["onnx"]
      and qs["fp32_nodes"] > 0 and qk["fp32_nodes"] > qs["fp32_nodes"], f"missing: {[x for x in int8 if x not in text]}")

print(f"\n{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all passed'}")
raise SystemExit(1 if FAILED else 0)
