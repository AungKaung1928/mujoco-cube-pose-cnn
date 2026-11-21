"""Static INT8 (QDQ, MinMax, per-channel weights) of the exported pose regressor,
scored with the same metric as training. One thread throughout.

    python quantize_int8.py            # hard_softargmax -> runs/hard_softargmax/int8_report.json
    python quantize_int8.py --fp32 /head/head.0/ --tag _fp32softargmax
    python quantize_int8.py --fp32 /head/head.0/ /kp/ --tag _fp32kp
    python quantize_int8.py --fp32 /head/head.0/Softmax --tag _fp32smonly
    python quantize_int8.py --fp32 /head/head.0/Mul /head/head.0/ReduceSum --tag _fp32expect
    python quantize_int8.py --act16 --tag _a16w8   # 16-bit activations, INT8 weights

The question is whether a sub-millimetre regressor survives INT8 the way a detector
does: the soft-argmax reads position off a spatial softmax, so quantised logits move
the expected coordinate directly. Calibration: 200 TRAIN images; scored on all of val.
Latency: model only, batch 1, fp32 and INT8 interleaved over 5 rounds.
"""
import argparse
import json
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                      QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

import common as C
from export_onnx import decode
from train import to_batch

ROOT = os.path.dirname(os.path.abspath(__file__))


class Calib(CalibrationDataReader):
    def __init__(self, imgs, n):
        self.it = iter(to_batch(imgs, np.array([i]))[0].numpy() for i in range(n))

    def get_next(self):
        x = next(self.it, None)
        return None if x is None else {"image": x}


def sess(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


def score(s, imgs, labels, bs=256):
    pxy, pyaw = np.empty((len(imgs), 2)), np.empty(len(imgs))
    for b in range(0, len(imgs), bs):
        i = np.arange(b, min(b + bs, len(imgs)))
        x, _ = to_batch(imgs, i)
        pxy[i], pyaw[i] = decode(s.run(["pose"], {"image": x.numpy()})[0].astype(np.float64))
    return C.pose_metrics(pxy, pyaw, labels[:, :2], labels[:, 4])


def main(a):
    run = os.path.join(ROOT, "runs", a.run)
    fp32 = os.path.join(run, "model.onnx")
    pre, q8 = os.path.join(run, "model_pre.onnx"), os.path.join(run, f"model_int8{a.tag}.onnx")
    tr_i = C.load_split(ROOT, a.regime, "train")[0]
    va_i, va_l = C.load_split(ROOT, a.regime, "val")[:2]
    quant_pre_process(fp32, pre, skip_symbolic_shape=False)
    keep = [n.name for n in onnx.load(pre).graph.node if any(n.name.startswith(p) for p in a.fp32)]
    quantize_static(pre, q8, Calib(tr_i, a.calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt16 if a.act16 else QuantType.QUInt8, weight_type=QuantType.QInt8,
                    per_channel=True, calibrate_method=CalibrationMethod.MinMax, nodes_to_exclude=keep)
    os.remove(pre)
    s32, s8 = sess(fp32), sess(q8)
    m32, m8 = score(s32, va_i, va_l), score(s8, va_i, va_l)
    xs = [to_batch(va_i, np.array([i]))[0].numpy() for i in range(a.n_time)]
    t = {"fp32": [], "int8": []}
    for r in range(a.rounds):
        for k, s in (((("fp32", s32), ("int8", s8))) if r % 2 == 0 else (("int8", s8), ("fp32", s32))):
            for x in xs[:20]:
                s.run(None, {"image": x})
            ts = []
            for x in xs:
                t0 = time.perf_counter()
                s.run(None, {"image": x})
                ts.append(time.perf_counter() - t0)
            t[k].append(float(np.median(ts)) * 1e3)
    out = {"run": a.run, "ort": ort.__version__, "fp32_prefixes": a.fp32, "fp32_nodes": len(keep), "activations": "uint16" if a.act16 else "uint8", "calib_images": a.calib, "calib_split": "train",
           "val_images": len(va_i), "bytes": {"fp32": os.path.getsize(fp32), "int8": os.path.getsize(q8)},
           "fp32": m32, "int8": m8,
           "latency_ms_p50_1t": {k: round(float(np.median(v)), 4) for k, v in t.items()},
           "rounds": a.rounds, "n_time": a.n_time,
           "load1": float(open("/proc/loadavg").read().split()[0])}
    print(json.dumps(out, indent=1))
    with open(os.path.join(run, f"int8_report{a.tag}.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="hard_softargmax")
    ap.add_argument("--regime", default="hard")
    ap.add_argument("--calib", type=int, default=200)
    ap.add_argument("--n-time", type=int, default=500)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--fp32", nargs="*", default=[], help="node-name prefixes left in fp32, e.g. /head/head.0/ /kp/")
    ap.add_argument("--act16", action="store_true", help="16-bit activations (A16W8) instead of uint8")
    ap.add_argument("--tag", default="", help="suffix for the report and graph file names")
    main(ap.parse_args())
