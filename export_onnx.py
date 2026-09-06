"""Export the trained regressor to ONNX and prove the exported graph is the model.

An export is not done when the file appears. It is done when the ONNX runtime
reproduces the PyTorch outputs to float tolerance AND reproduces the *task*
metrics, because a graph can be numerically close and still have a wrong output
order, a wrong normalisation, or a silently dropped layer.

Latency is measured at batch 1, which is the only batch size a real perception
node ever sees, and at both 1 and 8 threads -- a ROS 2 node does not get the
whole CPU.
"""
import argparse, json, os, time
import numpy as np
import torch

import common as C
from model import PoseNet
from train import to_batch, evaluate

ROOT = os.path.dirname(os.path.abspath(__file__))


def decode(o):
    """raw network output -> (xy metres, yaw radians). Same rule as training."""
    xy = o[:, :2] * C.XY_RANGE
    v = o[:, 2:]
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
    return xy, C.vec_to_yaw(v)


def main(a):
    import onnxruntime as ort

    run = os.path.join(ROOT, "runs", f"{a.regime}_{a.head}")
    net = PoseNet(a.head)
    net.load_state_dict(torch.load(os.path.join(run, "final.pt"), map_location="cpu"))
    net.eval()

    va_i, va_l, meta = C.load_split(ROOT, a.regime, "val")
    path = os.path.join(run, "model.onnx")

    dummy, _ = to_batch(va_i, np.array([0]))
    torch.onnx.export(
        net, dummy, path, opset_version=17, dynamo=False,
        input_names=["image"], output_names=["pose"],
        dynamic_axes={"image": {0: "batch"}, "pose": {0: "batch"}},
    )
    print(f"exported            {path}  ({os.path.getsize(path)/1024:.0f} KB, "
          f"{net.n_params():,} params)")

    # 1. numerical agreement on a real batch
    x, idx = to_batch(va_i, np.arange(256))
    with torch.no_grad():
        ref = net(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    got = sess.run(["pose"], {"image": x.numpy()})[0]
    dmax = float(np.abs(ref - got).max())
    print(f"max |torch - onnx|  {dmax:.3e}   {'OK' if dmax < 1e-4 else 'MISMATCH'}")

    # 2. task-level agreement on the whole split, decoded the same way
    n = len(va_i)
    pxy = np.empty((n, 2)); pyaw = np.empty(n)
    for s in range(0, n, 256):
        i = np.arange(s, min(s + 256, n))
        xb, _ = to_batch(va_i, i)
        pxy[i], pyaw[i] = decode(sess.run(["pose"], {"image": xb.numpy()})[0].astype(np.float64))
    m_onnx = C.pose_metrics(pxy, pyaw, va_l[:, :2], va_l[:, 4])
    m_torch = evaluate(net, va_i, va_l)

    # 3. batch-1 latency at 1 and 8 threads
    lat = {}
    for th in (1, 8):
        so = ort.SessionOptions()
        so.intra_op_num_threads = th
        so.inter_op_num_threads = 1
        s1 = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        feed = {"image": dummy.numpy()}
        for _ in range(20):
            s1.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(300):
            s1.run(None, feed)
        lat[th] = (time.perf_counter() - t0) / 300 * 1e3

    C.print_metrics(m_onnx, n, n, meta["px_per_m"], lat[1],
                    header=f"onnxruntime  {a.regime}/{a.head}  n={n}")
    print(f"  torch, same split   xy median {m_torch['xy_median_mm']:.2f} mm   "
          f"yaw median {m_torch['yaw_median_deg']:.2f} deg   <- must match the rows above")
    print(f"  latency 1 thread    {lat[1]:7.2f} ms/img")
    print(f"  latency 8 threads   {lat[8]:7.2f} ms/img")

    json.dump({"onnx": m_onnx, "torch": m_torch, "max_abs_diff": dmax,
               "latency_ms": lat}, open(os.path.join(run, "onnx_report.json"), "w"), indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="hard", choices=C.REGIMES)
    p.add_argument("--head", default="softargmax", choices=["flat", "softargmax"])
    main(p.parse_args())
