"""Train the cube-pose regressor on CPU.

Targets are normalised before the loss so that all four outputs live on the same
scale: xy is divided by XY_RANGE (-> [-1, 1]) and (sin 4t, cos 4t) is already in
[-1, 1]. Without that, 0.075 m of position error and 1.0 of sine error would be
weighted a thousand to one and the model would only learn yaw.

No augmentation here on purpose. Augmentation is block 2. This run measures what
the plain architecture does on the plain data, so that the augmentation result
later has something honest to be compared against.

Throughput is printed every epoch. On this machine there is no CPU temperature
sensor, so a sustained drop of more than 20% from the first epoch's rate is the
only available throttling signal, and the run says so out loud when it happens.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

import common as C
from model import PoseNet

ROOT = os.path.dirname(os.path.abspath(__file__))


def to_batch(imgs, idx):
    """uint8 (N,H,W,3) memmap + indices -> normalised float tensor (B,3,H,W)."""
    idx = np.sort(idx)                       # sequential reads from the memmap
    a = np.ascontiguousarray(imgs[idx])
    t = torch.from_numpy(a).permute(0, 3, 1, 2).float()
    return t.div_(127.5).sub_(1.0), idx


def targets(labels, idx):
    y = labels[idx]
    out = np.empty((len(idx), 4), np.float32)
    out[:, :2] = y[:, :2] / C.XY_RANGE
    out[:, 2:] = y[:, 2:4]
    return torch.from_numpy(out)


@torch.no_grad()
def evaluate(net, imgs, labels, bs=256):
    net.eval()
    n = len(imgs)
    pxy = np.empty((n, 2), np.float64)
    pyaw = np.empty(n, np.float64)
    for s in range(0, n, bs):
        idx = np.arange(s, min(s + bs, n))
        x, _ = to_batch(imgs, idx)
        o = net(x).numpy().astype(np.float64)
        pxy[idx] = o[:, :2] * C.XY_RANGE
        v = o[:, 2:]
        v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-8)
        pyaw[idx] = C.vec_to_yaw(v)
    net.train()
    return C.pose_metrics(pxy, pyaw, labels[:, :2], labels[:, 4])


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))

    tr_i, tr_l, meta = C.load_split(ROOT, a.regime, "train")
    va_i, va_l, _ = C.load_split(ROOT, a.regime, "val")
    if a.limit:
        tr_i, tr_l = tr_i[:a.limit], tr_l[:a.limit]
        va_i, va_l = va_i[:max(200, a.limit // 4)], va_l[:max(200, a.limit // 4)]
    n = len(tr_i)

    net = PoseNet(a.head)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.SmoothL1Loss(beta=0.1)

    out = os.path.join(ROOT, "runs", f"{a.regime}_{a.head}")
    os.makedirs(out, exist_ok=True)
    print(f"{a.regime}/{a.head}  train={n}  val={len(va_i)}  params={net.n_params():,}  "
          f"threads={torch.get_num_threads()}")

    best, first_rate = None, None
    for ep in range(1, a.epochs + 1):
        perm = np.random.permutation(n)
        tot, seen, t0 = 0.0, 0, time.perf_counter()
        for s in range(0, n - a.bs + 1, a.bs):
            idx = perm[s:s + a.bs]
            x, idx = to_batch(tr_i, idx)
            y = targets(tr_l, idx)
            loss = lossf(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
            seen += len(idx)
        sched.step()
        rate = seen / (time.perf_counter() - t0)
        m = evaluate(net, va_i, va_l)

        if first_rate is None:
            first_rate = rate
        drop = 100 * (1 - rate / first_rate)
        flag = f"   THROTTLE? -{drop:.0f}% vs epoch 1" if drop > 20 else ""
        print(f"  ep {ep:3d}  loss {tot/seen:.5f}  "
              f"xy {m['xy_median_mm']:6.2f} mm  yaw {m['yaw_median_deg']:5.2f} deg  "
              f"bias {m['radial_bias_mm']:+5.2f}  {rate:6.1f} img/s{flag}")

        if best is None or m["xy_median_mm"] < best["xy_median_mm"]:
            best = dict(m, epoch=ep)
            torch.save(net.state_dict(), os.path.join(out, "best.pt"))

    # single-image latency, the number that matters for a real perception node
    net.eval()
    x, _ = to_batch(va_i, np.array([0]))
    with torch.no_grad():
        for _ in range(10):
            net(x)
        t0 = time.perf_counter()
        for _ in range(200):
            net(x)
        lat = (time.perf_counter() - t0) / 200 * 1e3

    C.print_metrics(best, len(va_i), len(va_i), meta["px_per_m"], lat,
                    header=f"BEST  {a.regime}/{a.head}  epoch {best['epoch']}")
    json.dump({**best, "latency_ms": lat, "params": net.n_params(),
               "args": vars(a)}, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print(f"  saved               {out}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="hard", choices=C.REGIMES)
    p.add_argument("--head", default="flat", choices=["flat", "softargmax"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--limit", type=int, default=0, help="use only the first N training images")
    p.add_argument("--seed", type=int, default=0)
    main(p.parse_args())
