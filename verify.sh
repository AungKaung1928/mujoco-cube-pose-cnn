#!/usr/bin/env bash
# Reproduce the README's claims from a clean checkout, cheapest tier first.
#
# Tier 1 needs nothing but the venv: the yaw fold, the projection and the metric
# are checked against values derived on paper. If any of these is wrong, every
# number in the README is consistently wrong, and nothing else would notice.
# Tiers 2-3 need data/ (gitignored, ~1.3 GB, ~30 s to regenerate) and re-score
# the tracked checkpoints, so the CNN rows can be checked without retraining.
set -u
cd "$(dirname "$0")"
source ~/personal/ml/env.sh

hr() { printf '\n=== %s ===\n' "$1"; }
have_data() { [ -f "data/hard/val_images.npy" ]; }

hr "1/3  conventions -- hand-computed cases"
python test_common.py || exit 1

if ! have_data; then
  hr "2-3/3  skipped -- no dataset"
  cat <<'MSG'
data/ is gitignored. To regenerate it (~30 s total, single-threaded render):

    python gen_dataset.py --regime easy --n 12000
    python gen_dataset.py --regime hard --n 12000
MSG
  exit 0
fi

hr "2/3  classical baselines on val (v1 fixed priors, v2 calibrated on train)"
python baseline_cv.py --regime hard --method red || exit 1
python baseline_cv.py --regime hard --method sat || exit 1
python baseline_v2.py --regime hard --method sat || exit 1

hr "3/3  CNN rows -- re-score the tracked checkpoint through PyTorch and ONNX Runtime"
# Rewrites runs/hard_softargmax/model.onnx from the tracked final.pt and prints the
# task metrics twice (torch, onnxruntime); both must match the README table.
python export_onnx.py --regime hard --head softargmax || exit 1
echo
echo "Latency rows are only comparable on an idle box: check 'uptime' first."
