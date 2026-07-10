#!/usr/bin/env bash
# scripts/run_all.sh — loop qua mọi scene: train -> render test poses -> (tùy) eval.
#
# Dùng:
#   bash scripts/run_all.sh phase1/public_set   [ITER]   # public: có gt -> tự eval
#   bash scripts/run_all.sh phase1/private_set1 [ITER]   # private: chỉ render + zip
#
# ITER mặc định 30000. Đặt ITER=7000 để chạy nhanh sanity ban đầu.
#
# Quyết định đã chốt: train pinhole trực tiếp, --antialiasing, KHÔNG --eval.
# LƯU Ý: bản diff-gaussian-rasterization đang dùng KHÔNG có SparseGaussianAdam
# (repo có try/except -> tự fallback). Nên KHÔNG truyền --optimizer_type sparse_adam.

set -euo pipefail

DATA_ROOT="${1:?Thiếu DATA_ROOT, vd phase1/public_set}"
ITER="${2:-30000}"
AA_FLAG="--antialiasing"

# Tự dò python: máy thuê thường chỉ có python3, máy local có python.
PY="$(command -v python3 || command -v python)"
[ -z "$PY" ] && { echo "❌ Không tìm thấy python3/python"; exit 1; }
echo "[run_all] dùng PY=$PY"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GS="$ROOT/gaussian-splatting"
MODELS="$ROOT/outputs/models"
RENDERS="$ROOT/outputs/renders"
LOGS="$ROOT/outputs/logs"
mkdir -p "$MODELS" "$RENDERS" "$LOGS"

# public_set -> có ground-truth để eval; private -> không.
IS_PUBLIC=0
case "$DATA_ROOT" in *public*) IS_PUBLIC=1;; esac

echo "=== run_all: DATA_ROOT=$DATA_ROOT ITER=$ITER (public=$IS_PUBLIC) ==="

for scene_dir in "$DATA_ROOT"/*/; do
  scene="$(basename "$scene_dir")"
  src="$scene_dir/train"
  csv="$scene_dir/test/test_poses.csv"
  model="$MODELS/$scene"
  out="$RENDERS/$scene"

  if [[ ! -d "$src/images" || ! -f "$csv" ]]; then
    echo "[$scene] BỎ QUA (thiếu train/images hoặc test_poses.csv)"; continue
  fi

  echo ""
  echo "########## SCENE: $scene ##########"

  # 1) TRAIN
  echo "[$scene] TRAIN ($ITER iter)..."
  ( time "$PY" "$GS/train.py" \
      -s "$src" \
      -m "$model" \
      $AA_FLAG \
      --iterations "$ITER" \
      --save_iterations "$ITER" \
      --test_iterations "$ITER" \
      --disable_viewer \
      --quiet \
  ) 2>&1 | tee "$LOGS/${scene}_train.log"

  # 2) RENDER test poses theo CSV
  echo "[$scene] RENDER test poses..."
  "$PY" "$ROOT/comp/render_test_poses.py" \
    -m "$model" \
    --poses "$csv" \
    --out "$out" \
    --iteration "$ITER" \
    $AA_FLAG \
    2>&1 | tee "$LOGS/${scene}_render.log"

  # 3) EVAL (chỉ public)
  if [[ "$IS_PUBLIC" -eq 1 ]]; then
    echo "[$scene] EVAL (Score cục bộ)..."
    "$PY" "$ROOT/comp/local_eval.py" \
      --renders "$out" \
      --gt "$scene_dir/test/images" \
      2>&1 | tee "$LOGS/${scene}_eval.log"
  fi
done

echo ""
echo "=== XONG. Render ở $RENDERS ==="
echo "Đóng gói:  $PY comp/make_submission.py --renders outputs/renders --out submission/submission.zip"
echo "Validate:  $PY comp/validate_submission.py --zip submission/submission.zip --data $DATA_ROOT"
