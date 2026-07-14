#!/usr/bin/env bash
# scripts/ablation.sh — thử các cấu hình trên 1 scene public để CHỌN cấu hình thắng
# TRƯỚC KHI train private 1 lần cuối. Dùng local_eval (PSNR_max=50 khớp BTC).
#
# Dùng:
#   bash scripts/ablation.sh hcm0031 30000
#
# Chạy 4 cấu hình, mỗi cái train + render + eval, in bảng Score cuối cùng:
#   A) baseline (chỉ --antialiasing)
#   B) + exposure
#   C) + depth        (cần chạy gen_depth.sh cho scene này trước)
#   D) + depth + exposure
#
# -> Chọn cấu hình Score cao nhất, dùng cho run_all.sh private.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GS="$ROOT/gaussian-splatting"
PY="$(command -v python3 || command -v python)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

SCENE="${1:?Thiếu tên scene public, vd hcm0031}"
ITER="${2:-30000}"
SRC="$ROOT/phase1/public_set/$SCENE"
CSV="$SRC/test/test_poses.csv"
GT="$SRC/test/images"
EXP_ARGS="--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp"

RESULTS="$ROOT/outputs/ablation_${SCENE}.txt"
: > "$RESULTS"

run_cfg() {
  local tag="$1"; local train_extra="$2"; local render_extra="$3"
  local model="$ROOT/outputs/ablation/${SCENE}_${tag}"
  local out="$ROOT/outputs/ablation_renders/${SCENE}_${tag}"
  echo ""
  echo "########## CONFIG $tag ##########"
  rm -rf "$model"
  ( time "$PY" "$GS/train.py" -s "$SRC/train" -m "$model" --antialiasing \
      --iterations "$ITER" --save_iterations "$ITER" --test_iterations "$ITER" \
      $train_extra --disable_viewer --quiet ) 2>&1 | tail -3
  "$PY" "$ROOT/comp/render_test_poses.py" -m "$model" --poses "$CSV" \
      --out "$out" --iteration "$ITER" --antialiasing $render_extra 2>&1 | tail -1
  echo -n "[$tag] " | tee -a "$RESULTS"
  "$PY" "$ROOT/comp/local_eval.py" --renders "$out" --gt "$GT" --psnr_max 50 \
      2>/dev/null | tee -a "$RESULTS"
}

# A) baseline
run_cfg "A_baseline" "" ""
# B) + exposure
run_cfg "B_exposure" "$EXP_ARGS" "--train_test_exp"
# C) + depth (nếu đã sinh depth)
if [ -d "$SRC/train/depths" ] && [ -f "$SRC/train/sparse/0/depth_params.json" ]; then
  run_cfg "C_depth" "-d $SRC/train/depths" ""
  run_cfg "D_depth_exp" "-d $SRC/train/depths $EXP_ARGS" "--train_test_exp"
else
  echo "[ablation] ⚠️ Chưa có depth cho $SCENE -> bỏ config C,D. Chạy: bash scripts/gen_depth.sh $SRC"
fi

echo ""
echo "======================================================"
echo "=== BẢNG SO SÁNH SCORE ($SCENE, $ITER iter) ==="
cat "$RESULTS"
echo "======================================================"
echo "-> Chọn config Score cao nhất, dùng cho run_all.sh private."
