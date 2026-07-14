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
ITER="${2:-15000}"   # mặc định 15k để quyết nhanh (đủ để so tương đối)
SRC="$ROOT/phase1/public_set/$SCENE"
CSV="$SRC/test/test_poses.csv"
GT="$SRC/test/images"
EXP_ARGS="--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp"

mkdir -p "$ROOT/outputs"
RESULTS="$ROOT/outputs/ablation_${SCENE}.txt"
: > "$RESULTS"

run_cfg() {
  local tag="$1"; local train_extra="$2"; local render_extra="$3"
  local model="$ROOT/outputs/ablation/${SCENE}_${tag}"
  local out="$ROOT/outputs/ablation_renders/${SCENE}_${tag}"
  echo ""
  echo "########## CONFIG $tag ##########"
  # Resume: nếu config này đã train xong (có model iteration cuối) -> bỏ qua train.
  if [ -f "$model/point_cloud/iteration_${ITER}/point_cloud.ply" ]; then
    echo "[$tag] đã có model iteration_${ITER} -> bỏ qua train, chỉ render+eval."
  else
    rm -rf "$model"
    ( time "$PY" "$GS/train.py" -s "$SRC/train" -m "$model" --antialiasing \
        --iterations "$ITER" --save_iterations "$ITER" --test_iterations "$ITER" \
        $train_extra --disable_viewer --quiet ) 2>&1 | tail -3
  fi
  "$PY" "$ROOT/comp/render_test_poses.py" -m "$model" --poses "$CSV" \
      --out "$out" --iteration "$ITER" --antialiasing $render_extra 2>&1 | tail -1
  echo -n "[$tag] " | tee -a "$RESULTS"
  "$PY" "$ROOT/comp/local_eval.py" --renders "$out" --gt "$GT" --psnr_max 50 \
      2>/dev/null | tee -a "$RESULTS"
}

# Chỉ 2 cấu hình để quyết nhanh (mặc định 15k):
#   A) baseline (chỉ --antialiasing) — mốc tham chiếu
#   D) + depth + exposure — cấu hình "full" mạnh nhất
# So 2 cái: full thắng baseline -> dùng full cho private; ngược lại -> baseline.
# LƯU Ý exposure: TRAIN có $EXP_ARGS (--train_test_exp) nhưng RENDER test pose
# KHÔNG áp exposure (render_extra = ""). Vì exposure học per-training-image;
# test pose không nằm trong tập train -> get_exposure_from_name KeyError.
# Lợi ích exposure nằm ở MODEL (train sạch hơn), không ở lúc render test.
run_cfg "A_baseline" "" ""

if [ -d "$SRC/train/depths" ] && [ -f "$SRC/train/sparse/0/depth_params.json" ]; then
  run_cfg "D_depth_exp" "-d $SRC/train/depths $EXP_ARGS" ""
else
  echo "[ablation] ⚠️ Chưa có depth cho $SCENE -> chỉ chạy baseline + exposure (bỏ depth)."
  run_cfg "B_exposure" "$EXP_ARGS" ""
fi

echo ""
echo "======================================================"
echo "=== BẢNG SO SÁNH SCORE ($SCENE, $ITER iter) ==="
cat "$RESULTS"
echo "======================================================"
echo "-> Chọn config Score cao nhất, dùng cho run_all.sh private."
