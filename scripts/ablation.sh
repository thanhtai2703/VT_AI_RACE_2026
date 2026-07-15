#!/usr/bin/env bash
# scripts/ablation.sh — thử các cấu hình trên 1 scene public để CHỌN cấu hình thắng
# TRƯỚC KHI train private 1 lần cuối. Dùng local_eval (PSNR_max=50 khớp BTC).
#
# Dùng:
#   bash scripts/ablation.sh hcm0031 30000
#
# Chạy 4 cấu hình, mỗi cái train + render + eval, in bảng Score cuối cùng:
#   A_baseline — stock (chỉ --antialiasing)
#   B_densify  — densify tuning aerial
#   C_mip      — Mip-Splatting 3D filter
#   D_both     — densify + mip
#
# -> Chọn cấu hình Score cao nhất (ưu tiên LPIPS thấp), dùng cho run_all.sh private
#    (nhớ set env tương ứng: DENSIFY_GRAD/DENSIFY_UNTIL và/hoặc MIP=1).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GS="$ROOT/gaussian-splatting"
PY="$(command -v python3 || command -v python)"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1   # progress bar tqdm hiện real-time khi qua pipe tee

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
    # Stream progress bar ra màn hình + lưu log đầy đủ vào file (chẩn đoán nếu lỗi).
    # KHÔNG dùng --quiet để thấy cả test PSNR/L1 ở mốc test_iterations.
    ( time "$PY" "$GS/train.py" -s "$SRC/train" -m "$model" --antialiasing \
        --iterations "$ITER" --save_iterations "$ITER" --test_iterations "$ITER" \
        $train_extra --disable_viewer ) 2>&1 | tee "$ROOT/outputs/ablation_${SCENE}_${tag}_train.log"
  fi
  "$PY" "$ROOT/comp/render_test_poses.py" -m "$model" --poses "$CSV" \
      --out "$out" --iteration "$ITER" --antialiasing $render_extra 2>&1 | tail -1
  echo -n "[$tag] " | tee -a "$RESULTS"
  "$PY" "$ROOT/comp/local_eval.py" --renders "$out" --gt "$GT" --psnr_max 50 \
      2>/dev/null | tee -a "$RESULTS"
}

# 4 cấu hình nâng cấp backbone (mặc định 15k để quyết nhanh):
#   A_baseline — stock (chỉ --antialiasing), mốc tham chiếu.
#   B_densify  — densify tuning aerial (grad 0.00015 + until 20000).
#   C_mip      — Mip-Splatting 3D filter (--mip_filter train + render).
#   D_both     — densify tuning + mip filter (kỳ vọng mạnh nhất).
# ⚠️ Config có mip: RENDER phải kèm --mip_filter (filter_3D nằm trong PLY, cờ phải khớp).
# ⚠️ densify_until phải ≤ ITER; ở 15k thì dùng 12000 để còn giai đoạn tinh chỉnh sau densify.
DENSIFY_TUNE="--densify_grad_threshold 0.00015 --densify_until_iter $(( ITER > 20000 ? 20000 : (ITER * 4 / 5) ))"

run_cfg "A_baseline" "" ""
run_cfg "B_densify"  "$DENSIFY_TUNE" ""
run_cfg "C_mip"      "--mip_filter" "--mip_filter"
run_cfg "D_both"     "$DENSIFY_TUNE --mip_filter" "--mip_filter"

echo ""
echo "======================================================"
echo "=== BẢNG SO SÁNH SCORE ($SCENE, $ITER iter) ==="
cat "$RESULTS"
echo "======================================================"
echo "-> Chọn config Score cao nhất, dùng cho run_all.sh private."
