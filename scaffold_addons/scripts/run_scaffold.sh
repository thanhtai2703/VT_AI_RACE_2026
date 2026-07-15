#!/usr/bin/env bash
# scaffold_addons/scripts/run_scaffold.sh
#
# Loop mọi scene: train Scaffold-GS -> render test poses theo CSV -> (public) eval.
# CHẠY TRONG thư mục Scaffold-GS/ đã clone (có train.py + render_test_poses_scaffold.py).
#
# Tiền đề (làm 1 lần):
#   1) Đã build submodule Scaffold, import OK.
#   2) Đã vá scene/dataset_readers.py theo PATCH_dataset_readers.md (2 fix bắt buộc).
#   3) Đã copy render_test_poses_scaffold.py vào thư mục Scaffold-GS/.
#   4) Đã chạy setup_symlinks.sh cho set tương ứng -> có data/<scene>/{images,sparse}.
#
# Dùng:
#   REPO=/path/VT_AI_RACE_2026 bash <path>/run_scaffold.sh public_set   15000   # ablation/public
#   REPO=/path/VT_AI_RACE_2026 bash <path>/run_scaffold.sh private_set1 30000   # bài nộp
#
# REPO = đường dẫn repo chính (để lấy test_poses.csv, GT public, và comp/local_eval.py).
#
set -euo pipefail

SET_NAME="${1:?Thiếu set: public_set hoặc private_set1}"
ITER="${2:-30000}"
REPO="${REPO:?Thiếu env REPO=đường dẫn repo chính VT_AI_RACE_2026}"
REPO="$(cd "$REPO" && pwd)"
PY="$(command -v python3 || command -v python)"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[ -f "./train.py" ] || { echo "❌ Chạy TRONG thư mục Scaffold-GS/"; exit 1; }
[ -f "./render_test_poses_scaffold.py" ] || { echo "❌ Thiếu render_test_poses_scaffold.py trong thư mục này"; exit 1; }

# --- Config aerial (điều chỉnh cho ảnh drone ngoài trời cảnh lớn) ---
# voxel_size=0.001 (mặc định Scaffold, hợp cảnh cỡ metric). Đặt 0 để tự 1-NN median nếu scale lạ.
# update_init_factor=16 (mặc định). appearance_dim=32: BẬT appearance embedding vì ảnh drone
#   ngoài trời đổi phơi sáng giữa các frame (Scaffold khuyến nghị cho "wild scenes"). ratio=1.
VOXEL_SIZE="${VOXEL_SIZE:-0.001}"
UPDATE_INIT_FACTOR="${UPDATE_INIT_FACTOR:-16}"
APPEARANCE_DIM="${APPEARANCE_DIM:-32}"
RATIO="${RATIO:-1}"

SRC_SET="$REPO/phase1/$SET_NAME"
IS_PUBLIC=0; case "$SET_NAME" in *public*) IS_PUBLIC=1;; esac
OUT_RENDERS="$REPO/outputs/scaffold_renders"
mkdir -p "$OUT_RENDERS"

echo "=== run_scaffold: SET=$SET_NAME ITER=$ITER public=$IS_PUBLIC voxel=$VOXEL_SIZE app_dim=$APPEARANCE_DIM ==="

for scene_dir in "$SRC_SET"/*/; do
  scene="$(basename "$scene_dir")"
  csv="$scene_dir/test/test_poses.csv"
  data="data/$scene"
  model="outputs/$scene"
  out="$OUT_RENDERS/$scene"

  if [[ ! -d "$data/images" || ! -f "$csv" ]]; then
    echo "[$scene] BỎ QUA (thiếu data/$scene/images hoặc test_poses.csv). Chạy setup_symlinks.sh chưa?"; continue
  fi

  echo ""
  echo "########## SCENE: $scene ##########"

  # 1) TRAIN. KHÔNG --eval: train TOÀN BỘ ảnh (render test qua CSV riêng, như pipeline Inria của ta).
  #    Resume: bỏ qua nếu đã có point_cloud iteration cuối.
  if [ -f "$model/point_cloud/iteration_${ITER}/point_cloud.ply" ]; then
    echo "[$scene] đã train xong iteration_${ITER} -> bỏ qua train."
  else
    rm -rf "$model"
    echo "[$scene] TRAIN Scaffold ($ITER iter)..."
    "$PY" train.py -s "$data" -m "$model" \
        --iterations "$ITER" \
        --voxel_size "$VOXEL_SIZE" \
        --update_init_factor "$UPDATE_INIT_FACTOR" \
        --appearance_dim "$APPEARANCE_DIM" \
        --ratio "$RATIO" \
        --save_iterations "$ITER"
    # ⚠️ NẾU train.py báo lỗi thiếu ảnh test do bỏ --eval, xem ghi chú cuối file.
  fi

  # 2) RENDER test poses theo CSV.
  echo "[$scene] RENDER test poses (iteration ${ITER})..."
  "$PY" render_test_poses_scaffold.py \
      -m "$model" \
      --poses "$csv" \
      --out "$out" \
      --iteration "$ITER"

  # 3) EVAL (chỉ public — có GT).
  if [[ "$IS_PUBLIC" -eq 1 ]]; then
    gt="$scene_dir/test/images"
    echo "[$scene] EVAL (Score cục bộ, psnr_max=50)..."
    "$PY" "$REPO/comp/local_eval.py" --renders "$out" --gt "$gt" --psnr_max 50
  fi
done

echo ""
echo "=== XONG. Render ở $OUT_RENDERS ==="
echo "Đóng gói (từ repo chính):"
echo "  $PY comp/make_submission.py --renders outputs/scaffold_renders --out submission/scaffold.zip --format jpg --quality 95 --ext JPG"
echo "  $PY comp/validate_submission.py --zip submission/scaffold.zip --data phase1/$SET_NAME"
echo ""
echo "GHI CHÚ: nếu bỏ --eval khiến train.py lỗi (một số bản Scaffold cần --eval để chạy),"
echo "  THÊM --eval vào lệnh train — nó chỉ đổi split train/test NỘI BỘ của Scaffold, KHÔNG"
echo "  ảnh hưởng render CSV của ta (ta render qua render_test_poses_scaffold.py, không dùng"
echo "  getTestCameras của Scaffold). Model vẫn train trên phần lớn ảnh."
