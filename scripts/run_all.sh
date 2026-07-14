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

# --- Chống OOM (24GB VRAM) ---
# expandable_segments gom lại VRAM bị phân mảnh -> AN TOÀN, KHÔNG đổi chất lượng.
# Mặc định KHÔNG ép densify (giữ nguyên chất lượng như các scene đã train).
# Chỉ khi 1 scene vẫn OOM thì set env, vd:
#   DENSIFY_GRAD=0.0004 bash scripts/run_all.sh phase1/private_set1 30000
#   (hoặc thêm EXTRA_TRAIN="--data_device cpu")
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DENSIFY_GRAD="${DENSIFY_GRAD:-}"                  # rỗng = dùng mặc định repo (0.0002)
EXTRA_TRAIN="${EXTRA_TRAIN:-}"                    # vd "--data_device cpu" nếu vẫn OOM

# --- Cải tiến chất lượng (bật qua env) ---
# DEPTH=1     -> dùng depth reg, thêm '-d <scene>/train/depths' (phải chạy gen_depth.sh trước)
# EXPOSURE=1  -> bật exposure compensation (data drone ngoài trời)
DEPTH="${DEPTH:-0}"
EXPOSURE="${EXPOSURE:-0}"
EXPOSURE_ARGS="--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp"

# --- Resume: bỏ qua scene đã train xong (đã có point_cloud của iteration cuối) ---
RESUME="${RESUME:-1}"

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

  # 1) TRAIN (resume: bỏ qua nếu ĐÃ có model ở BẤT KỲ iteration nào)
  # Dùng iteration lớn nhất hiện có để render (không bắt buộc đúng $ITER) -> giữ lại
  # các scene đã train ở iter khác (vd HCM0249 đã xong 30k) khi loạt này chạy 15k.
  # '|| true' để set -e + pipefail KHÔNG giết script khi scene chưa có model
  # (ls fail vì không có thư mục iteration_* -> pipeline trả exit != 0).
  existing_iter="$(ls -d "$model"/point_cloud/iteration_* 2>/dev/null \
                    | sed 's/.*iteration_//' | sort -n | tail -1 || true)"
  if [[ "$RESUME" -eq 1 && -n "$existing_iter" \
        && -f "$model/point_cloud/iteration_${existing_iter}/point_cloud.ply" ]]; then
    echo "[$scene] ĐÃ TRAIN xong (iteration_${existing_iter}) -> bỏ qua train."
    RENDER_ITER="$existing_iter"
  else
    # Chỉ thêm --densify_grad_threshold khi DENSIFY_GRAD được set (giữ mặc định nếu rỗng).
    # Dùng if (không dùng '&&') để tránh set -e giết script khi DENSIFY_GRAD rỗng.
    DENSIFY_ARG=""
    if [ -n "$DENSIFY_GRAD" ]; then DENSIFY_ARG="--densify_grad_threshold $DENSIFY_GRAD"; fi

    # Depth reg: chỉ bật khi DEPTH=1 VÀ có thư mục depths + depth_params.json.
    DEPTH_ARG=""
    if [ "$DEPTH" -eq 1 ]; then
      if [ -d "$src/depths" ] && [ -f "$src/sparse/0/depth_params.json" ]; then
        DEPTH_ARG="-d $src/depths"
      else
        echo "[$scene] ⚠️ DEPTH=1 nhưng thiếu depths/ hoặc depth_params.json -> BỎ depth reg. Chạy gen_depth.sh trước."
      fi
    fi

    # Exposure compensation.
    EXP_ARG=""
    if [ "$EXPOSURE" -eq 1 ]; then EXP_ARG="$EXPOSURE_ARGS"; fi

    echo "[$scene] TRAIN ($ITER iter)... [densify='${DENSIFY_GRAD:-default}' depth=$DEPTH exposure=$EXPOSURE extra='$EXTRA_TRAIN']"
    rm -rf "$model"   # xoá model dở (nếu có) để train sạch
    ( time "$PY" "$GS/train.py" \
        -s "$src" \
        -m "$model" \
        $AA_FLAG \
        --iterations "$ITER" \
        --save_iterations "$ITER" \
        --test_iterations "$ITER" \
        $DENSIFY_ARG \
        $DEPTH_ARG \
        $EXP_ARG \
        $EXTRA_TRAIN \
        --disable_viewer \
        --quiet \
    ) 2>&1 | tee "$LOGS/${scene}_train.log"
    RENDER_ITER="$ITER"
  fi

  # 2) RENDER test poses theo CSV (dùng đúng iteration của model)
  # Nếu train có exposure -> render phải khớp --train_test_exp.
  RENDER_EXP_ARG=""
  if [ "$EXPOSURE" -eq 1 ]; then RENDER_EXP_ARG="--train_test_exp"; fi
  echo "[$scene] RENDER test poses (iteration ${RENDER_ITER})..."
  "$PY" "$ROOT/comp/render_test_poses.py" \
    -m "$model" \
    --poses "$csv" \
    --out "$out" \
    --iteration "$RENDER_ITER" \
    $AA_FLAG \
    $RENDER_EXP_ARG \
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
