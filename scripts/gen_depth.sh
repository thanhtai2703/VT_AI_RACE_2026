#!/usr/bin/env bash
# scripts/gen_depth.sh — sinh depth maps (Depth-Anything-V2) cho depth regularization.
#
# Chạy MỘT LẦN trên máy GPU trước khi train với -d. Cho từng scene:
#   1. Sinh inverse-depth map cho mọi ảnh train -> <scene>/train/depths/
#   2. Tạo <scene>/train/sparse/0/depth_params.json (scale/offset khớp COLMAP)
#
# Dùng:
#   bash scripts/gen_depth.sh phase1/public_set/hcm0031      # 1 scene
#   bash scripts/gen_depth.sh phase1/private_set1            # cả tập (mọi scene con)
#
# Sau đó train thêm cờ:  -d <scene>/train/depths
#
# Yêu cầu: đã cài Depth-Anything-V2 (script tự clone + tải weights nếu chưa có).

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GS="$ROOT/gaussian-splatting"
PY="$(command -v python3 || command -v python)"
DAV2="$ROOT/Depth-Anything-V2"
CKPT="$DAV2/checkpoints/depth_anything_v2_vitl.pth"

TARGET="${1:?Thiếu đường dẫn scene hoặc tập, vd phase1/public_set/hcm0031}"

# --- Cài Depth-Anything-V2 nếu chưa có ---
if [[ ! -f "$DAV2/run.py" ]]; then
  echo "[depth] clone Depth-Anything-V2..."
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$DAV2"
fi
if [[ ! -f "$CKPT" ]]; then
  echo "[depth] tải weights vitl (~1.3GB)..."
  mkdir -p "$DAV2/checkpoints"
  wget -O "$CKPT" \
    "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
fi
$PY -m pip install -q opencv-python-headless matplotlib 2>/dev/null || true

# --- Danh sách scene cần xử lý ---
gen_one() {
  local scene_dir="$1"
  local scene="$(basename "$scene_dir")"
  # Đường dẫn TUYỆT ĐỐI — bắt buộc, vì run.py chạy sau khi 'cd' vào $DAV2.
  local imgs="$(cd "$scene_dir/train" && pwd)/images"
  local depths="$(cd "$scene_dir/train" && pwd)/depths"
  local sparse="$(cd "$scene_dir/train" && pwd)"

  if [[ ! -d "$imgs" ]]; then
    echo "[$scene] BỎ QUA (không có train/images)"; return 0
  fi
  echo ""
  echo "########## DEPTH: $scene ##########"

  # 1) Sinh inverse-depth map (grayscale, giữ giá trị). --outdir TUYỆT ĐỐI.
  mkdir -p "$depths"
  ( cd "$DAV2" && $PY run.py --encoder vitl --pred-only --grayscale \
      --img-path "$imgs" --outdir "$depths" )

  # 2) Tạo depth_params.json (khớp scale COLMAP)
  $PY "$GS/utils/make_depth_scale.py" \
      --base_dir "$sparse" \
      --depths_dir "$depths" \
      --model_type bin
  echo "[$scene] depth xong -> $depths + sparse/0/depth_params.json"
}

# Nếu TARGET có train/images -> 1 scene; ngược lại coi là tập nhiều scene.
if [[ -d "$TARGET/train/images" ]]; then
  gen_one "$TARGET"
else
  for scene_dir in "$TARGET"/*/; do
    gen_one "$scene_dir"
  done
fi

echo ""
echo "=== DEPTH XONG. Train thêm cờ:  -d <scene>/train/depths ==="
