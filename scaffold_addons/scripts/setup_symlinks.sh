#!/usr/bin/env bash
# scaffold_addons/scripts/setup_symlinks.sh
#
# Tạo layout data mà Scaffold-GS mong đợi (<scene>/images + <scene>/sparse)
# bằng SYMLINK tới data phase1 của repo chính — KHÔNG copy 3.2GB.
#
# Scaffold ăn: Scaffold-GS/data/<scene>/{images, sparse}
# Data ta có:  <REPO>/phase1/<set>/<scene>/train/{images, sparse}
#
# Dùng (chạy TRONG thư mục Scaffold-GS/ đã clone):
#   bash <path>/setup_symlinks.sh <REPO>/phase1/public_set
#   bash <path>/setup_symlinks.sh <REPO>/phase1/private_set1
#
set -euo pipefail

SRC_ROOT="${1:?Thiếu đường dẫn set, vd /path/VT_AI_RACE_2026/phase1/public_set}"
SRC_ROOT="$(cd "$SRC_ROOT" && pwd)"   # tuyệt đối (symlink cần đích tuyệt đối)

# Chạy trong thư mục Scaffold-GS (nơi có train.py). Kiểm nhẹ.
[ -f "./train.py" ] || { echo "❌ Chạy script này TRONG thư mục Scaffold-GS/ (không thấy train.py)"; exit 1; }

mkdir -p data
for scene_dir in "$SRC_ROOT"/*/; do
  scene="$(basename "$scene_dir")"
  img="$scene_dir/train/images"
  spa="$scene_dir/train/sparse"
  if [[ ! -d "$img" || ! -d "$spa" ]]; then
    echo "[$scene] BỎ QUA (thiếu train/images hoặc train/sparse)"; continue
  fi
  mkdir -p "data/$scene"
  ln -sfn "$(cd "$img" && pwd)" "data/$scene/images"
  ln -sfn "$(cd "$spa" && pwd)" "data/$scene/sparse"
  n=$(ls "data/$scene/images" | wc -l)
  echo "[$scene] symlink OK ($n ảnh) -> data/$scene/{images,sparse}"
done
echo "=== XONG. Kiểm: ls -la data/<scene>/ ==="
