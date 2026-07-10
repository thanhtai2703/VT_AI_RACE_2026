#!/usr/bin/env bash
# scripts/setup_gpu.sh — dựng môi trường 3DGS trong container Docker GPU.
#
# GIẢ ĐỊNH: chạy trong image PyTorch `-devel` (đã có torch + CUDA + nvcc).
#   vd: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel
#
# Cách dùng (chạy TỪ THƯ MỤC GỐC của repo VT_AI_RACE_2026):
#   bash scripts/setup_gpu.sh
#
# Script này:
#   1. Kiểm tra nvcc / GPU / torch (fail sớm nếu image sai loại).
#   2. Clone repo Inria (kèm submodule) vào ./gaussian-splatting nếu chưa có.
#   3. Build 3 submodule CUDA (bước hay lỗi nhất).
#   4. Cài deps của comp/.
#   5. Verify import — in OK nếu sẵn sàng train.

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
echo "=== ROOT = $ROOT ==="

# --- 0) Tiện ích hệ thống (image tối giản hay thiếu git/compiler) ---
if ! command -v git >/dev/null 2>&1; then
  echo "[setup] cài git + build-essential..."
  apt-get update -y && apt-get install -y git build-essential ninja-build
fi

# --- 1) KIỂM TRA MÔI TRƯỜNG (fail sớm) ---
echo "=== 1) Kiểm tra môi trường ==="
if ! command -v nvcc >/dev/null 2>&1; then
  echo "❌ KHÔNG thấy nvcc. Bạn đang dùng image '-runtime' -> KHÔNG build được submodule."
  echo "   Đổi sang image '-devel' (vd pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel)."
  exit 1
fi
nvcc --version | grep release || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "❌ nvidia-smi lỗi — GPU chưa gắn vào container."; exit 1; }
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available in torch'; \
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '|', torch.cuda.get_device_name(0))"

# --- 2) Repo Inria (clone kèm submodule nếu chưa có source) ---
echo "=== 2) Repo Inria gaussian-splatting ==="
if [[ ! -f gaussian-splatting/train.py ]]; then
  echo "[setup] chưa có -> clone mới kèm --recursive"
  rm -rf gaussian-splatting
  git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git
elif [[ ! -f gaussian-splatting/submodules/diff-gaussian-rasterization/setup.py ]]; then
  echo "[setup] có repo nhưng submodule rỗng -> lấy submodule"
  ( cd gaussian-splatting && git submodule update --init --recursive )
else
  echo "[setup] gaussian-splatting đã đầy đủ."
fi

# --- 3) Build 3 submodule CUDA (bước hay lỗi nhất) ---
echo "=== 3) Build submodule CUDA ==="
# TORCH_CUDA_ARCH_LIST: 3090 Ti/3090 = 8.6 ; 4090 = 8.9. Đặt cả 2 cho an toàn.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6;8.9}"
echo "[setup] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
pip install ./gaussian-splatting/submodules/diff-gaussian-rasterization
pip install ./gaussian-splatting/submodules/simple-knn
pip install ./gaussian-splatting/submodules/fused-ssim

# --- 4) Deps của comp/ ---
echo "=== 4) Deps comp/ ==="
pip install plyfile opencv-python-headless Pillow tqdm numpy

# --- 5) Verify ---
echo "=== 5) Verify import ==="
python - <<'PY'
import importlib, sys
ok = True
for m in ["torch", "torchvision", "plyfile", "cv2", "PIL", "tqdm",
          "diff_gaussian_rasterization", "simple_knn", "fused_ssim"]:
    try:
        importlib.import_module(m); print(f"  ok  {m}")
    except Exception as e:
        ok = False; print(f"  FAIL {m}: {e}")
try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa
    print("  ok  SparseGaussianAdam (sparse_adam khả dụng)")
except Exception as e:
    print(f"  WARN SparseGaussianAdam: {e} -> bỏ cờ --optimizer_type sparse_adam")
sys.exit(0 if ok else 1)
PY

echo ""
echo "✅ SETUP XONG. Kế tiếp:"
echo "   1) Đảm bảo data ở ./phase1/ (public_set + private_set1)"
echo "   2) Sanity 1 scene:  bash scripts/run_all.sh phase1/public_set 7000"
echo "   3) Full private:    bash scripts/run_all.sh phase1/private_set1 30000"
