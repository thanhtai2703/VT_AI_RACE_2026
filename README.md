# NVS Competition — 3D Gaussian Splatting pipeline

Giải bài Novel View Synthesis: train 3DGS mỗi scene → render đúng `test_poses.csv` → đóng gói submission. Backbone: [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) (đặt sẵn ở `gaussian-splatting/`). Chiến lược đầy đủ: xem [PLAN.md](PLAN.md).

## Bố cục
```
gaussian-splatting/     # repo Inria (train.py, render.py, metrics.py) — KHÔNG sửa lõi
phase1/
  public_set/   <scene>/{train,test}   # CÓ ảnh test -> tự eval Score
  private_set1/ <scene>/{train,test}   # KHÔNG ảnh test -> BTC chấm
comp/                   # code đặc thù cuộc thi (phần tự viết)
  test_pose_loader.py   #  ⭐ đọc CSV -> Camera đúng convention COLMAP (dễ sai nhất)
  render_test_poses.py  #  load .ply -> render theo CSV -> PNG (không resize)
  make_submission.py    #  gom ảnh -> submission.zip
  validate_submission.py#  kiểm scene/ảnh/tên/size TRƯỚC khi nộp
  local_eval.py         #  Score = 0.4(1-LPIPS)+0.3 SSIM+0.3 PSNR_norm trên public
configs/base.yaml       # cờ mặc định (train pinhole, --antialiasing, sparse_adam)
scripts/run_all.sh      # loop mọi scene: train -> render -> (public) eval
outputs/                # models/ renders/ logs/
```

## Dữ liệu (đã khảo sát)
- **13 scene**: 5 public (có gt test) + 8 private (BTC chấm). Ảnh drone DJI, scale 1/4 → **1320×989**.
- Camera model COLMAP = **`SIMPLE_RADIAL`** (`k≈0.008`), nhưng `test_poses.csv` là **pinhole thuần** với `fx/fy/cx/cy` khớp train.
  → **Chốt: train pinhole trực tiếp**, bỏ qua `k` (nhất quán train↔test, sai số không đáng kể).
- Principal point khớp tâm ảnh (`cx=w/2, cy=h/2`) → `getProjectionMatrix` của repo dùng trực tiếp, không cần vá lệch tâm.

## Setup (máy GPU, khuyến nghị 24GB — 3090 Ti/4090 đều tốt)
```bash
# 1) env core Inria (conda)
cd gaussian-splatting
conda env create -f environment.yml       # hoặc bản CUDA mới hơn nếu cần
conda activate gaussian_splatting
# (đảm bảo 3 submodule đã build: diff-gaussian-rasterization, simple-knn, fused-ssim)

# 2) phụ thuộc comp/
pip install -r ../requirements.txt
```

## Chạy
```bash
# Sanity nhanh 1 scene public 7k iter -> KIỂM convention trước khi chạy tất cả
python gaussian-splatting/train.py -s phase1/public_set/hcm0031/train \
    -m outputs/models/hcm0031 --antialiasing --optimizer_type sparse_adam \
    --iterations 7000 --save_iterations 7000 --test_iterations 7000 --disable_viewer
python comp/render_test_poses.py -m outputs/models/hcm0031 \
    --poses phase1/public_set/hcm0031/test/test_poses.csv \
    --out outputs/renders/hcm0031 --iteration 7000 --antialiasing
python comp/local_eval.py --renders outputs/renders/hcm0031 \
    --gt phase1/public_set/hcm0031/test/images
# -> Score hợp lý (LPIPS thấp, ảnh giống gt) = convention ĐÚNG.

# Full loop (30k iter)
bash scripts/run_all.sh phase1/public_set       # kiểm Score toàn public
bash scripts/run_all.sh phase1/private_set1      # tạo render cho bài nộp

# Đóng gói + validate
python comp/make_submission.py --renders outputs/renders --out submission/submission.zip
python comp/validate_submission.py --zip submission/submission.zip --data phase1/private_set1
```

## Điểm PHẢI kiểm trước khi nộp
- [ ] Sanity check convention pose PASS (render ≈ ảnh train/test gốc). Xem PLAN mục 4 bước 2.
- [ ] **Format submission** (tên thư mục scene, đuôi ảnh) khớp thể lệ BTC — hiện README data KHÔNG nêu rõ; `make_submission.py` mặc định giữ tên gốc + `.png`, có cờ `--keep-ext-jpg`, `--lower-scene`. **Xác nhận với thể lệ.**
- [ ] `validate_submission.py` báo ✅ (đủ scene, đủ ảnh, đúng size).
- [ ] `--antialiasing` khớp giữa train và render.
```
