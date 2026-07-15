# Scaffold-GS add-ons — hướng dẫn chạy trên máy thuê

Bộ file để chạy **Scaffold-GS** (city-super, CVPR 2024) trên data cuộc thi, nhắm gap ~14 điểm
(Score 63 → top 10 = 76.85). Ta **clone repo Scaffold gốc** và chạy trực tiếp, chỉ thêm 1 script
render theo `test_poses.csv` + 2 fix data. KHÔNG đụng repo chính.

⚠️ **Đây là canh bạc có cơ sở, KHÔNG đảm bảo thắng.** Cổng quyết định = ablation 1 scene public
(Bước 5). Nếu Scaffold thua mốc densify (Score 0.6900 trên hcm0031) thì bỏ, quay về densify tuning.

## File trong bộ này
- `render_test_poses_scaffold.py` — render theo CSV (API Scaffold). Copy vào thư mục `Scaffold-GS/`.
- `PATCH_dataset_readers.md` — 2 fix bắt buộc cho `Scaffold-GS/scene/dataset_readers.py`.
- `scripts/setup_symlinks.sh` — tạo layout data cho Scaffold bằng symlink (không copy 3.2GB).
- `scripts/run_scaffold.sh` — loop train → render → (public) eval.

---

## Các bước (trên máy thuê 3090 24GB)

Giả sử repo chính ở `~/VT_AI_RACE_2026` (đã có `phase1/` + `comp/`). Đặt `REPO=~/VT_AI_RACE_2026`.

### B1. Clone Scaffold + build submodule
```bash
cd ~
git clone https://github.com/city-super/Scaffold-GS.git --recursive
cd Scaffold-GS

# Torch khớp container (KHÔNG dùng environment.yml gốc — nó pin CUDA 11.6/torch cũ).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Build submodule Scaffold (rasterizer variant riêng + simple-knn).
pip install ./submodules/*      # hoặc pip install từng cái nếu * không ăn
# nếu submodules/ có nhiều thư mục, cài rõ:
#   pip install ./submodules/diff-gaussian-rasterization
#   pip install ./submodules/simple-knn

pip install plyfile tqdm opencv-python einops   # phụ thuộc runtime Scaffold

# Verify import (không lỗi = OK)
python -c "from gaussian_renderer import render, prefilter_voxel, GaussianModel; from scene import Scene; print('SCAFFOLD IMPORT OK')"
```

### B2. Vá 2 fix data bắt buộc
Mở `scene/dataset_readers.py`, áp đúng 2 sửa trong **`PATCH_dataset_readers.md`** (SIMPLE_RADIAL as
pinhole + skip camera thiếu ảnh). KHÔNG vá → train crash ngay.

### B3. Copy script render vào thư mục Scaffold
```bash
cp $REPO/scaffold_addons/render_test_poses_scaffold.py ~/Scaffold-GS/
```

### B4. Tạo symlink data (public trước để ablation)
```bash
cd ~/Scaffold-GS
bash $REPO/scaffold_addons/scripts/setup_symlinks.sh $REPO/phase1/public_set
ls -la data/hcm0031/    # phải thấy images -> ...  sparse -> ...
```

### B5. ⚠️ SANITY 7k TRƯỚC (bắt buộc — bắt lỗi convention sớm)
```bash
cd ~/Scaffold-GS
python train.py -s data/hcm0031 -m outputs/hcm0031_sanity \
    --iterations 7000 --voxel_size 0.001 --update_init_factor 16 \
    --appearance_dim 32 --ratio 1 --save_iterations 7000
python render_test_poses_scaffold.py -m outputs/hcm0031_sanity \
    --poses $REPO/phase1/public_set/hcm0031/test/test_poses.csv \
    --out $REPO/outputs/scaffold_renders/hcm0031_sanity --iteration 7000
python $REPO/comp/local_eval.py \
    --renders $REPO/outputs/scaffold_renders/hcm0031_sanity \
    --gt $REPO/phase1/public_set/hcm0031/test/images --psnr_max 50
```
**MỞ vài ảnh trong `outputs/scaffold_renders/hcm0031_sanity/`.** Nếu ảnh giống cảnh (nhà, đường) và
LPIPS hợp lý (~0.2-0.3) → convention ĐÚNG, đi tiếp. Nếu **ảnh ĐEN/nhiễu/lệch** → sai convention
Camera Scaffold (RỦI RO #1), DỪNG và báo lại — đừng chạy tiếp.

### B6. Ablation đầy đủ 15k (cổng quyết định)
```bash
cd ~/Scaffold-GS
REPO=$REPO bash $REPO/scaffold_addons/scripts/run_scaffold.sh public_set 15000
```
So Score in ra với mốc đã biết trên hcm0031: **baseline=0.6801, densify=0.6900**.
- Scaffold **> 0.69 rõ** (ưu tiên LPIPS thấp) → THẮNG, đi B7.
- Không hơn → BỎ Scaffold, quay về densify tuning (repo chính, đã sẵn sàng).

### B7. (Nếu thắng) Train 8 scene private + đóng gói
```bash
cd ~/Scaffold-GS
bash $REPO/scaffold_addons/scripts/setup_symlinks.sh $REPO/phase1/private_set1
REPO=$REPO bash $REPO/scaffold_addons/scripts/run_scaffold.sh private_set1 30000

# Đo dung lượng sau scene ĐẦU (từ repo chính):
cd $REPO
python comp/make_submission.py --renders outputs/scaffold_renders \
    --out /tmp/probe.zip --format jpg --quality 95 --ext JPG
ls -lh /tmp/probe.zip     # 1 scene >45MB -> hạ --quality 90 lúc đóng gói cuối

# Đóng gói cuối + validate:
python comp/make_submission.py --renders outputs/scaffold_renders \
    --out submission/scaffold.zip --format jpg --quality 95 --ext JPG
python comp/validate_submission.py --zip submission/scaffold.zip --data phase1/private_set1
# -> phải: 434 ảnh, đuôi .JPG HOA, ≤350MB, missing=0 thừa=0.
```

---

## Config aerial (đã set trong run_scaffold.sh, override bằng env nếu cần)
- `APPEARANCE_DIM=32` — BẬT appearance embedding (ảnh drone ngoài trời đổi phơi sáng). Đây là khác
  biệt chính so với default Scaffold (appearance_dim=0).
- `VOXEL_SIZE=0.001`, `UPDATE_INIT_FACTOR=16`, `RATIO=1` — mặc định Scaffold, hợp cảnh cỡ metric.
- Nếu OOM (khó, Scaffold nhẹ VRAM): giảm `RATIO` hoặc `UPDATE_INIT_FACTOR` nhỏ hơn.

## Rủi ro (đọc trước khi chạy lâu)
1. **Convention Camera** — cổng B5 chặn. Ảnh đen = dừng ngay.
2. **Scaffold không thắng aerial** — cổng B6 chặn. Có fallback densify.
3. **Bỏ --eval train.py lỗi** — xem ghi chú cuối run_scaffold.sh (thêm --eval, không ảnh hưởng render CSV).
