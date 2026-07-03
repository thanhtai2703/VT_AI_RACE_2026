# Kế hoạch triển khai — Tái dựng scene 3D & Sinh ảnh góc nhìn mới

Tài liệu này mô tả chiến lược, cấu trúc repo và pipeline để giải bài toán Novel View Synthesis (NVS) của cuộc thi. BTC đã cung cấp sẵn COLMAP sparse reconstruction, nên bước Structure-from-Motion (khó nhất) đã xong. Trọng tâm còn lại là **train tốt → render đúng test pose → đóng gói đúng format**.

Backbone chọn dùng: **repo chính thức của Inria** — `graphdeco-inria/gaussian-splatting`. Bản này (cập nhật 10/2024) đã tích hợp sẵn anti-aliasing (EWA filter từ Mip-Splatting), depth regularization và exposure compensation, nên không cần fork sang repo Mip-Splatting tách rời nữa.

---

## 1. Chiến lược tổng thể (bám theo metric)

Công thức chấm điểm:

```
Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
```

Nhận xét quan trọng: **LPIPS có trọng số cao nhất (0.4)** → chất lượng *cảm quan* quan trọng hơn sai số pixel tuyệt đối. Điều này định hướng các quyết định kỹ thuật:

- Repo Inria đã tối ưu sẵn loss `L1 + D-SSIM` (λ ≈ 0.2) → tốt cho SSIM/PSNR. Để kéo LPIPS xuống, cân nhắc fine-tune giai đoạn cuối bằng perceptual loss.
- Test poses là **góc nhìn mới** (out-of-distribution), có thể ở khoảng cách/zoom khác train → 3DGS thuần bị aliasing nặng (erosion/dilation). Bật cờ **`--antialiasing`** (EWA filter tích hợp sẵn) để render alias-free ở các pose OOD.
- Floaters (Gaussian rác lơ lửng) làm hỏng cả 3 metric ở góc nhìn mới → bật **depth regularization** (`-d`) để giảm floaters, hiệu quả nhất ở vùng ít texture (đường, tường).
- Dữ liệu drone/camera cầm tay hay bị chênh phơi sáng giữa các khung hình → bật **exposure compensation** để bù.

**Quy tắc vàng:** có submission hợp lệ trước, tối ưu sau.

---

## 2. Cấu trúc repo

Cách tổ chức: fork repo Inria làm submodule/thư mục lõi, bọc thêm lớp script riêng để xử lý phần *đặc thù cuộc thi* (đọc `test_poses.csv`, render theo CSV, đóng gói ZIP). Không sửa lõi Inria trừ khi thật cần — giữ dễ tái lập.

```text
nvs-competition/
├── README.md
├── requirements.txt
├── gaussian-splatting/            # repo Inria (git submodule --recursive)
│   ├── train.py                   # train sẵn có
│   ├── render.py                  # render train/test split sẵn có
│   ├── metrics.py                 # LPIPS/SSIM/PSNR sẵn có (đúng 3 metric cuộc thi)
│   ├── convert.py                 # KHÔNG cần dùng (BTC đã cho sparse sẵn)
│   └── submodules/                # diff-gaussian-rasterization, simple-knn
├── configs/
│   ├── base.yaml                  # hyperparams + cờ mặc định cho mọi scene
│   └── scene_overrides/           # override riêng cho scene khó (nếu cần)
│       └── scene_003.yaml
├── data/
│   ├── raw/                       # data gốc BTC (symlink, không commit)
│   │   ├── scene_001/{train,test}
│   │   └── ...
│   └── prepared/                  # data đã sắp lại đúng layout repo Inria yêu cầu
├── comp/                          # ⭐ CODE ĐẶC THÙ CUỘC THI — phần phải tự viết
│   ├── prepare_data.py            # raw BTC → layout repo Inria
│   ├── test_pose_loader.py        # đọc test_poses.csv → Camera objects  ⚠️ QUAN TRỌNG NHẤT
│   ├── render_test_poses.py       # render đúng các pose trong CSV (không phải split mặc định)
│   ├── make_submission.py         # gom ảnh → submission.zip đúng format
│   ├── validate_submission.py     # check số scene / số ảnh / tên / size trước khi nộp
│   └── local_eval.py              # tự tính Score trên tập "test giả" tách từ train
├── scripts/
│   └── run_all.sh                 # loop qua tất cả scene: prepare → train → render → zip
├── outputs/
│   ├── models/scene_xxx/          # checkpoint + config từng scene
│   ├── renders/scene_xxx/         # ảnh render theo test pose
│   └── logs/                      # training logs (BTC yêu cầu để tái lập)
└── submission/
    └── submission.zip
```

---

## 3. Repo Inria ánh xạ vào bài thi thế nào

Đây là phần mấu chốt: repo Inria gần như "cắm là chạy" với dữ liệu cuộc thi, nhưng có **đúng một điểm phải tự viết thêm** — render theo `test_poses.csv`.

### 3.1. Khớp định dạng dữ liệu

Repo Inria mong đợi mỗi scene có layout:

```text
<location>/
├── images/                 ← ảnh training
└── sparse/0/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

Dữ liệu BTC cho `train/images/` + `train/sparse/0/*.bin` → **trùng khớp hoàn toàn**. Nghĩa là trỏ thẳng `train.py -s data/raw/scene_xxx/train` là train được ngay, không cần chạy `convert.py`.

⚠️ Kiểm tra một việc: rasterizer chỉ nhận camera model `SIMPLE_PINHOLE` hoặc `PINHOLE`. Đọc `cameras.bin` xem BTC dùng model gì; nếu là `OPENCV`/`RADIAL` (có distortion) thì phải undistort trước.

### 3.2. Điểm phải tự viết: render theo test_poses.csv

`render.py` gốc của Inria chỉ render theo **các camera đã nạp từ COLMAP** (train/test split qua `--eval`). Nhưng cuộc thi đưa test poses dưới dạng **CSV rời**, không nằm trong `images.bin`. Đây chính là lý do cần `comp/render_test_poses.py`:

1. Đọc từng dòng CSV → dựng đối tượng `Camera` (tận dụng class có sẵn của repo) với đúng `qw,qx,qy,qz`, `tx,ty,tz`, `fx,fy,cx,cy`, `width,height`.
2. Gọi hàm render trong `gaussian_renderer/__init__.py` (dùng trực tiếp từ Python rất đơn giản, README có nói).
3. Lưu PNG đúng tên `image_name` và đúng kích thước.

Convention phải khớp COLMAP: `(q, t)` là biến đổi **world → camera**. Sai chiều hoặc sai thứ tự quaternion là mất điểm oan → xem sanity check ở Bước 2 mục 4.

### 3.3. Tận dụng sẵn `metrics.py`

`metrics.py` của repo đã tính đúng cả ba LPIPS/SSIM/PSNR. Bạn chỉ cần bọc thêm công thức tổng hợp `Score` và bước chuẩn hóa PSNR để có vòng eval nội bộ (`comp/local_eval.py`).

---

## 4. Pipeline 5 bước (kèm lệnh cụ thể)

### Bước 1 — Chuẩn hóa dữ liệu (`comp/prepare_data.py`)
Tạo symlink/sắp lại `data/raw/scene_xxx/train` về đúng layout repo. Kiểm tra camera model trong `cameras.bin`, undistort nếu cần.

### Bước 2 — Đọc test poses cho đúng (`comp/test_pose_loader.py`) — CHỖ DỄ SAI NHẤT
Các lỗi thường gặp:
- Nhầm world→camera với camera→world (phải đúng chiều, hoặc invert).
- Nhầm thứ tự quaternion (`qw, qx, qy, qz` — Hamilton, scalar-first).
- Xây projection từ `fx, fy, cx, cy` mà quên principal point có thể lệch tâm.

**Sanity check bắt buộc:** lấy vài pose từ chính tập *train* (đã biết ảnh gốc), cho đi qua loader + renderer, so với ảnh train. Khớp → convention đúng. Đây là bài test đáng giá nhất trước khi chạy toàn bộ.

### Bước 3 — Train từng scene
Mỗi scene train độc lập, khởi tạo từ `points3D.bin`. Lệnh gợi ý:

```bash
python gaussian-splatting/train.py \
  -s data/prepared/scene_001 \
  -m outputs/models/scene_001 \
  --antialiasing \
  -d outputs/depths/scene_001 \      # depth regularization (tạo depth bằng Depth-Anything-V2 trước)
  --optimizer_type sparse_adam \     # train nhanh hơn ~2.7×
  --iterations 30000 \
  --save_iterations 7000 30000
```

Cân nhắc thêm exposure compensation cho dữ liệu chụp ngoài trời:
`--exposure_lr_init 0.001 --exposure_lr_final 0.0001 --exposure_lr_delay_steps 5000 --exposure_lr_delay_mult 0.001 --train_test_exp`

Lưu checkpoint + log đầy đủ (BTC yêu cầu tái lập — mục 10.3).

### Bước 4 — Render test poses (`comp/render_test_poses.py`)

```bash
python comp/render_test_poses.py \
  -m outputs/models/scene_001 \
  --poses data/raw/scene_001/test/test_poses.csv \
  --out outputs/renders/scene_001 \
  --antialiasing        # PHẢI khớp cờ đã dùng khi train
```

**Không resize sau khi render** — render thẳng đúng `width × height` trong CSV để tránh mất điểm SSIM/LPIPS.

### Bước 5 — Đóng gói + validate
```bash
python comp/make_submission.py --renders outputs/renders --out submission/submission.zip
python comp/validate_submission.py --zip submission/submission.zip --data data/raw
```
Mục 8 nêu rõ **thiếu/thừa scene → không được tính điểm**. Validate phải kiểm: đủ số scene, đủ số ảnh mỗi scene, đúng tên file, đúng kích thước. Chạy validate trước mỗi lần nộp.

Format submission:

```text
submission.zip
├── scene_001/
│   ├── 0001.png
│   ├── 0002.png
│   └── ...
├── scene_002/
│   └── ...
└── ...
```

---

## 5. Tối ưu để leo bảng (sau khi có điểm hợp lệ)

- **Local eval loop:** tách một phần train poses làm "test giả", chạy `comp/local_eval.py` tính đúng công thức Score. Chốt `PSNR_max` giống BTC (nếu công bố) để chuẩn hóa cho khớp.
- **Bật đủ 3 tính năng tích hợp:** `--antialiasing`, depth reg (`-d`), exposure comp — thường cho lợi ích rõ trên dữ liệu drone/cầm tay ngoài trời.
- **Fine-tune giảm LPIPS:** LPIPS nặng ký nhất → thêm perceptual loss ở giai đoạn cuối training.
- **Densification / learning rate tuning:** với scene lớn hoặc đa tỉ lệ (cận cảnh lẫn xa), giảm `--position_lr_init`, `--scaling_lr` (theo gợi ý FAQ của repo) để bớt artifact; chỉnh `--densify_grad_threshold` để mọc thêm Gaussian ở vùng thiếu chi tiết.

---

## 6. Checklist tuân thủ quy định (mục 10)

- [ ] Chỉ dùng dữ liệu BTC cung cấp, không dùng dữ liệu ngoài liên quan scene thi.
- [ ] Ảnh sinh **hoàn toàn tự động**, không chỉnh sửa thủ công.
- [ ] Lưu đầy đủ: code, config, danh sách thư viện + phiên bản, checkpoint, training logs.
- [ ] Pipeline tái lập được kết quả nộp bài.

---

## 7. Lưu ý phần cứng & môi trường

- GPU CUDA Compute Capability ≥ 7.0; khuyến nghị **24GB VRAM** cho chất lượng cao (FAQ repo có hướng giảm VRAM: tăng `--densify_grad_threshold`, giảm `--densify_until_iter`).
- Cài qua Conda (`environment.yml` của repo), giả định CUDA SDK 11.
- Clone kèm `--recursive` để lấy submodule rasterizer + simple-knn.
- Depth regularization cần chạy Depth-Anything-V2 sinh depth map + `utils/make_depth_scale.py` trước khi train.

---

## 8. Tham khảo

- Backbone chính thức: https://github.com/graphdeco-inria/gaussian-splatting
- 3D Gaussian Splatting (Kerbl et al., ACM TOG 2023)
- Mip-Splatting / EWA anti-aliasing: https://arxiv.org/abs/2311.16493
- Depth-Anything-V2 (sinh depth cho depth regularization)
- LPIPS: Zhang et al., CVPR 2018 — https://arxiv.org/abs/1801.03924
- SSIM/PSNR: Wang et al., IEEE TIP 2004 — doi:10.1109/TIP.2003.819861
