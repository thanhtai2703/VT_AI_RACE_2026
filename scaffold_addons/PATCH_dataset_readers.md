# PATCH bắt buộc cho `Scaffold-GS/scene/dataset_readers.py`

Data cuộc thi có 2 đặc điểm khiến `readColmapCameras` của Scaffold (giống Inria gốc)
**crash** nếu không vá. Cả hai đã verify trên repo Inria của ta — đây là bản chuyển sang Scaffold.

Mở `Scaffold-GS/scene/dataset_readers.py`, tìm hàm `readColmapCameras`, áp 2 sửa dưới.

---

## FIX 1 — Xử lý camera SIMPLE_RADIAL/RADIAL như pinhole

COLMAP cameras.bin của data = model `SIMPLE_RADIAL` (params = [focal, cx, cy, k]).
Scaffold gốc chỉ nhận SIMPLE_PINHOLE/PINHOLE → gặp SIMPLE_RADIAL sẽ
`assert False, "Colmap camera model not handled..."`.

**TÌM** khối `if intr.model=="SIMPLE_PINHOLE": ... elif intr.model=="PINHOLE": ... else: assert False`
(thường ngay sau `R = np.transpose(qvec2rotmat(extr.qvec))` / `T = np.array(extr.tvec)`).

**THÊM một nhánh `elif` NGAY TRƯỚC `else: assert False`:**

```python
        elif intr.model in ("SIMPLE_RADIAL", "RADIAL"):
            # Data cuộc thi = SIMPLE_RADIAL: params = [f, cx, cy, k(, k2)].
            # Train PINHOLE trực tiếp — bỏ distortion k (k≈0.008 rất nhỏ, và
            # test_poses.csv là pinhole thuần với fx/fy khớp train). params[0]=focal.
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
```

Giữ nguyên dòng `else: assert False, "...camera model not handled..."` ngay sau nhánh mới.

---

## FIX 2 — Bỏ qua camera COLMAP không có file ảnh trên đĩa

images.bin đăng ký ~388 camera nhưng BTC chỉ phát 200 ảnh trong train/images
(dư = ảnh test + ảnh COLMAP-only). Không skip → `FileNotFoundError` khi load ảnh.

**TÌM** đoạn dựng `image_path` rồi (ở Scaffold gốc) mở ảnh, đại loại:

```python
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
```

**THÊM NGAY TRƯỚC dòng `image = Image.open(image_path)` (hoặc trước khi tạo CameraInfo):**

```python
        # Data cuộc thi: images.bin đăng ký nhiều camera hơn số ảnh được phát.
        # Bỏ qua camera không có file ảnh trên đĩa -> chỉ train ảnh thật sự có.
        if not os.path.isfile(image_path):
            continue
```

⚠️ Đặt `continue` TRƯỚC mọi lệnh mở/đọc ảnh của camera đó, nếu không vẫn crash.

---

## KIỂM sau khi vá
Chạy thử train 1 scene ~100 iter, nếu KHÔNG còn "camera model not handled" và KHÔNG
`FileNotFoundError` là 2 fix đã ăn:

```bash
python train.py -s data/hcm0031 -m /tmp/probe --iterations 100 --eval  # (bỏ --eval nếu Scaffold không hỗ trợ)
```
(Có thể dừng sớm bằng Ctrl-C ngay khi thấy nó bắt đầu training loop.)
