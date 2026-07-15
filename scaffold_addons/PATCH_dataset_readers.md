# PATCH cho `Scaffold-GS/scene/dataset_readers.py`

Data cuộc thi có 2 đặc điểm khiến `readColmapCameras` crash. **Đã kiểm bản Scaffold hiện tại
(2026-07): FIX 1 CÓ SẴN rồi — chỉ cần làm FIX 2.**

Mở `Scaffold-GS/scene/dataset_readers.py`, hàm `readColmapCameras`.

---

## FIX 1 — SIMPLE_RADIAL as pinhole → THƯỜNG ĐÃ CÓ SẴN, chỉ kiểm

Bản Scaffold hiện tại đã viết:
```python
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            ...
```
tức ĐÃ xử lý SIMPLE_RADIAL (lấy params[0] làm focal — đúng). **Nếu thấy dòng này thì KHÔNG cần
làm gì cho FIX 1.** Chỉ khi bản của bạn KHÁC (else assert ngay, không có SIMPLE_RADIAL) mới thêm
`or intr.model == "SIMPLE_RADIAL"` vào điều kiện `if` đầu.

---

## FIX 2 — Bỏ qua camera COLMAP không có file ảnh trên đĩa (BẮT BUỘC LÀM)

images.bin đăng ký ~388 camera nhưng chỉ có ~200 ảnh trong train/images (dư = test + COLMAP-only).
Không skip → `FileNotFoundError` ở `Image.open`.

**TÌM 3 dòng này:**
```python
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
```

**SỬA THÀNH (thêm 2 dòng ở giữa):**
```python
        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        if not os.path.isfile(image_path):
            continue
        image = Image.open(image_path)
```

### Cách nhanh (dán vào terminal, tự chèn đúng chỗ — ĐỔI đường dẫn nếu cần):
```bash
python3 - <<'PY'
f = "/root/Scaffold-GS/scene/dataset_readers.py"   # ĐỔI nếu clone chỗ khác
import io
s = io.open(f, encoding="utf-8").read()
old = '''        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)'''
new = '''        image_name = os.path.basename(image_path).split(".")[0]
        if not os.path.isfile(image_path):
            continue
        image = Image.open(image_path)'''
if "if not os.path.isfile(image_path):" in s:
    print("Da va roi, bo qua.")
elif old in s:
    io.open(f, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print("VA THANH CONG fix 2.")
else:
    print("KHONG tim thay doan can va - dan readColmapCameras cho reviewer kiem.")
PY
```

---

## KIỂM đã vá
```bash
grep -n "isfile\|SIMPLE_RADIAL" scene/dataset_readers.py
```
Phải thấy CẢ dòng `SIMPLE_RADIAL` (fix 1) VÀ dòng `if not os.path.isfile` (fix 2).
