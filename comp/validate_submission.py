#
# comp/validate_submission.py
#
# Kiểm tra submission.zip TRƯỚC KHI NỘP, đối chiếu với data thật.
# Mục 8 thể lệ: thiếu/thừa scene -> KHÔNG được tính điểm. Nên phải chắc chắn:
#   - Đủ số scene (đúng các scene trong --data).
#   - Mỗi scene đủ số ảnh = số dòng test_poses.csv.
#   - Tên ảnh khớp image_name trong CSV (so theo phần stem, bỏ đuôi).
#   - Kích thước ảnh = width×height trong CSV (không resize nhầm).
#
# Ví dụ:
#   python comp/validate_submission.py --zip submission/submission.zip \
#       --data phase1/private_set1
#

import os
import sys
import csv
import zipfile
import argparse
import io

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from PIL import Image
    HAVE_PIL = True
except Exception:
    HAVE_PIL = False


def read_expected(data_root):
    """{scene: {stem: (width,height)}} từ test_poses.csv của mỗi scene."""
    expected = {}
    for scene in sorted(os.listdir(data_root)):
        csv_path = os.path.join(data_root, scene, "test", "test_poses.csv")
        if not os.path.isfile(csv_path):
            continue
        imgs = {}
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                stem = os.path.splitext(row["image_name"].strip())[0]
                imgs[stem] = (int(round(float(row["width"]))), int(round(float(row["height"]))))
        expected[scene] = imgs
    return expected


def read_zip(zip_path):
    """{scene: {stem: (arcname, width_or_None, height_or_None)}}."""
    got = {}
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            parts = info.filename.replace("\\", "/").split("/")
            if len(parts) < 2:
                continue
            scene, fname = parts[-2], parts[-1]
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            stem = os.path.splitext(fname)[0]
            w = h = None
            if HAVE_PIL:
                try:
                    with zf.open(info) as fp:
                        im = Image.open(io.BytesIO(fp.read()))
                        w, h = im.size
                except Exception:
                    pass
            got.setdefault(scene, {})[stem] = (info.filename, w, h)
    return got


def validate(zip_path, data_root, strict_scene_name=True):
    expected = read_expected(data_root)
    got = read_zip(zip_path)

    errors, warnings = [], []

    # So sánh tên scene (không phân biệt hoa/thường để bắt lỗi nhẹ).
    exp_map = {s.lower(): s for s in expected}
    got_map = {s.lower(): s for s in got}

    missing = set(exp_map) - set(got_map)
    extra = set(got_map) - set(exp_map)
    for s in sorted(missing):
        errors.append(f"THIẾU scene: {exp_map[s]}")
    for s in sorted(extra):
        errors.append(f"THỪA scene (không có trong data): {got_map[s]}")

    for low in sorted(set(exp_map) & set(got_map)):
        es, gs = exp_map[low], got_map[low]
        if strict_scene_name and es != gs:
            warnings.append(f"Tên scene lệch hoa/thường: zip='{gs}' vs data='{es}'")
        exp_imgs, got_imgs = expected[es], got[gs]

        miss_i = set(exp_imgs) - set(got_imgs)
        extra_i = set(got_imgs) - set(exp_imgs)
        if miss_i:
            errors.append(f"[{gs}] thiếu {len(miss_i)} ảnh, vd: {sorted(miss_i)[:3]}")
        if extra_i:
            errors.append(f"[{gs}] thừa {len(extra_i)} ảnh, vd: {sorted(extra_i)[:3]}")

        # Kiểm tra kích thước.
        for stem in sorted(set(exp_imgs) & set(got_imgs)):
            ew, eh = exp_imgs[stem]
            _, gw, gh = got_imgs[stem]
            if gw is not None and (gw, gh) != (ew, eh):
                errors.append(f"[{gs}] {stem}: size {gw}×{gh} ≠ CSV {ew}×{eh}")

    # In báo cáo.
    print(f"=== VALIDATE {zip_path} ===")
    print(f"Scene trong data: {len(expected)} | trong zip: {len(got)}")
    for s in sorted(set(exp_map) & set(got_map)):
        gs = got_map[s]
        print(f"  {gs}: {len(got[gs])}/{len(expected[exp_map[s]])} ảnh")
    if not HAVE_PIL:
        warnings.append("Chưa cài Pillow -> KHÔNG kiểm được kích thước ảnh.")

    for w in warnings:
        print(f"[WARN] {w}")
    if errors:
        print(f"\n❌ {len(errors)} LỖI — KHÔNG nộp:")
        for e in errors:
            print(f"   - {e}")
        return False
    print("\n✅ HỢP LỆ — sẵn sàng nộp.")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--data", required=True, help="thư mục data, vd phase1/private_set1")
    args = ap.parse_args()
    ok = validate(args.zip, args.data)
    sys.exit(0 if ok else 1)
