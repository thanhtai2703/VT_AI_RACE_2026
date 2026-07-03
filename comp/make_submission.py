#
# comp/make_submission.py
#
# Gom ảnh render của tất cả scene -> submission.zip đúng format.
#
# ⚠️ FORMAT SUBMISSION CHÍNH XÁC CHƯA ĐƯỢC BTC CÔNG BỐ RÕ trong README.
#    Mặc định (an toàn nhất, suy từ dữ liệu):
#      submission.zip
#      ├── <scene_name>/            # tên scene GỐC, vd HCM0249, hcm0031
#      │   ├── <image_name>.png     # tên ảnh GỐC trong CSV, đổi đuôi -> .png
#      │   └── ...
#    Có cờ để đổi hành vi nếu luật thi yêu cầu khác:
#      --keep-ext-jpg   : giữ đuôi .JPG thay vì .png (ảnh vẫn là PNG bên trong)
#      --lower-scene    : hạ scene name về chữ thường
#    -> XÁC NHẬN VỚI THỂ LỆ TRƯỚC KHI NỘP.
#
# Ví dụ:
#   python comp/make_submission.py --renders outputs/renders --out submission/submission.zip
#

import os
import sys
import zipfile
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IMG_EXT = (".png", ".jpg", ".jpeg")


def collect(renders_root):
    """Trả về dict {scene_name: [đường dẫn ảnh...]}."""
    scenes = {}
    for scene in sorted(os.listdir(renders_root)):
        sdir = os.path.join(renders_root, scene)
        if not os.path.isdir(sdir):
            continue
        imgs = sorted(f for f in os.listdir(sdir) if f.lower().endswith(IMG_EXT))
        if imgs:
            scenes[scene] = [os.path.join(sdir, f) for f in imgs]
    return scenes


def make_zip(renders_root, out_zip, keep_ext_jpg=False, lower_scene=False):
    scenes = collect(renders_root)
    if not scenes:
        raise SystemExit(f"[make_submission] Không thấy ảnh nào trong {renders_root}")

    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)
    total = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene, imgs in scenes.items():
            arc_scene = scene.lower() if lower_scene else scene
            for img in imgs:
                base = os.path.basename(img)
                stem, _ = os.path.splitext(base)
                arc_name = stem + (".JPG" if keep_ext_jpg else ".png")
                zf.write(img, arcname=f"{arc_scene}/{arc_name}")
                total += 1
            print(f"  {arc_scene}: {len(imgs)} ảnh")

    size_mb = os.path.getsize(out_zip) / 1e6
    print(f"[make_submission] {len(scenes)} scene, {total} ảnh -> {out_zip} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="thư mục gốc chứa outputs/renders/<scene>/*.png")
    ap.add_argument("--out", required=True, help="đường dẫn submission.zip")
    ap.add_argument("--keep-ext-jpg", action="store_true", help="đặt tên trong zip với đuôi .JPG")
    ap.add_argument("--lower-scene", action="store_true", help="hạ tên scene về chữ thường")
    args = ap.parse_args()
    make_zip(args.renders, args.out, args.keep_ext_jpg, args.lower_scene)
