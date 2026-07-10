#
# comp/make_submission.py
#
# Gom ảnh render của tất cả scene -> submission.zip đúng format.
#
# ✅ FORMAT ĐÃ XÁC NHẬN VỚI BTC (2026-07):
#      submission.zip
#      ├── <scene_name>/            # tên scene GỐC (HCM0249, hcm0031...) — KHÔNG đánh số
#      │   ├── <image_name>.png     # tên ảnh GỐC trong CSV, đổi đuôi .JPG -> .png
#      │   └── ...
#
# ⚠️ GIỚI HẠN DUNG LƯỢNG: BTC giới hạn 350MB. PNG lossless 1320×989 cho 434 ảnh
#    ~800MB -> VƯỢT. Dùng cờ nén:
#      --format jpg --quality 95    : xuất JPG q95 (giảm ~5-8×, gần như không mất
#                                     chất lượng cảm quan). Đuôi .jpg trong zip.
#      --format png-optimize        : nén PNG (PIL optimize; nếu có pngquant thì
#                                     dùng pngquant giảm mạnh hơn). Giữ đuôi .png.
#      --format png                 : (mặc định) copy PNG gốc, không nén lại.
#
# Ví dụ:
#   python comp/make_submission.py --renders outputs/renders_private \
#       --out submission/submission.zip --format jpg --quality 95
#

import os
import sys
import io
import shutil
import zipfile
import argparse
import subprocess

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


def _encode_jpg(src_path, quality):
    """Đọc ảnh -> trả về bytes JPG chất lượng `quality`."""
    from PIL import Image
    im = Image.open(src_path).convert("RGB")
    buf = io.BytesIO()
    # subsampling=0 (4:4:4) giữ màu tốt cho SSIM/LPIPS; KHÔNG dùng optimize
    # (một số phiên bản libjpeg lỗi "broken data stream" khi optimize+444).
    im.save(buf, format="JPEG", quality=quality, subsampling=0)
    return buf.getvalue()


def _encode_png_optimized(src_path):
    """Nén PNG (lossless PIL optimize; nếu có pngquant -> nén palette mạnh hơn).
    Trả về bytes PNG."""
    # Ưu tiên pngquant (giảm mạnh hơn nhiều) nếu có trên máy.
    if shutil.which("pngquant"):
        try:
            out = subprocess.run(
                ["pngquant", "--quality=80-95", "--speed", "1", "-", ],
                input=open(src_path, "rb").read(),
                stdout=subprocess.PIPE, check=True,
            )
            return out.stdout
        except Exception:
            pass  # fallback PIL
    from PIL import Image
    im = Image.open(src_path)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def make_zip(renders_root, out_zip, fmt="png", quality=95,
             ext_override=None, lower_scene=False):
    scenes = collect(renders_root)
    if not scenes:
        raise SystemExit(f"[make_submission] Không thấy ảnh nào trong {renders_root}")

    # Quyết định đuôi file trong zip.
    if ext_override:
        ext = ext_override if ext_override.startswith(".") else "." + ext_override
    elif fmt == "jpg":
        ext = ".jpg"
    else:
        ext = ".png"

    os.makedirs(os.path.dirname(os.path.abspath(out_zip)), exist_ok=True)
    total = 0
    # Ảnh đã nén sẵn -> ZIP_STORED (không nén lại, nhanh & không phình).
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_STORED) as zf:
        for scene, imgs in scenes.items():
            arc_scene = scene.lower() if lower_scene else scene
            for img in imgs:
                stem, _ = os.path.splitext(os.path.basename(img))
                arc = f"{arc_scene}/{stem}{ext}"
                if fmt == "jpg":
                    zf.writestr(arc, _encode_jpg(img, quality))
                elif fmt == "png-optimize":
                    zf.writestr(arc, _encode_png_optimized(img))
                else:  # png: copy nguyên gốc
                    zf.write(img, arcname=arc)
                total += 1
            print(f"  {arc_scene}: {len(imgs)} ảnh")

    size_mb = os.path.getsize(out_zip) / 1e6
    limit = 350
    status = "✅ dưới giới hạn" if size_mb <= limit else f"❌ VƯỢT {limit}MB"
    print(f"[make_submission] {len(scenes)} scene, {total} ảnh, fmt={fmt} "
          f"-> {out_zip} ({size_mb:.1f} MB) {status}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="thư mục gốc chứa <scene>/*.png")
    ap.add_argument("--out", required=True, help="đường dẫn submission.zip")
    ap.add_argument("--format", dest="fmt", default="png",
                    choices=["png", "jpg", "png-optimize"],
                    help="png=copy gốc; jpg=nén JPG; png-optimize=nén PNG (pngquant nếu có)")
    ap.add_argument("--quality", type=int, default=95, help="chất lượng JPG (mặc định 95)")
    ap.add_argument("--ext", dest="ext_override", default=None,
                    help="ép đuôi file trong zip (vd 'png' dù nội dung là jpg)")
    ap.add_argument("--lower-scene", action="store_true", help="hạ tên scene về chữ thường")
    args = ap.parse_args()
    make_zip(args.renders, args.out, fmt=args.fmt, quality=args.quality,
             ext_override=args.ext_override, lower_scene=args.lower_scene)
