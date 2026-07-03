#
# comp/local_eval.py
#
# Tính Score cuộc thi trên public_set (có ground-truth ảnh test).
#
#   Score = 0.4 × (1 − LPIPS) + 0.3 × SSIM + 0.3 × PSNR_norm
#
# Dùng ĐÚNG 3 hàm metric của repo Inria (khớp metrics.py):
#   - ssim   : utils.loss_utils.ssim
#   - psnr   : utils.image_utils.psnr
#   - lpips  : lpipsPyTorch.lpips(net_type='vgg')   <- LPIPS nặng ký nhất (0.4)
#
# ⚠️ PSNR_norm: BTC CHƯA công bố cách chuẩn hoá. Mặc định giả định:
#       PSNR_norm = clip(PSNR / PSNR_MAX, 0, 1),  PSNR_MAX = 30 dB
#    Chỉnh --psnr_max cho khớp khi BTC công bố. Con số Score tuyệt đối chỉ để
#    SO SÁNH TƯƠNG ĐỐI giữa các cấu hình của ta; xếp hạng ít nhạy với PSNR_MAX.
#
# Ví dụ (so render với ảnh test thật của 1 scene public):
#   python comp/local_eval.py \
#       --renders outputs/renders/hcm0031 \
#       --gt phase1/public_set/hcm0031/test/images \
#       --psnr_max 30
#

import os
import sys
import argparse

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gaussian-splatting"))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torchvision.transforms.functional as tf
from PIL import Image

from utils.loss_utils import ssim as ssim_fn
from utils.image_utils import psnr as psnr_fn
from lpipsPyTorch import lpips as lpips_fn


def _load(path):
    im = Image.open(path).convert("RGB")
    return tf.to_tensor(im).unsqueeze(0)[:, :3].cuda()


def match_pairs(renders_dir, gt_dir):
    """Ghép ảnh render <-> gt theo stem (bỏ đuôi), vì gt là .JPG, render là .png."""
    def index(d):
        out = {}
        for f in os.listdir(d):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                out[os.path.splitext(f)[0]] = os.path.join(d, f)
        return out
    r, g = index(renders_dir), index(gt_dir)
    common = sorted(set(r) & set(g))
    only_r, only_g = set(r) - set(g), set(g) - set(r)
    if only_r:
        print(f"[WARN] {len(only_r)} render không có gt, vd: {sorted(only_r)[:3]}")
    if only_g:
        print(f"[WARN] {len(only_g)} gt không có render, vd: {sorted(only_g)[:3]}")
    return [(r[k], g[k]) for k in common]


def eval_scene(renders_dir, gt_dir, psnr_max=30.0):
    pairs = match_pairs(renders_dir, gt_dir)
    if not pairs:
        raise SystemExit(f"Không ghép được ảnh nào giữa {renders_dir} và {gt_dir}")

    ssims, psnrs, lpipss = [], [], []
    with torch.no_grad():
        for rp, gp in pairs:
            r, g = _load(rp), _load(gp)
            if r.shape != g.shape:
                # An toàn: nếu lệch size thì báo lỗi rõ (không tự resize -> tránh che lỗi).
                raise SystemExit(f"Lệch kích thước: {rp} {tuple(r.shape)} vs {gp} {tuple(g.shape)}")
            ssims.append(ssim_fn(r, g).item())
            psnrs.append(psnr_fn(r, g).mean().item())
            lpipss.append(lpips_fn(r, g, net_type="vgg").item())

    import statistics as st
    SSIM = st.fmean(ssims)
    PSNR = st.fmean(psnrs)
    LPIPS = st.fmean(lpipss)
    PSNR_norm = max(0.0, min(1.0, PSNR / psnr_max))
    Score = 0.4 * (1 - LPIPS) + 0.3 * SSIM + 0.3 * PSNR_norm
    return {
        "n": len(pairs), "SSIM": SSIM, "PSNR": PSNR, "LPIPS": LPIPS,
        "PSNR_norm": PSNR_norm, "Score": Score,
    }


def _print(scene, m):
    print(f"[{scene}] n={m['n']}  "
          f"LPIPS={m['LPIPS']:.4f}  SSIM={m['SSIM']:.4f}  "
          f"PSNR={m['PSNR']:.2f}dB (norm={m['PSNR_norm']:.4f})  "
          f"=> Score={m['Score']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, help="thư mục ảnh render (hoặc gốc chứa nhiều scene nếu --multi)")
    ap.add_argument("--gt", required=True, help="thư mục ảnh gt (hoặc data root nếu --multi)")
    ap.add_argument("--psnr_max", type=float, default=30.0)
    ap.add_argument("--multi", action="store_true",
                    help="renders=outputs/renders, gt=phase1/public_set -> eval mọi scene & tính Score trung bình")
    args = ap.parse_args()

    if not args.multi:
        m = eval_scene(args.renders, args.gt, args.psnr_max)
        _print(os.path.basename(args.renders.rstrip("/\\")), m)
    else:
        scores = []
        for scene in sorted(os.listdir(args.renders)):
            rdir = os.path.join(args.renders, scene)
            gdir = os.path.join(args.gt, scene, "test", "images")
            if not (os.path.isdir(rdir) and os.path.isdir(gdir)):
                continue
            m = eval_scene(rdir, gdir, args.psnr_max)
            _print(scene, m)
            scores.append(m["Score"])
        if scores:
            import statistics as st
            print(f"\n=== Score TRUNG BÌNH {len(scores)} scene: {st.fmean(scores):.4f} ===")
