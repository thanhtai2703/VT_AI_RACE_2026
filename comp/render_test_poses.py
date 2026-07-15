#
# comp/render_test_poses.py
#
# Render đúng các pose trong test_poses.csv (KHÔNG phải split mặc định của repo).
#
# Luồng:
#   1. Đọc CSV -> list Camera (comp/test_pose_loader.py, đúng convention COLMAP).
#   2. Load model .ply đã train (<model_path>/point_cloud/iteration_<N>/point_cloud.ply).
#   3. Gọi gaussian_renderer.render(...) cho từng camera.
#   4. Lưu PNG đúng tên image_name (đổi đuôi -> .png) và ĐÚNG W×H trong CSV.
#      -> KHÔNG resize sau render (tránh mất điểm SSIM/LPIPS).
#
# ⚠️ Cờ --antialiasing PHẢI khớp cờ đã dùng khi train.
#
# Ví dụ:
#   python comp/render_test_poses.py \
#       -m outputs/models/hcm0031 \
#       --poses phase1/public_set/hcm0031/test/test_poses.csv \
#       --out outputs/renders/hcm0031 \
#       --antialiasing
#

import os
import sys

# UTF-8 cho console Windows.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gaussian-splatting"))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torchvision
from tqdm import tqdm

from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state

# import loader theo đường dẫn để không lệ thuộc cwd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pose_loader import read_test_poses_csv, build_cameras, warn_principal_point  # noqa: E402

try:
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401
    SPARSE_ADAM_AVAILABLE = True
except Exception:
    SPARSE_ADAM_AVAILABLE = False


def find_ply(model_path, iteration):
    """Tìm point_cloud.ply cho iteration cho trước, hoặc iteration lớn nhất nếu -1."""
    pc_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(pc_root):
        raise FileNotFoundError(f"Không thấy {pc_root}. Đã train xong scene này chưa?")
    iters = []
    for d in os.listdir(pc_root):
        if d.startswith("iteration_") and os.path.isfile(os.path.join(pc_root, d, "point_cloud.ply")):
            try:
                iters.append(int(d.split("_")[1]))
            except ValueError:
                pass
    if not iters:
        raise FileNotFoundError(f"Không thấy point_cloud.ply nào trong {pc_root}")
    chosen = max(iters) if iteration < 0 else iteration
    if chosen not in iters:
        raise FileNotFoundError(f"Không có iteration {chosen} trong {pc_root} (có: {sorted(iters)})")
    return os.path.join(pc_root, f"iteration_{chosen}", "point_cloud.ply"), chosen


class _Pipe:
    """PipelineParams tối giản cho render (khớp field mà render() đọc)."""
    def __init__(self, antialiasing, mip_filter=False):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = antialiasing
        self.mip_filter = mip_filter


def render_test_poses(model_path, poses_csv, out_dir, sh_degree=3,
                      iteration=-1, antialiasing=False, white_background=False,
                      use_trained_exp=False, mip_filter=False, quiet=False):
    safe_state(quiet)

    poses = read_test_poses_csv(poses_csv)
    warn_principal_point(poses)          # cảnh báo nếu pp lệch tâm
    print(f"[render] {len(poses)} pose từ {poses_csv}")

    ply_path, used_iter = find_ply(model_path, iteration)
    print(f"[render] load model iteration {used_iter}: {ply_path}")

    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(ply_path, use_train_test_exp=use_trained_exp)

    bg = torch.tensor([1, 1, 1] if white_background else [0, 0, 0],
                      dtype=torch.float32, device="cuda")
    pipe = _Pipe(antialiasing, mip_filter=mip_filter)
    if mip_filter and gaussians.filter_3D.numel() == 0:
        print("[render] ⚠️ --mip_filter bật nhưng PLY không có filter_3D "
              "(model train KHÔNG bật mip). Render như baseline.")

    os.makedirs(out_dir, exist_ok=True)
    cameras = build_cameras(poses)

    with torch.no_grad():
        for cam, p in zip(tqdm(cameras, desc=f"Render {os.path.basename(model_path)}"), poses):
            image = render(cam, gaussians, pipe, bg,
                           use_trained_exp=use_trained_exp,
                           separate_sh=SPARSE_ADAM_AVAILABLE,
                           use_3D_filter=mip_filter)["render"]
            image = image.clamp(0.0, 1.0)
            # Lưu ĐÚNG tên (đổi đuôi -> .png) và đúng W×H (image đã là W×H của camera).
            out_name = os.path.splitext(p.image_name)[0] + ".png"
            torchvision.utils.save_image(image, os.path.join(out_dir, out_name))

    print(f"[render] xong -> {out_dir} ({len(cameras)} ảnh)")
    return len(cameras)


if __name__ == "__main__":
    from argparse import ArgumentParser
    ap = ArgumentParser("Render test poses theo CSV cuộc thi")
    ap.add_argument("-m", "--model_path", required=True)
    ap.add_argument("--poses", required=True, help="test_poses.csv")
    ap.add_argument("--out", required=True, help="thư mục lưu PNG")
    ap.add_argument("--iteration", type=int, default=-1, help="-1 = iteration lớn nhất")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--antialiasing", action="store_true", help="PHẢI khớp cờ khi train")
    ap.add_argument("--mip_filter", action="store_true", help="Mip-Splatting 3D filter (PHẢI khớp cờ khi train)")
    ap.add_argument("--white_background", action="store_true")
    ap.add_argument("--train_test_exp", action="store_true", help="nếu train có exposure comp")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    render_test_poses(
        model_path=args.model_path, poses_csv=args.poses, out_dir=args.out,
        sh_degree=args.sh_degree, iteration=args.iteration,
        antialiasing=args.antialiasing, white_background=args.white_background,
        use_trained_exp=args.train_test_exp, mip_filter=args.mip_filter, quiet=args.quiet,
    )
