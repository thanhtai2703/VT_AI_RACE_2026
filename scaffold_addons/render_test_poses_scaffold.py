#
# scaffold_addons/render_test_poses_scaffold.py
#
# Render các pose trong test_poses.csv bằng model Scaffold-GS đã train.
# ĐẶT FILE NÀY VÀO THƯ MỤC GỐC repo Scaffold-GS đã clone (cạnh render.py gốc)
# rồi chạy — nó import scene/gaussian_renderer/arguments của Scaffold.
#
# Tương đương comp/render_test_poses.py (bản Inria) nhưng dùng API Scaffold:
#   - Load model: GaussianModel(...) + Scene(dataset, gaussians, load_iteration)
#     qua get_combined_args (đọc lại cfg_args đã lưu lúc train -> đúng feat_dim,
#     n_offsets, voxel_size... KHÔNG cần gõ tay).
#   - gaussians.eval()  (MLP về eval mode — BẮT BUỘC, khác 3DGS thường).
#   - mỗi cam: mask = prefilter_voxel(...) -> render(..., visible_mask=mask).
#
# ⚠️ Convention Camera (R/T/FoV) của Scaffold KẾ THỪA y hệt Inria, nên phần dựng
#    camera từ CSV tái dùng logic test_pose_loader (copy kèm trong scaffold_addons).
#    Lưu ảnh theo TÊN GỐC trong CSV (đổi .JPG -> .png) để local_eval/make_submission
#    ghép được theo stem — KHÔNG lưu theo index 00001.png như render.py gốc.
#
# Cách dùng (trên máy thuê, trong thư mục Scaffold-GS/):
#   python render_test_poses_scaffold.py \
#       -m outputs/hcm0031 \
#       --poses <repo>/phase1/public_set/hcm0031/test/test_poses.csv \
#       --out  <repo>/outputs/scaffold_renders/hcm0031 \
#       --iteration 30000
#
# -m PHẢI là model_path Scaffold đã train (chứa cfg_args + point_cloud/iteration_N/).
#

import os
import sys

# UTF-8 cho console (an toàn Linux máy thuê).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import csv
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from argparse import ArgumentParser

# --- API Scaffold-GS (file này phải nằm trong repo Scaffold-GS) ---
from scene import Scene
from scene.cameras import Camera
from gaussian_renderer import render, prefilter_voxel, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import focal2fov
from scene.colmap_loader import qvec2rotmat


# ============================================================================
# Parse test_poses.csv (logic KHỚP comp/test_pose_loader.py — cùng convention).
# ============================================================================
REQUIRED_COLUMNS = ["image_name", "qw", "qx", "qy", "qz",
                    "tx", "ty", "tz", "fx", "fy", "cx", "cy", "width", "height"]


def read_test_poses_csv(csv_path):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Không thấy test_poses.csv: {csv_path}")
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV thiếu cột {missing}. Có: {reader.fieldnames}")
        for i, row in enumerate(reader):
            try:
                rows.append({
                    "image_name": row["image_name"].strip(),
                    "qvec": np.array([float(row["qw"]), float(row["qx"]),
                                      float(row["qy"]), float(row["qz"])], dtype=np.float64),
                    "tvec": np.array([float(row["tx"]), float(row["ty"]),
                                      float(row["tz"])], dtype=np.float64),
                    "fx": float(row["fx"]), "fy": float(row["fy"]),
                    "cx": float(row["cx"]), "cy": float(row["cy"]),
                    "width": int(round(float(row["width"]))),
                    "height": int(round(float(row["height"]))),
                })
            except (ValueError, KeyError) as e:
                raise ValueError(f"Dòng {i+2} trong {csv_path} lỗi: {e}")
    return rows


def warn_principal_point(rows, tol_px=1.0):
    ok = True
    for p in rows:
        if abs(p["cx"] - p["width"] / 2.0) > tol_px or abs(p["cy"] - p["height"] / 2.0) > tol_px:
            ok = False
            print(f"[WARN] {p['image_name']}: principal point lệch tâm "
                  f"(cx={p['cx']} vs {p['width']/2}, cy={p['cy']} vs {p['height']/2})")
    if ok:
        print("[render] principal point khớp tâm ảnh (OK).")
    return ok


def build_scaffold_cameras(rows, data_device="cuda"):
    """Dựng list Camera của Scaffold từ các dòng CSV.

    ⚠️ R = transpose(qvec2rotmat(qvec)), T = tvec — ĐÚNG như readColmapCameras.
    ⚠️ Camera của Scaffold nhận `image` là TENSOR CHW (đọc shape[1]=H, shape[2]=W),
       KHÁC bản Inria (nhận PIL + resolution). Ta đưa tensor đen đúng H×W làm
       placeholder resolution — nội dung không ảnh hưởng render (chỉ lấy kích thước).
    Chữ ký Scaffold (đã verify): Camera(colmap_id, R, T, FoVx, FoVy, image,
       gt_alpha_mask, image_name, uid, trans, scale, data_device).
    """
    cameras = []
    for uid, p in enumerate(rows):
        R = np.transpose(qvec2rotmat(p["qvec"]))     # KHÔNG quên transpose
        T = np.array(p["tvec"], dtype=np.float64)
        FovX = focal2fov(p["fx"], p["width"])
        FovY = focal2fov(p["fy"], p["height"])
        # Tensor đen CHW đúng H×W (float 0..1). Camera đọc shape[1]=H, shape[2]=W.
        placeholder = torch.zeros((3, p["height"], p["width"]), dtype=torch.float32)
        cam = Camera(
            colmap_id=uid, R=R, T=T, FoVx=FovX, FoVy=FovY,
            image=placeholder, gt_alpha_mask=None,
            image_name=p["image_name"], uid=uid,
            data_device=data_device,
        )
        cameras.append(cam)
    return cameras


# ============================================================================
# Render
# ============================================================================
def render_from_csv(dataset, pipeline, iteration, poses_csv, out_dir):
    rows = read_test_poses_csv(poses_csv)
    print(f"[render] {len(rows)} pose từ {poses_csv}")
    warn_principal_point(rows)

    with torch.no_grad():
        # Load model Scaffold: constructor + Scene đọc iteration đã lưu.
        gaussians = GaussianModel(
            dataset.feat_dim, dataset.n_offsets, dataset.voxel_size,
            dataset.update_depth, dataset.update_init_factor,
            dataset.update_hierachy_factor, dataset.use_feat_bank,
            dataset.appearance_dim, dataset.ratio,
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist,
        )
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()   # MLP eval mode — BẮT BUỘC
        print(f"[render] load model iteration {scene.loaded_iter} từ {dataset.model_path}")

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        os.makedirs(out_dir, exist_ok=True)
        cameras = build_scaffold_cameras(rows)

        for cam, p in zip(tqdm(cameras, desc=f"Render {os.path.basename(dataset.model_path)}"), rows):
            voxel_visible_mask = prefilter_voxel(cam, gaussians, pipeline, background)
            rendering = render(cam, gaussians, pipeline, background,
                               visible_mask=voxel_visible_mask)["render"]
            rendering = rendering.clamp(0.0, 1.0)
            # Lưu theo TÊN GỐC (đổi .JPG -> .png) — để ghép stem ở local_eval/make_submission.
            out_name = os.path.splitext(p["image_name"])[0] + ".png"
            torchvision.utils.save_image(rendering, os.path.join(out_dir, out_name))

    print(f"[render] xong -> {out_dir} ({len(cameras)} ảnh)")
    return len(cameras)


if __name__ == "__main__":
    parser = ArgumentParser("Render test poses (Scaffold-GS)")
    model = ModelParams(parser, sentinel=True)     # đọc cfg_args đã lưu qua get_combined_args
    pipeline = PipelineParams(parser)
    parser.add_argument("--poses", required=True, help="test_poses.csv")
    parser.add_argument("--out", required=True, help="thư mục lưu PNG (tên gốc)")
    parser.add_argument("--iteration", default=-1, type=int, help="-1 = iteration lớn nhất")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    render_from_csv(
        model.extract(args), pipeline.extract(args),
        args.iteration, args.poses, args.out,
    )
