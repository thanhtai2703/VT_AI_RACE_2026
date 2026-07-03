#
# comp/test_pose_loader.py
#
# Đọc test_poses.csv của cuộc thi -> dựng danh sách Camera của repo Inria.
#
# ⚠️ ĐÂY LÀ FILE DỄ SAI NHẤT. Convention phải khớp CHÍNH XÁC với repo Inria:
#   - CSV cho quaternion (qw,qx,qy,qz) và translation (tx,ty,tz) theo COLMAP,
#     tức là biến đổi WORLD -> CAMERA (giống extr.qvec / extr.tvec trong COLMAP).
#   - Repo dựng R,T như sau (scene/dataset_readers.py readColmapCameras):
#         R = np.transpose(qvec2rotmat(qvec))   # transpose vì code CUDA dùng glm (column-major)
#         T = np.array(tvec)
#     rồi Camera/getWorld2View2(R, T) tự lo phần còn lại.
#   - FoV: FovX = focal2fov(fx, width), FovY = focal2fov(fy, height).
#   - getProjectionMatrix của repo GIẢ ĐỊNH principal point ở tâm ảnh.
#     Với dữ liệu cuộc thi cx≈width/2, cy≈height/2 nên khớp; nếu lệch tâm nhiều
#     sẽ cần vá thêm (xem hàm warn_principal_point bên dưới).
#
# Cách dùng độc lập để kiểm tra parsing (không cần GPU cho phần đọc CSV):
#   python comp/test_pose_loader.py --csv phase1/public_set/hcm0031/test/test_poses.csv --check
#

import os
import sys
import csv
import numpy as np

# Windows console mặc định cp1252 -> không in được tiếng Việt / ký hiệu °.
# Ép UTF-8 để log không vỡ (an toàn trên cả Linux máy thuê).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- Cho phép import các util của repo Inria ---
_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gaussian-splatting")
_REPO = os.path.abspath(_REPO)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

def _load_module_from_file(mod_name, rel_path):
    """Load 1 module theo đường dẫn file, tránh kích hoạt scene/__init__.py
    (vốn import plyfile/torch — không cần cho phần parsing CSV)."""
    import importlib.util
    path = os.path.join(_REPO, rel_path)
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_colmap_loader = _load_module_from_file("_cl_colmap_loader", "scene/colmap_loader.py")
_graphics_utils = _load_module_from_file("_cl_graphics_utils", "utils/graphics_utils.py")
qvec2rotmat = _colmap_loader.qvec2rotmat
focal2fov = _graphics_utils.focal2fov


# Các cột bắt buộc trong CSV, theo README của BTC.
REQUIRED_COLUMNS = [
    "image_name", "qw", "qx", "qy", "qz",
    "tx", "ty", "tz", "fx", "fy", "cx", "cy", "width", "height",
]


class TestPose:
    """Một dòng test_poses.csv đã parse (chưa cần GPU)."""
    __slots__ = ("image_name", "qvec", "tvec", "fx", "fy", "cx", "cy", "width", "height")

    def __init__(self, image_name, qvec, tvec, fx, fy, cx, cy, width, height):
        self.image_name = image_name
        self.qvec = qvec            # np.array([qw,qx,qy,qz]) world->camera (COLMAP)
        self.tvec = tvec            # np.array([tx,ty,tz])   world->camera
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.width = int(round(float(width)))
        self.height = int(round(float(height)))

    def R_T(self):
        """Trả về (R, T) đúng convention Camera của repo Inria."""
        R = np.transpose(qvec2rotmat(self.qvec))   # KHÔNG được quên transpose
        T = np.array(self.tvec, dtype=np.float64)
        return R, T

    def fovs(self):
        FovX = focal2fov(self.fx, self.width)
        FovY = focal2fov(self.fy, self.height)
        return FovX, FovY


def read_test_poses_csv(csv_path):
    """Đọc file CSV -> list[TestPose]. Không phụ thuộc torch/CUDA."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Không thấy test_poses.csv: {csv_path}")

    poses = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(
                f"CSV thiếu cột {missing}. Có: {reader.fieldnames}"
            )
        for i, row in enumerate(reader):
            try:
                qvec = np.array([float(row["qw"]), float(row["qx"]),
                                 float(row["qy"]), float(row["qz"])], dtype=np.float64)
                tvec = np.array([float(row["tx"]), float(row["ty"]),
                                 float(row["tz"])], dtype=np.float64)
                poses.append(TestPose(
                    image_name=row["image_name"].strip(),
                    qvec=qvec, tvec=tvec,
                    fx=row["fx"], fy=row["fy"], cx=row["cx"], cy=row["cy"],
                    width=row["width"], height=row["height"],
                ))
            except (ValueError, KeyError) as e:
                raise ValueError(f"Dòng {i+2} trong {csv_path} lỗi: {e}")
    return poses


def warn_principal_point(poses, tol_px=1.0):
    """Cảnh báo nếu principal point lệch tâm > tol_px pixel.

    getProjectionMatrix của repo giả định pp ở tâm. Nếu lệch nhiều thì render
    sẽ sai lệch tịnh tiến. Trả về True nếu MỌI pose đều an toàn.
    """
    ok = True
    for p in poses:
        dx = abs(p.cx - p.width / 2.0)
        dy = abs(p.cy - p.height / 2.0)
        if dx > tol_px or dy > tol_px:
            ok = False
            print(f"[WARN] {p.image_name}: principal point lệch tâm "
                  f"(cx={p.cx} vs w/2={p.width/2}, cy={p.cy} vs h/2={p.height/2})")
    return ok


def build_cameras(poses, data_device="cuda"):
    """Dựng list Camera của repo Inria từ list[TestPose].

    Import torch/Camera ở đây (lazy) để phần đọc CSV vẫn chạy được không cần GPU.
    Camera của repo cần 1 ảnh để lấy resolution -> ta đưa ảnh đen đúng W×H
    (chỉ dùng làm placeholder resolution, không ảnh hưởng render).
    """
    import torch  # noqa: F401
    from PIL import Image
    from scene.cameras import Camera

    cameras = []
    for uid, p in enumerate(poses):
        R, T = p.R_T()
        FovX, FovY = p.fovs()
        placeholder = Image.new("RGB", (p.width, p.height))  # đúng W×H -> resolution đúng
        cam = Camera(
            resolution=(p.width, p.height),
            colmap_id=uid, R=R, T=T, FoVx=FovX, FoVy=FovY,
            depth_params=None, image=placeholder, invdepthmap=None,
            image_name=p.image_name, uid=uid,
            data_device=data_device,
        )
        cameras.append(cam)
    return cameras


def _self_check(csv_path):
    """Kiểm tra parsing + in tóm tắt, không cần GPU."""
    poses = read_test_poses_csv(csv_path)
    print(f"Đọc {len(poses)} pose từ {csv_path}")
    p0 = poses[0]
    R, T = p0.R_T()
    FovX, FovY = p0.fovs()
    print(f"  pose[0] image_name = {p0.image_name}")
    print(f"  W×H = {p0.width}×{p0.height}, fx={p0.fx}, fy={p0.fy}, cx={p0.cx}, cy={p0.cy}")
    print(f"  FovX={np.degrees(FovX):.2f}°, FovY={np.degrees(FovY):.2f}°")
    print(f"  det(R) = {np.linalg.det(R):.6f}  (phải ≈ +1 cho rotation hợp lệ)")
    print(f"  |q| = {np.linalg.norm(p0.qvec):.6f}  (phải ≈ 1)")
    all_ok = warn_principal_point(poses)
    print(f"  principal point {'OK (khớp tâm)' if all_ok else 'LỆCH TÂM — cần xử lý'}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="đường dẫn test_poses.csv")
    ap.add_argument("--check", action="store_true", help="chạy self-check parsing")
    args = ap.parse_args()
    _self_check(args.csv)
