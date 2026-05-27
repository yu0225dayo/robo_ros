"""
把持パイプライン (Client PC 側)

フロー:
    1. ROS PC の HTTP サーバから RGBD を取得
    2. GPU サーバ (SAM-3D + SAM-6D) で物体姿勢推定
    3. GraspGenerator (Shape2Gesture) で把持姿勢生成
    4. 手首座標 hand[0] を ROS PC へ HTTP POST → ROS トピックに転送される

依存:
    pip install requests opencv-python numpy

使用例:
    python grasp_pipeline.py \
        --ros-pc http://192.168.1.10:5000 \
        --gpu-server http://10.40.1.126:8080 \
        --click-x 320 --click-y 240

出力 ROS トピック (ROS PC 側):
    /wrist_left   geometry_msgs/PointStamped  左手首 カメラ座標系 [m]
    /wrist_right  geometry_msgs/PointStamped  右手首 カメラ座標系 [m]
"""

import argparse
import base64
import json
import os
import sys
import time

import cv2
import numpy as np
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from pipeline.grasp_generator import GraspGenerator
from utils.coord_transform import ObjectPose, normalized_to_camera


# --------------------------------------------------------------------------
# Step 1: ROS PC から RGBD を取得
# --------------------------------------------------------------------------

def get_rgbd(ros_pc_url: str) -> tuple:
    """
    ROS PC の HTTP サーバから最新 RGBD フレームを取得する

    Returns:
        bgr        : (H, W, 3) uint8  BGR 画像
        depth_mm   : (H, W) uint16   深度 [mm]
        intrinsics : dict  {fx, fy, cx, cy, depth_scale}
    """
    url = f"{ros_pc_url.rstrip('/')}/rgbd"
    print(f"[Step 1] RGBD 取得: GET {url}")
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    bgr      = cv2.imdecode(np.frombuffer(base64.b64decode(data["rgb_b64"]),  np.uint8),
                             cv2.IMREAD_COLOR)
    depth_mm = cv2.imdecode(np.frombuffer(base64.b64decode(data["depth_b64"]), np.uint8),
                             cv2.IMREAD_UNCHANGED).astype(np.uint16)
    intrinsics = data["intrinsics"]

    print(f"  rgb:   {bgr.shape}, depth: {depth_mm.shape}")
    print(f"  intr:  fx={intrinsics['fx']:.1f} fy={intrinsics['fy']:.1f} "
          f"cx={intrinsics['cx']:.1f} cy={intrinsics['cy']:.1f}")
    return bgr, depth_mm, intrinsics


# --------------------------------------------------------------------------
# Step 2: GPU サーバで姿勢推定
# --------------------------------------------------------------------------

def estimate_pose(
    gpu_server_url: str,
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    intrinsics: dict,
    click_x: int = -1,
    click_y: int = -1,
    timeout: float = 600.0,
) -> dict:
    """
    GPU サーバ (/full_pipeline) に RGBD を送り R, t, points を受け取る

    Returns:
        dict with keys: R (3x3), t (3,), points (N,3)
    """
    url = f"{gpu_server_url.rstrip('/')}/full_pipeline"
    print(f"[Step 2] 姿勢推定: POST {url}")

    _, rgb_buf = cv2.imencode(".png", bgr)
    _, dep_buf = cv2.imencode(".png", depth_mm)

    # intrinsics から depth_scale を除いて送る (サーバが期待するキーのみ)
    intr_for_server = {k: intrinsics[k] for k in ("fx", "fy", "cx", "cy")}

    files = {
        "image": ("rgb.png",   rgb_buf.tobytes(), "image/png"),
        "depth": ("depth.png", dep_buf.tobytes(), "image/png"),
    }
    data = {
        "intrinsics_json": json.dumps(intr_for_server),
        "click_x": click_x,
        "click_y": click_y,
    }

    t0 = time.time()
    resp = requests.post(url, files=files, data=data, timeout=timeout)
    resp.raise_for_status()
    result = resp.json()
    print(f"  完了 ({time.time() - t0:.1f}s)")

    R = np.array(result["R"], dtype=np.float64)           # (3, 3)
    t = np.array(result["t"], dtype=np.float64)           # (3,)  [m]
    points = np.array(result["points"], dtype=np.float64) # (N, 3)
    print(f"  t = [{t[0]*1000:.1f}, {t[1]*1000:.1f}, {t[2]*1000:.1f}] mm")
    print(f"  点群: {len(points)} 点")

    return {"R": R, "t": t, "points": points}


# --------------------------------------------------------------------------
# Step 3: GraspGenerator で把持姿勢生成
# --------------------------------------------------------------------------

def generate_grasp(
    points: np.ndarray,
    pose: ObjectPose,
    model_dir: str = "save_model",
    epoch: int = 69,
    num_samples: int = 1,
) -> tuple:
    """
    Shape2Gesture で把持姿勢を生成し、手首座標をカメラ座標系で返す

    Args:
        points     : (N, 3) 物体点群 (物体座標系, mm 単位)
        pose       : ObjectPose (R, t, scale)
        model_dir  : Shape2Gesture モデルディレクトリ
        epoch      : PositionVAE エポック
        num_samples: 生成サンプル数 (最初の 1 件のみ使用)

    Returns:
        left_wrist  : (3,) 左手首 カメラ座標系 [m]
        right_wrist : (3,) 右手首 カメラ座標系 [m]
        left_hand   : (23, 3) 左手全関節 カメラ座標系 [m]
        right_hand  : (23, 3) 右手全関節 カメラ座標系 [m]
    """
    print(f"[Step 3] GraspGenerator 起動 (epoch={epoch}, samples={num_samples})")
    generator = GraspGenerator(model_dir=model_dir, epoch=epoch)
    generator.load_models()

    results = generator.generate(points, num_samples=num_samples)
    lh_norm, rh_norm = results[0]   # (23, 3) 正規化座標

    lh_cam = normalized_to_camera(lh_norm, pose)  # (23, 3) カメラ座標 [m]
    rh_cam = normalized_to_camera(rh_norm, pose)

    left_wrist  = lh_cam[0]   # hand[0] = 手首関節
    right_wrist = rh_cam[0]

    print(f"  左手首  (カメラ座標): [{left_wrist[0]:.4f},  {left_wrist[1]:.4f},  {left_wrist[2]:.4f}] m")
    print(f"  右手首  (カメラ座標): [{right_wrist[0]:.4f}, {right_wrist[1]:.4f}, {right_wrist[2]:.4f}] m")

    return left_wrist, right_wrist, lh_cam, rh_cam


# --------------------------------------------------------------------------
# Step 4: 手首座標を ROS PC へ送信
# --------------------------------------------------------------------------

def send_wrist(ros_pc_url: str, left_wrist: np.ndarray, right_wrist: np.ndarray):
    """
    手首座標 [m] を ROS PC の HTTP サーバへ POST する。
    サーバ側で geometry_msgs/PointStamped として ROS トピックに転送される。

    Topics (ROS PC 側):
        /wrist_left   — 左手首
        /wrist_right  — 右手首
    """
    url = f"{ros_pc_url.rstrip('/')}/wrist"
    payload = {
        "left":  left_wrist.tolist(),
        "right": right_wrist.tolist(),
    }
    print(f"[Step 4] 手首座標送信: POST {url}")
    resp = requests.post(url, json=payload, timeout=5)
    resp.raise_for_status()
    print(f"  送信完了: {resp.json()}")


# --------------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="把持パイプライン (Client PC 側)")
    parser.add_argument(
        "--ros-pc", required=True,
        help="ROS PC の HTTP URL (例: http://192.168.1.10:5000)",
    )
    parser.add_argument(
        "--gpu-server", default="http://10.40.1.126:8080",
        help="GPU サーバの URL",
    )
    parser.add_argument("--click-x",     type=int,   default=-1,
                        help="物体クリック座標 X (-1: 未指定)")
    parser.add_argument("--click-y",     type=int,   default=-1,
                        help="物体クリック座標 Y (-1: 未指定)")
    parser.add_argument("--model-dir",   default="save_model",
                        help="Shape2Gesture モデルディレクトリ")
    parser.add_argument("--epoch",       type=int,   default=69)
    parser.add_argument("--num-samples", type=int,   default=1)
    parser.add_argument("--timeout",     type=float, default=600.0,
                        help="GPU サーバへのタイムアウト秒数")
    args = parser.parse_args()

    print("=" * 55)
    print("  把持パイプライン")
    print(f"  ROS PC    : {args.ros_pc}")
    print(f"  GPU Server: {args.gpu_server}")
    print("=" * 55)

    # Step 1: RGBD 取得
    bgr, depth_mm, intrinsics = get_rgbd(args.ros_pc)

    # Step 2: 姿勢推定
    pose_result = estimate_pose(
        args.gpu_server, bgr, depth_mm, intrinsics,
        click_x=args.click_x, click_y=args.click_y,
        timeout=args.timeout,
    )
    R      = pose_result["R"]
    t      = pose_result["t"]
    points = pose_result["points"]   # (N, 3) 物体点群

    # スケール計算: normalize_pointcloud と同じ max(bbox辺)/2 [mm→m]
    centered     = points - points.mean(axis=0)
    _bbox_ext    = centered.max(axis=0) - centered.min(axis=0)
    mesh_scale_m = float(np.max(_bbox_ext)) / 1000.0
    pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)
    print(f"  mesh_scale_m = {mesh_scale_m:.4f} m")

    # Step 3: 把持姿勢生成
    left_wrist, right_wrist, _, _ = generate_grasp(
        points, pose,
        model_dir=args.model_dir,
        epoch=args.epoch,
        num_samples=args.num_samples,
    )

    # Step 4: ROS PC へ送信
    send_wrist(args.ros_pc, left_wrist, right_wrist)

    print("\n" + "=" * 55)
    print("  完了")
    print(f"  左手首: [{left_wrist[0]:.4f}, {left_wrist[1]:.4f}, {left_wrist[2]:.4f}] m")
    print(f"  右手首: [{right_wrist[0]:.4f}, {right_wrist[1]:.4f}, {right_wrist[2]:.4f}] m")
    print("=" * 55)


if __name__ == "__main__":
    main()
