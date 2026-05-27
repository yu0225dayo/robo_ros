#!/usr/bin/env python3
"""
sciurus17 把持コントロールパネル (tkinter GUI) — メインモード

ボタン1クリックで全工程を操作できる統合GUI。
カメラ映像を表示し、クリックで物体選択→計算機送信→把持推定→アーム移動まで対応。

起動:
  # 実機モード (ROS2 + sciurus17_ros 必須)
  python3 ros/sciurus17_gui_main.py --server-url http://10.40.1.126:8080

  # デモモード (ROS2 不要・GUI テスト用)
  python3 ros/sciurus17_gui_main.py --demo
  python3 ros/sciurus17_gui_main.py --demo --demo-image client/saved_data/test_20260422_200416

操作フロー:
  1. [sciurus17 起動] で ROS ノードを起動  (デモ時はスキップ可)
  2. [カメラ接続] でカメラ映像表示開始
  3. [フレーム取得] で映像を凍結
  4. [頭部制御] で首を指定角度に傾ける → 重力方向が自動計算される
  5. 凍結映像をクリックして物体選択
  6. [計算機へ送信・姿勢推定] でメッシュ生成 + 6DoF pose 推定
  7. [把持姿勢生成] で Shape2Gesture 実行 → hand[0] → TF 変換
  8. [両アームを移動] または [左/右アームのみ移動] で MoveItPy グラスプ実行

依存:
  pip install Pillow opencv-python numpy
  (実機モードのみ) ROS2 + MoveIt2 + sciurus17_ros + requests + torch
"""

import argparse
import math
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    print("Pillow が必要です: pip install Pillow")
    sys.exit(1)

# ── ROS2 (省略可能) ───────────────────────────────────────────────────────────
_ROS2_OK = _TF2_OK = _MOVEIT_OK = False
try:
    import rclpy
    import rclpy.duration
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Point, Pose, PointStamped, PoseStamped, Quaternion
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo
    from sensor_msgs.msg import Image as RosImage
    _ROS2_OK = True
except ImportError:
    pass

try:
    import tf2_geometry_msgs  # noqa: F401
    from tf2_ros import Buffer, TransformListener
    _TF2_OK = True
except ImportError:
    pass

try:
    from moveit.core.robot_state import RobotState
    from moveit.planning import MoveItPy, PlanRequestParameters
    _MOVEIT_OK = True
except ImportError:
    pass

# ── client/ モジュール (省略可能) ─────────────────────────────────────────────
_SERVER_OK = False
_GRASP_OK  = False

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "client"))

try:
    from pipeline.sam6d_detector import SAM6DClient
    from utils.coord_transform import CameraIntrinsics, ObjectPose, normalized_to_camera
    from utils.pointcloud_utils import load_pointcloud_ply
    _SERVER_OK = True
except ImportError as e:
    print(f"[情報] サーバ通信モジュール未読み込み: {e}")

try:
    from pipeline.grasp_generator import GraspGenerator
    _GRASP_OK = True
except ImportError as e:
    print(f"[情報] 把持生成モジュール未読み込み (torch なし?): {e}")

# ── 定数 ──────────────────────────────────────────────────────────────────────
CANVAS_W = 640
CANVAS_H = 480
LIVE_MS  = 50

S_IDLE        = "idle"
S_CAMERA      = "camera"
S_CAPTURED    = "captured"
S_SELECTED    = "selected"
S_ESTIMATING  = "estimating"
S_GRASP_READY = "grasp_ready"
S_MOVING      = "moving"
S_DONE        = "done"

BG       = "#2b2b2b"
PANEL_BG = "#3c3f41"
BTN_BG   = "#555759"
BTN_FG   = "white"
LOG_BG   = "#1e1e1e"
LOG_FG   = "#cccccc"


# ── ユーティリティ ─────────────────────────────────────────────────────────────

def _gravity_from_neck_pitch(neck_pitch_deg: float) -> np.ndarray:
    """
    首ピッチ角（下向き正）からカメラ座標系での重力方向を計算する。
    カメラ慣例: Y-up / Z-前方 / 水平時の重力方向 = [0, -1, 0]
    """
    theta = math.radians(neck_pitch_deg)
    return np.array([0.0, -math.cos(theta), -math.sin(theta)], dtype=np.float32)


def _align_from_gravity(pts: np.ndarray,
                        gravity_cam: np.ndarray | None = None):
    """
    重力方向ベクトルを使って点群を Y-down に揃える回転を返す。
    gravity_cam が None の場合は固定補正 diag([1,-1,-1]) を使用。
    """
    if gravity_cam is None:
        R = np.diag([1.0, -1.0, -1.0])
    else:
        g = gravity_cam / np.linalg.norm(gravity_cam)
        target = np.array([0.0, -1.0, 0.0])
        v = np.cross(g, target)
        s = float(np.linalg.norm(v))
        c = float(np.dot(g, target))
        if s < 1e-9:
            R = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
        else:
            vx = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]], dtype=np.float64)
            R = np.eye(3) + vx + vx @ vx * (1.0 - c) / (s ** 2)
    R = np.asarray(R, dtype=np.float64)
    return (R @ pts.T).T.astype(pts.dtype), R


def _project(xyz, intr):
    """3D点をカメラ内部パラメータで2D画素に投影する。z<=0 なら None。"""
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    if z <= 1e-6:
        return None
    return (int(intr.fx * x / z + intr.cx),
            int(intr.fy * y / z + intr.cy))


def _draw_pose_axes(rgb_bgr: np.ndarray, R: np.ndarray,
                    t: np.ndarray, intr, axis_len: float = 0.08):
    """推定姿勢の座標軸 (X=赤, Y=緑, Z=青) を画像に描画する。"""
    out = rgb_bgr.copy()
    if intr is None:
        return out
    origin = _project(t, intr)
    if origin is None:
        return out
    for i, (label, color) in enumerate([
        ("X", (0,   0,   255)),
        ("Y", (0,   255, 0  )),
        ("Z", (255, 0,   0  )),
    ]):
        tip = _project(np.asarray(t) + np.asarray(R)[:, i] * axis_len, intr)
        if tip:
            cv2.arrowedLine(out, origin, tip, color, 2, tipLength=0.3)
            cv2.putText(out, label, (tip[0] + 4, tip[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.circle(out, origin, 6, (255, 255, 255), -1)
    return out


# Shape2Gesture 23関節の骨接続
# handinf=[0,1,2,3,4,18, 0,5,6,7,19, 0,8,9,10,20, 0,11,12,13,21, 0,14,15,16,17,22]
# 0 がセパレータ(手首)、18〜22 が各指先
_HAND_CONNECTIONS = [
    (0, 1),  (1, 2),  (2, 3),  (3, 4),  (4, 18),   # 親指
    (0, 5),  (5, 6),  (6, 7),  (7, 19),             # 人差し指
    (0, 8),  (8, 9),  (9, 10), (10, 20),            # 中指
    (0, 11), (11,12), (12,13), (13,21),             # 薬指
    (0, 14), (14,15), (15,16), (16,17), (17,22),   # 小指
]


def _draw_hand_skeleton(rgb_bgr: np.ndarray, lh_cam: np.ndarray,
                         rh_cam: np.ndarray, intr) -> np.ndarray:
    """左右の手の骨格 (23関節) をカメラ座標から画像に投影して描画する。"""
    out = rgb_bgr.copy()
    if intr is None:
        return out
    lh_cam = np.asarray(lh_cam, dtype=np.float64)
    rh_cam = np.asarray(rh_cam, dtype=np.float64)
    for joints, color_bone, color_joint, label in [
        (lh_cam, (0,   0, 200), (0,   0, 255), "L"),  # 左手: 赤
        (rh_cam, (180, 0,   0), (255, 0,   0), "R"),  # 右手: 青
    ]:
        if joints.ndim != 2:
            continue
        pts2d = [_project(joints[i], intr) for i in range(len(joints))]
        for i, j in _HAND_CONNECTIONS:
            if i < len(pts2d) and j < len(pts2d) and pts2d[i] and pts2d[j]:
                cv2.line(out, pts2d[i], pts2d[j], color_bone, 2, cv2.LINE_AA)
        for k, p in enumerate(pts2d):
            if p:
                r = 6 if k == 0 else 3
                cv2.circle(out, p, r, color_joint, -1, cv2.LINE_AA)
        if pts2d[0]:
            cv2.putText(out, label, (pts2d[0][0] + 8, pts2d[0][1] + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_joint, 2)
    return out


# ── 可視化ウィンドウ (4パネル固定) ───────────────────────────────────────────────

class VisualizationWindow:
    """
    パイプライン各ステップの画像を常時表示する4パネルウィンドウ。

      [ カメラ映像  ] [ 物体選択     ]
      [ 姿勢推定結果 ] [ 把持姿勢結果 ]
    """

    PW, PH = 320, 240

    _PANELS = [
        ("camera", "カメラ映像",   0, 0),
        ("frozen", "物体選択",     0, 1),
        ("pose",   "姿勢推定結果", 1, 0),
        ("grasp",  "把持姿勢結果", 1, 1),
    ]

    def __init__(self, parent: tk.Tk):
        self.win = tk.Toplevel(parent)
        self.win.title("推定結果ビューア")
        self.win.configure(bg=BG)
        self.win.resizable(False, False)
        self.win.protocol("WM_DELETE_WINDOW", self.win.withdraw)

        self._canvases: dict[str, tk.Canvas] = {}
        self._tk_imgs:  dict[str, object]    = {}

        for key, label, row, col in self._PANELS:
            frame = tk.Frame(self.win, bg=BG)
            frame.grid(row=row, column=col, padx=4, pady=(4, 0))
            tk.Label(frame, text=label, bg=BG, fg="#bbbbbb",
                     font=("Helvetica", 9, "bold")).pack(anchor="w")
            c = tk.Canvas(frame, width=self.PW, height=self.PH, bg="#111111")
            c.pack()
            c.create_text(self.PW // 2, self.PH // 2, text="待機中",
                          fill="#444444", font=("Helvetica", 12))
            self._canvases[key] = c
            self._tk_imgs[key]  = None

    def _show(self, key: str, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(self.PW / w, self.PH / h)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (nw, nh))
        pad = np.zeros((self.PH, self.PW, 3), dtype=np.uint8)
        y0 = (self.PH - nh) // 2
        x0 = (self.PW - nw) // 2
        pad[y0:y0+nh, x0:x0+nw] = rgb
        tk_img = ImageTk.PhotoImage(Image.fromarray(pad))
        self._tk_imgs[key] = tk_img
        c = self._canvases[key]
        c.delete("all")
        c.create_image(0, 0, anchor="nw", image=tk_img)

    def update_camera(self, bgr: np.ndarray):
        self._show("camera", bgr)

    def update_frozen(self, bgr: np.ndarray,
                      click_x: int = -1, click_y: int = -1):
        img = bgr.copy()
        if click_x >= 0:
            cv2.circle(img, (click_x, click_y), 14, (80, 80, 255), 2, cv2.LINE_AA)
            cv2.circle(img, (click_x, click_y),  4, (80, 80, 255), -1)
        cv2.putText(img, "FROZEN", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 80), 2)
        self._show("frozen", img)

    def update_pose(self, bgr: np.ndarray, R: np.ndarray,
                    t: np.ndarray, intr):
        img = _draw_pose_axes(bgr, R, t, intr)
        t_txt = f"t [{t[0]:+.3f} {t[1]:+.3f} {t[2]:+.3f}]"
        cv2.putText(img, t_txt, (6, img.shape[0] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        self._show("pose", img)

    def update_grasp(self, bgr: np.ndarray, lh_cam: np.ndarray,
                     rh_cam: np.ndarray, intr):
        lh_cam = np.asarray(lh_cam, dtype=np.float64)
        rh_cam = np.asarray(rh_cam, dtype=np.float64)
        img = _draw_hand_skeleton(bgr, lh_cam, rh_cam, intr)
        lh_wrist = lh_cam[0] if lh_cam.ndim == 2 else lh_cam
        rh_wrist = rh_cam[0] if rh_cam.ndim == 2 else rh_cam
        for i, (v, lbl) in enumerate([(lh_wrist, "L"), (rh_wrist, "R")]):
            cv2.putText(img,
                        f"{lbl}[{v[0]:+.2f} {v[1]:+.2f} {v[2]:+.2f}]",
                        (6, img.shape[0] - 8 - i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        self._show("grasp", img)


def _plan_and_execute(robot, comp, log_fn, params) -> bool:
    result = comp.plan(single_plan_parameters=params)
    if result:
        robot.execute(result.trajectory, controllers=[])
        return True
    log_fn("軌道計画失敗")
    return False


# ── ROS2 実機ノード ────────────────────────────────────────────────────────────

if _ROS2_OK:
    class RobotNode(Node):
        """カメラ購読 + TF2 変換 + アーム制御を担う ROS2 ノード。"""

        def __init__(self, color_topic, depth_topic, info_topic):
            super().__init__("sciurus17_gui_node")
            self._bridge = CvBridge()
            self._lock   = threading.Lock()
            self._rgb: np.ndarray | None      = None
            self._depth_mm: np.ndarray | None = None
            self._intrinsics                  = None

            if _TF2_OK:
                self._tf_buffer   = Buffer()
                self._tf_listener = TransformListener(self._tf_buffer, self)

            self.create_subscription(RosImage,   color_topic, self._color_cb, 1)
            self.create_subscription(RosImage,   depth_topic, self._depth_cb, 1)
            self.create_subscription(CameraInfo, info_topic,  self._info_cb,  1)

        def _color_cb(self, msg):
            try:
                bgr = self._bridge.imgmsg_to_cv2(msg, "bgr8")
                with self._lock:
                    self._rgb = bgr
            except Exception as e:
                self.get_logger().error(f"color_cb: {e}")

        def _depth_cb(self, msg):
            try:
                d = self._bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                with self._lock:
                    self._depth_mm = d.copy()
            except Exception as e:
                self.get_logger().error(f"depth_cb: {e}")

        def _info_cb(self, msg):
            with self._lock:
                if self._intrinsics is None and _SERVER_OK:
                    self._intrinsics = CameraIntrinsics(
                        fx=msg.k[0], fy=msg.k[4],
                        cx=msg.k[2], cy=msg.k[5],
                        width=msg.width, height=msg.height,
                    )

        def get_latest_rgb(self):
            with self._lock:
                return self._rgb.copy() if self._rgb is not None else None

        def get_snapshot(self):
            with self._lock:
                return (
                    self._rgb.copy()      if self._rgb is not None      else None,
                    self._depth_mm.copy() if self._depth_mm is not None else None,
                    self._intrinsics,
                )

        def camera_to_base(self, xyz_cam, camera_frame, base_frame):
            if not _TF2_OK:
                raise RuntimeError("tf2_geometry_msgs が利用できません")
            pt = PointStamped()
            pt.header.frame_id = camera_frame
            pt.header.stamp    = self.get_clock().now().to_msg()
            pt.point.x, pt.point.y, pt.point.z = (
                float(xyz_cam[0]), float(xyz_cam[1]), float(xyz_cam[2])
            )
            pt_b = self._tf_buffer.transform(
                pt, base_frame, timeout=rclpy.duration.Duration(seconds=2.0)
            )
            return np.array([pt_b.point.x, pt_b.point.y, pt_b.point.z])

        def move_arm(self, robot, arm_comp, params, xyz_base, pose_link, orientation):
            goal = PoseStamped()
            goal.header.frame_id = "base_link"
            goal.pose = Pose(
                position=Point(x=float(xyz_base[0]),
                               y=float(xyz_base[1]),
                               z=float(xyz_base[2])),
                orientation=orientation,
            )
            arm_comp.set_start_state_to_current_state()
            arm_comp.set_goal_state(pose_stamped_msg=goal, pose_link=pose_link)
            return _plan_and_execute(robot, arm_comp, self.get_logger().error, params)


# ── デモノード (ROS2 不要) ────────────────────────────────────────────────────

class DemoRobotNode:
    """
    デモ用ノード。ROS2 なしで動作する。
    - カメラ: ウェブカム → 保存画像 → 単色フォールバック
    - TF2: 固定オフセットで代替
    - アーム/頭部制御: ログ出力のみ
    """

    def __init__(self, mock_image_dir: str | None = None):
        self._lock      = threading.Lock()
        self._rgb       = None
        self._depth_mm  = None
        self._cam_json  = None
        self._cap       = None

        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            self._cap = cap
            print("[Demo] ウェブカム (device 0) を使用します")
            return
        cap.release()

        dirs_to_try = []
        if mock_image_dir:
            dirs_to_try.append(mock_image_dir)
        for rel in (
            "client/saved_data/test_20260422_200416",
            "client/saved_data/test_20260422_200438",
            "client/captured/20260401_221353",
        ):
            dirs_to_try.append(os.path.join(_REPO_ROOT, rel))

        for d in dirs_to_try:
            p = os.path.join(d, "rgb.png")
            if os.path.exists(p):
                img = cv2.imread(p)
                if img is not None:
                    with self._lock:
                        self._rgb = img
                    print(f"[Demo] テスト画像: {p}")
                    dp = os.path.join(d, "depth.png")
                    if os.path.exists(dp):
                        self._depth_mm = cv2.imread(dp, cv2.IMREAD_ANYDEPTH)
                    import json
                    cp = os.path.join(d, "cam.json")
                    if os.path.exists(cp):
                        with open(cp) as f:
                            self._cam_json = json.load(f)
                    return

        self._rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        self._rgb[:] = (50, 50, 80)
        cv2.putText(self._rgb, "Mock Camera — No Image Found",
                    (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
        print("[Demo] テスト画像なし。カラー画像でフォールバック")

    def _read_webcam(self):
        if self._cap is not None and self._cap.isOpened():
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._rgb = frame

    def get_latest_rgb(self) -> np.ndarray | None:
        self._read_webcam()
        with self._lock:
            return self._rgb.copy() if self._rgb is not None else None

    def get_snapshot(self) -> tuple:
        rgb = self.get_latest_rgb()
        h, w = rgb.shape[:2] if rgb is not None else (480, 640)

        if self._depth_mm is not None:
            dep = self._depth_mm
        else:
            dep = (np.random.rand(h, w) * 1000).astype(np.uint16)

        intr = None
        if _SERVER_OK:
            if self._cam_json:
                K = self._cam_json["cam_K"]
                intr = CameraIntrinsics(
                    fx=K[0], fy=K[4], cx=K[2], cy=K[5],
                    width=self._cam_json.get("width", w),
                    height=self._cam_json.get("height", h),
                )
            else:
                intr = CameraIntrinsics(fx=600.0, fy=600.0,
                                        cx=w / 2, cy=h / 2,
                                        width=w, height=h)
        return rgb, dep, intr

    def camera_to_base(self, xyz_cam, camera_frame, base_frame) -> np.ndarray:
        return xyz_cam + np.array([0.30, 0.0, 0.20])

    def move_arm(self, robot, arm_comp, params, xyz_base, pose_link, orientation) -> bool:
        time.sleep(0.5)
        return True

    def __del__(self):
        if self._cap is not None:
            self._cap.release()


# ── GUI ───────────────────────────────────────────────────────────────────────

class SciurusGUI:
    """sciurus17 把持コントロールパネル (tkinter)"""

    def __init__(self, root: tk.Tk, node, args):
        self.root = root
        self.node = node
        self.args = args

        self._state   = S_IDLE
        self._working = False

        self._captured_rgb:   np.ndarray | None = None
        self._captured_depth: np.ndarray | None = None
        self._intrinsics                        = None
        self._click_x = self._click_y = -1
        self._scale_x = self._scale_y = 1.0

        self._mesh_path: str | None      = None
        self._sam6d_client               = None
        self._R: np.ndarray | None       = None
        self._t: np.ndarray | None       = None
        self._gravity: np.ndarray | None = None
        self._lh_cam:  np.ndarray | None = None
        self._rh_cam:  np.ndarray | None = None
        self._lh_base: np.ndarray | None = None
        self._rh_base: np.ndarray | None = None
        self._robot                      = None
        self._process: subprocess.Popen | None = None
        self._vis_win: VisualizationWindow | None = None

        self._log_q: queue.Queue     = queue.Queue()
        self._btns: dict[str, tk.Button] = {}

        self._build_ui()

        if args.demo:
            self._log(f"[モード] デモモード — カメラ:保存画像 / 推定:実サーバ {args.server_url}")
        else:
            self._log("[モード] メインモード — sciurus17 実機")
        # 首ピッチ角から常に重力方向を計算してセット
        self._on_neck_pitch_change()

        self._refresh_ui()
        self._open_vis_win()
        self._poll_live()
        self._poll_log()

    def _open_vis_win(self):
        self._vis_win = VisualizationWindow(self.root)

    def _on_neck_pitch_change(self, *_):
        """首ピッチ角から重力方向を計算して self._gravity にセットし、ラベルを更新する。"""
        pitch = self._neck_pitch_var.get()
        self._gravity = _gravity_from_neck_pitch(pitch)
        g = self._gravity
        txt = f"[{g[0]:+.3f}  {g[1]:+.3f}  {g[2]:+.3f}]"
        if self._lbl_gravity is not None:
            self._lbl_gravity.config(text=txt)
        self._log(f"[重力] 首ピッチ {pitch:.1f}° → gravity={g.round(3)}")

    # ──────────────────────── UI 構築 ──────────────────────────────────────────

    def _build_ui(self):
        root = self.root
        if self.args.demo:
            title_color = "#ffcc44"
            title_text  = "sciurus17 把持コントロールパネル  【デモモード】"
            root.title("sciurus17 把持コントロールパネル  [DEMO]")
        else:
            title_color = "white"
            title_text  = "sciurus17 把持コントロールパネル  【メインモード】"
            root.title("sciurus17 把持コントロールパネル  [MAIN]")
        root.configure(bg=BG)
        root.resizable(False, False)

        tk.Label(root, text=title_text,
                 font=("Helvetica", 13, "bold"), bg=BG, fg=title_color,
                 ).grid(row=0, column=0, columnspan=2, pady=(8, 2), padx=10, sticky="ew")

        # ── 左: ボタンパネル ──
        left = tk.Frame(root, bg=BG, width=280)
        left.grid(row=1, column=0, sticky="ns", padx=(8, 4), pady=4)
        left.grid_propagate(False)

        def section(title):
            f = tk.LabelFrame(left, text=title, bg=PANEL_BG, fg="#bbbbbb",
                              font=("Helvetica", 9, "bold"),
                              padx=6, pady=4, relief=tk.GROOVE)
            f.pack(fill=tk.X, pady=3)
            return f

        def add_btn(parent, label, cmd, key, *, color=BTN_BG):
            b = tk.Button(parent, text=label, command=cmd,
                          bg=color, fg=BTN_FG,
                          activebackground="#6e6e6e", activeforeground="white",
                          relief=tk.FLAT, font=("Helvetica", 10),
                          padx=4, pady=4, width=24, anchor="w")
            b.pack(fill=tk.X, pady=2)
            self._btns[key] = b
            return b

        # ── 1. 起動 ──
        f1 = section("1. 起動")
        add_btn(f1, "▶  sciurus17 起動", self._on_launch, "launch", color="#3d6b3d")
        add_btn(f1, "■  sciurus17 停止", self._on_kill,   "kill",   color="#6b3d3d")

        # ── 2. カメラ ──
        f2 = section("2. カメラ")
        add_btn(f2, "▶  カメラ接続",    self._on_connect_camera, "cam_connect")
        add_btn(f2, "↓  首を動かす",    self._on_head_move,      "head_move2")
        add_btn(f2, "フレーム取得",      self._on_capture,        "capture")

        # ── 3. 頭部制御 ──
        f3 = section("3. 頭部制御")

        # ヨー行
        yaw_row = tk.Frame(f3, bg=PANEL_BG)
        yaw_row.pack(fill=tk.X, pady=2)
        tk.Label(yaw_row, text="ヨー[deg]:", bg=PANEL_BG, fg="#aaaaaa",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        self._neck_yaw_var = tk.DoubleVar(value=0.0)
        tk.Spinbox(
            yaw_row, textvariable=self._neck_yaw_var,
            from_=-90.0, to=90.0, increment=5.0, width=6,
            bg=BTN_BG, fg=BTN_FG, font=("Helvetica", 9),
        ).pack(side=tk.LEFT, padx=4)
        tk.Label(yaw_row, text="(左+)", bg=PANEL_BG, fg="#666666",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)

        # ピッチ行 (重力計算・高さ計測に連動)
        pitch_row = tk.Frame(f3, bg=PANEL_BG)
        pitch_row.pack(fill=tk.X, pady=2)
        tk.Label(pitch_row, text="ピッチ[deg]:", bg=PANEL_BG, fg="#aaaaaa",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        self._neck_pitch_var = tk.DoubleVar(value=getattr(self.args, "neck_pitch", 0.0))
        tk.Spinbox(
            pitch_row, textvariable=self._neck_pitch_var,
            from_=-90.0, to=90.0, increment=5.0, width=6,
            bg=BTN_BG, fg=BTN_FG, font=("Helvetica", 9),
            command=self._on_neck_pitch_change,
        ).pack(side=tk.LEFT, padx=4)
        tk.Label(pitch_row, text="(下+ → 重力方向・高さ計測に使用)",
                 bg=PANEL_BG, fg="#666666", font=("Helvetica", 8)).pack(side=tk.LEFT)

        add_btn(f3, "↓  首を動かす",    self._on_head_move, "head_move")
        add_btn(f3, "↑  首を初期位置へ", self._on_head_home, "head_home")

        # 重力方向表示
        grav_row = tk.Frame(f3, bg=PANEL_BG)
        grav_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(grav_row, text="重力方向:", bg=PANEL_BG, fg="#aaaaaa",
                 font=("Helvetica", 8)).pack(side=tk.LEFT)
        self._lbl_gravity = tk.Label(grav_row, text="計算中...",
                                      bg=PANEL_BG, fg="#88aaff",
                                      font=("Courier", 8))
        self._lbl_gravity.pack(side=tk.LEFT, padx=4)

        # ── 4. 姿勢推定 ──
        f4 = section("4. 姿勢推定")
        tk.Label(f4, text="↑ 画像をクリックして物体選択",
                 bg=PANEL_BG, fg="#888888", font=("Helvetica", 8)).pack(anchor="w")
        add_btn(f4, "→  計算機へ送信・姿勢推定", self._on_estimate, "estimate")
        add_btn(f4, "→  把持姿勢生成",           self._on_grasp,    "grasp")

        # ── 5. アーム制御 ──
        f5 = section("5. アーム制御")
        add_btn(f5, "▶  両アームを移動",   self._on_move,        "move",       color="#3d5b7a")
        add_btn(f5, "▶  左アームのみ移動", self._on_move_left,   "move_left",  color="#3a547a")
        add_btn(f5, "▶  右アームのみ移動", self._on_move_right,  "move_right", color="#3a547a")
        add_btn(f5, "○  グリッパ 開",     self._on_gripper_open,  "g_open")
        add_btn(f5, "●  グリッパ 閉",     self._on_gripper_close, "g_close")
        add_btn(f5, "⌂  初期姿勢へ",      self._on_home,          "home")

        # 座標表示
        fr = section("把持座標 (base_link) [m]")
        self._lbl_lh    = tk.Label(fr, text="左手首: -", bg=PANEL_BG, fg="#aaffaa",
                                    font=("Courier", 8), anchor="w")
        self._lbl_lh.pack(fill=tk.X)
        self._lbl_rh    = tk.Label(fr, text="右手首: -", bg=PANEL_BG, fg="#aaffaa",
                                    font=("Courier", 8), anchor="w")
        self._lbl_rh.pack(fill=tk.X)
        self._lbl_click = tk.Label(fr, text="選択点: -", bg=PANEL_BG, fg="#ffaaaa",
                                    font=("Courier", 8), anchor="w")
        self._lbl_click.pack(fill=tk.X)

        self._lbl_status = tk.Label(left, text="● 待機中",
                                     bg=BG, fg="#888888",
                                     font=("Helvetica", 9), anchor="w")
        self._lbl_status.pack(fill=tk.X, pady=(6, 0))

        # ── 右: カメラビュー + ログ ──
        right = tk.Frame(root, bg=BG)
        right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=4)

        cam_lf = tk.LabelFrame(right,
                                text="カメラ映像  ←  凍結後にクリックで物体選択",
                                bg=BG, fg="#bbbbbb", font=("Helvetica", 9, "bold"))
        cam_lf.pack(fill=tk.X)

        self._canvas = tk.Canvas(cam_lf, width=CANVAS_W, height=CANVAS_H, bg="black")
        self._canvas.pack()
        self._canvas.bind("<Button-1>", self._on_canvas_click)
        self._canvas_img_id = None

        log_lf = tk.LabelFrame(right, text="ログ",
                                bg=BG, fg="#bbbbbb", font=("Helvetica", 9, "bold"))
        log_lf.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self._log_area = scrolledtext.ScrolledText(
            log_lf, width=82, height=8,
            bg=LOG_BG, fg=LOG_FG, font=("Courier", 9),
            state=tk.DISABLED, wrap=tk.WORD,
        )
        self._log_area.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ──────────────────────── ログ ─────────────────────────────────────────────

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self._log_q.put(f"[{ts}] {msg}\n")

    def _poll_log(self):
        try:
            while True:
                msg = self._log_q.get_nowait()
                self._log_area.configure(state=tk.NORMAL)
                self._log_area.insert(tk.END, msg)
                self._log_area.see(tk.END)
                self._log_area.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)

    # ──────────────────────── ライブ映像 ───────────────────────────────────────

    def _poll_live(self):
        if self._state == S_CAMERA:
            frame = self.node.get_latest_rgb()
            if frame is not None:
                self._show_frame(frame, overlay_text="LIVE")
                if self._vis_win:
                    self._vis_win.update_camera(frame)
        self.root.after(LIVE_MS, self._poll_live)

    def _show_frame(self, bgr: np.ndarray,
                    markers: list | None = None,
                    overlay_text: str | None = None):
        h, w = bgr.shape[:2]
        self._scale_x = CANVAS_W / w
        self._scale_y = CANVAS_H / h
        disp = cv2.resize(bgr, (CANVAS_W, CANVAS_H))
        disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

        if overlay_text:
            cv2.putText(disp, overlay_text,
                        (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 220, 80), 2, cv2.LINE_AA)

        if markers:
            for (ix, iy, color) in markers:
                cx, cy = int(ix * self._scale_x), int(iy * self._scale_y)
                cv2.circle(disp, (cx, cy), 14, color, 2, cv2.LINE_AA)
                cv2.circle(disp, (cx, cy),  4, color, -1)
                cv2.putText(disp, f"({ix},{iy})", (cx + 16, cy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        pil = Image.fromarray(disp)
        self._tk_img = ImageTk.PhotoImage(pil)
        if self._canvas_img_id is None:
            self._canvas_img_id = self._canvas.create_image(
                0, 0, anchor="nw", image=self._tk_img)
        else:
            self._canvas.itemconfig(self._canvas_img_id, image=self._tk_img)

    # ──────────────────────── キャンバスクリック ────────────────────────────────

    def _on_canvas_click(self, event):
        if self._state not in (S_CAPTURED, S_SELECTED):
            return
        img_x = max(0, int(event.x / self._scale_x))
        img_y = max(0, int(event.y / self._scale_y))
        self._click_x, self._click_y = img_x, img_y
        self._lbl_click.config(text=f"選択点: ({img_x}, {img_y})")
        self._log(f"物体選択: ({img_x}, {img_y})")

        if self._captured_rgb is not None:
            self._show_frame(self._captured_rgb,
                             markers=[(img_x, img_y, (255, 80, 80))],
                             overlay_text="FROZEN")
            if self._vis_win:
                self._vis_win.update_frozen(self._captured_rgb, img_x, img_y)
        self._set_state(S_SELECTED)

    # ──────────────────────── ステート管理 ─────────────────────────────────────

    def _set_state(self, s: str):
        self._state = s
        self.root.after(0, self._refresh_ui)

    def _refresh_ui(self):
        s, w = self._state, self._working
        can_arm = not w and s not in (S_IDLE,)

        enabled = {
            "launch":      True,
            "kill":        True,
            "cam_connect": s == S_IDLE,
            "capture":     s == S_CAMERA and not w,
            "head_move":   can_arm,
            "head_move2":  can_arm,
            "head_home":   can_arm,
            "estimate":    s == S_SELECTED and not w,
            "grasp":       s == S_GRASP_READY and not w,
            "move":        s == S_DONE and not w,
            "move_left":   s == S_DONE and not w,
            "move_right":  s == S_DONE and not w,
            "g_open":      can_arm,
            "g_close":     can_arm,
            "home":        can_arm,
        }
        for key, btn in self._btns.items():
            btn.config(state=tk.NORMAL if enabled.get(key, False) else tk.DISABLED)

        STATUS = {
            S_IDLE:        ("● 待機中",                                "#888888"),
            S_CAMERA:      ("● カメラ接続中",                          "#88cc88"),
            S_CAPTURED:    ("● フレーム取得済 ─ 画像をクリックして選択", "#88aaff"),
            S_SELECTED:    ("● 物体選択済 ─ 計算機へ送信してください",    "#ffcc44"),
            S_ESTIMATING:  ("⏳ 推定中...",                             "#ffaa44"),
            S_GRASP_READY: ("● 姿勢推定済 ─ 把持姿勢生成してください",   "#aaffcc"),
            S_MOVING:      ("⏳ アーム移動中...",                       "#ff8844"),
            S_DONE:        ("✓ 完了 ─ アームを移動できます",             "#88ff88"),
        }
        text, color = STATUS.get(s, ("●", "white"))
        self._lbl_status.config(text=text, fg=color)
        self._canvas.config(cursor="crosshair" if s in (S_CAPTURED, S_SELECTED) else "")

    # ──────────────────────── バックグラウンド実行 ──────────────────────────────

    def _run_bg(self, func, *a, on_error_state: str | None = None):
        self._working = True
        self._refresh_ui()

        def wrapper():
            try:
                func(*a)
            except Exception as e:
                self._log(f"[エラー] {e}")
                if on_error_state is not None:
                    self.root.after(0, lambda: self._set_state(on_error_state))
            finally:
                self._working = False
                self.root.after(0, self._refresh_ui)

        threading.Thread(target=wrapper, daemon=True).start()

    # ──────────────────────── ボタンハンドラ ───────────────────────────────────

    def _on_launch(self):
        cmd = self.args.launch_cmd
        if not cmd:
            self._log(
                "[起動] --launch-cmd が未指定です。\n"
                "  例: --launch-cmd 'ros2 launch sciurus17_bringup sciurus17_bringup.launch.py'"
            )
            return
        self._log(f"[起動] {cmd}")
        try:
            self._process = subprocess.Popen(
                cmd, shell=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            def _read():
                for line in self._process.stdout:
                    self._log(f"[ROS] {line.rstrip()}")
            threading.Thread(target=_read, daemon=True).start()
        except Exception as e:
            self._log(f"[起動エラー] {e}")

    def _on_kill(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log("[停止] sciurus17 プロセスを終了しました")
        else:
            self._log("[停止] 起動済みプロセスがありません")

    def _on_connect_camera(self):
        self._log("カメラ映像受信開始...")
        self._set_state(S_CAMERA)

    def _on_capture(self):
        rgb, dep, intr = self.node.get_snapshot()
        if rgb is None:
            self._log("[エラー] カメラフレームが届いていません")
            return
        self._captured_rgb   = rgb
        self._captured_depth = dep
        self._intrinsics     = intr
        self._click_x = self._click_y = -1
        self._lbl_click.config(text="選択点: -")
        self._show_frame(rgb, overlay_text="FROZEN")
        if self._vis_win:
            self._vis_win.update_frozen(rgb)
        self._log(f"フレーム取得完了: shape={rgb.shape}  ─  クリックで物体を選択してください")
        self._set_state(S_CAPTURED)

    # ──────────────────────── 頭部制御 ─────────────────────────────────────────

    def _on_head_move(self):
        yaw   = self._neck_yaw_var.get()
        pitch = self._neck_pitch_var.get()
        self._run_bg(self._do_head_move, yaw, pitch)

    def _do_head_move(self, yaw_deg: float, pitch_deg: float):
        if self.args.demo:
            self._log(f"[Demo] 首を動かします: ヨー={yaw_deg:.1f}°, ピッチ={pitch_deg:.1f}°")
            time.sleep(0.8)
            self._log("[Demo] 首移動完了")
            return
        robot    = self._get_robot()
        plan_p   = self._make_plan_params(robot, vel=0.3)
        head_comp = robot.get_planning_component(self.args.head_group)
        model    = robot.get_robot_model()
        rs       = RobotState(model)
        rs.set_joint_positions({
            self.args.neck_yaw_joint:   math.radians(yaw_deg),
            self.args.neck_pitch_joint: math.radians(pitch_deg),
        })
        head_comp.set_start_state_to_current_state()
        head_comp.set_goal_state(robot_state=rs)
        _plan_and_execute(robot, head_comp, self._log, plan_p)
        self._log(f"[頭部] 移動完了: ヨー={yaw_deg:.1f}°, ピッチ={pitch_deg:.1f}°")

    def _on_head_home(self):
        self._run_bg(self._do_head_home)

    def _do_head_home(self):
        if self.args.demo:
            self._log("[Demo] 首を初期位置へ戻します")
            time.sleep(0.5)
            self._log("[Demo] 首初期化完了")
            return
        robot    = self._get_robot()
        plan_p   = self._make_plan_params(robot, vel=0.3)
        head_comp = robot.get_planning_component(self.args.head_group)
        head_comp.set_start_state_to_current_state()
        head_comp.set_goal_state(configuration_name="head_init_pose")
        _plan_and_execute(robot, head_comp, self._log, plan_p)
        self._log("[頭部] 初期位置へ移動完了")

    # ──────────────────────── 姿勢推定 ─────────────────────────────────────────

    def _on_estimate(self):
        if self._click_x < 0:
            self._log("[エラー] 画像をクリックして物体を選択してください")
            return
        self._set_state(S_ESTIMATING)
        self._run_bg(self._do_estimate, on_error_state=S_SELECTED)

    def _do_estimate(self):
        if not _SERVER_OK:
            raise RuntimeError("サーバ通信モジュールが利用できません (requests/numpy を確認してください)")
        rgb    = self._captured_rgb
        dep_mm = self._captured_depth
        intr   = self._intrinsics
        depth_m = dep_mm.astype(np.float32) * self.args.depth_scale \
                  if dep_mm is not None else None

        grasp_url = self.args.grasp_url or None
        if not grasp_url:
            import urllib.parse
            parsed = urllib.parse.urlparse(self.args.server_url)
            grasp_url = f"{parsed.scheme}://{parsed.hostname}:8082"
        client = SAM6DClient(server_url=self.args.server_url,
                             timeout_mesh=300.0, timeout_pose=60.0,
                             grasp_server_url=grasp_url)
        self._sam6d_client = client
        self._log(f"[Step 1-3] server_grasp.py で pose+grasp 一括実行中 (選択点: {self._click_x}, {self._click_y})...")
        result = client.estimate_and_generate_grasp(
            rgb, depth_m, intr,
            click_x=self._click_x, click_y=self._click_y,
            gravity=self._gravity,
            mesh_method=self.args.mesh_method,
            num_samples=1,
        )
        self._mesh_path = result.get("mesh_path", self.args.mesh_out)
        R, t = result["R"], result["t"]
        self._R, self._t = R, t
        self._log(f"[Step 1-3完了] mesh: {self._mesh_path}")
        self._log(f"[Step 1-3完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
        rgb_snap, intr_snap = rgb, intr
        self.root.after(0, lambda: self._vis_win and self._vis_win.update_pose(
            rgb_snap, R, t, intr_snap))

        grasp = result["grasps"][0]
        lh_norm = grasp["left_hand"]
        rh_norm = grasp["right_hand"]
        mesh_scale_m = result["mesh_scale_m"]
        R_corr = result["R_corr"].astype(np.float64)
        R_final = (R.astype(np.float64) @ R_corr.T).astype(np.float32)
        pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R_final)
        lh_cam = normalized_to_camera(np.asarray(lh_norm), pose)
        rh_cam = normalized_to_camera(np.asarray(rh_norm), pose)
        self._finish_grasp(lh_cam, rh_cam)

    def _on_grasp(self):
        if self._sam6d_client is not None and self._sam6d_client._server_mesh_path:
            self._run_bg(self._do_grasp_server, on_error_state=S_GRASP_READY)
        elif _GRASP_OK:
            self._run_bg(self._do_grasp_local, on_error_state=S_GRASP_READY)
        else:
            self._log("[情報] サーバ未接続 / torch なし — デモ把持姿勢を使用します")
            self._run_bg(self._do_grasp_demo, on_error_state=S_GRASP_READY)

    def _do_grasp_server(self):
        self._log("[Step 3] Shape2Gesture で把持姿勢生成中 (サーバ側)...")
        result = self._sam6d_client.generate_grasp(
            gravity=self._gravity, num_samples=1
        )
        grasp = result["grasps"][0]
        lh_norm      = grasp["left_hand"]
        rh_norm      = grasp["right_hand"]
        mesh_scale_m = result["mesh_scale_m"]
        R_corr       = result["R_corr"].astype(np.float64)
        R = (self._R.astype(np.float64) @ R_corr.T).astype(np.float32)
        pose = ObjectPose(center_3d=self._t, scale=mesh_scale_m, R=R)
        lh_cam = normalized_to_camera(np.asarray(lh_norm), pose)
        rh_cam = normalized_to_camera(np.asarray(rh_norm), pose)
        self._finish_grasp(lh_cam, rh_cam)

    def _do_grasp_local(self):
        self._log("[Step 3] Shape2Gesture で把持姿勢生成中 (ローカル)...")
        mesh_pts = load_pointcloud_ply(self._mesh_path, target_points=2048)
        # 重力方向で高さ軸を揃えてから把持姿勢を生成する
        mesh_pts_aligned, R_corr = _align_from_gravity(mesh_pts, self._gravity)
        # 重力方向に沿ったオブジェクト高さを計測してログに出力
        g_dir = (self._gravity / np.linalg.norm(self._gravity)) if self._gravity is not None \
                else np.array([0.0, -1.0, 0.0])
        heights = mesh_pts_aligned @ (-g_dir)
        obj_height_m = float((heights.max() - heights.min())) / 1000.0 \
                       if mesh_pts_aligned.shape[1] == 3 else 0.0
        self._log(f"[高さ計測] 重力方向物体高さ ≈ {obj_height_m * 1000:.1f} mm "
                  f"(gravity={g_dir.round(3)})")
        R = (self._R.astype(np.float64) @ R_corr.T).astype(np.float32)
        centered     = mesh_pts_aligned - mesh_pts_aligned.mean(axis=0)
        mesh_scale_m = float(np.max(np.linalg.norm(centered, axis=1))) / 1000.0
        pose = ObjectPose(center_3d=self._t, scale=mesh_scale_m, R=R)
        generator = GraspGenerator(model_dir=self.args.model_dir, epoch=self.args.model_epoch)
        generator.load_models()
        results    = generator.generate(mesh_pts_aligned, num_samples=1)
        lh_norm, rh_norm = results[0]
        lh_cam = normalized_to_camera(lh_norm, pose)
        rh_cam = normalized_to_camera(rh_norm, pose)
        self._finish_grasp(lh_cam, rh_cam)

    def _finish_grasp(self, lh_cam_all: np.ndarray, rh_cam_all: np.ndarray):
        """把持姿勢取得後の共通処理: TF2 変換 + 座標ラベル更新。
        lh_cam_all / rh_cam_all: (23, 3) カメラ座標系の全関節, または (3,) 手首のみ
        """
        lh_cam_all = np.asarray(lh_cam_all, dtype=np.float64)
        rh_cam_all = np.asarray(rh_cam_all, dtype=np.float64)
        lh_wrist = lh_cam_all[0] if lh_cam_all.ndim == 2 else lh_cam_all
        rh_wrist = rh_cam_all[0] if rh_cam_all.ndim == 2 else rh_cam_all
        self._lh_cam, self._rh_cam = lh_wrist, rh_wrist
        self._log(f"  左手首(cam): {lh_wrist}")
        self._log(f"  右手首(cam): {rh_wrist}")
        self._log("[Step 4] TF2 変換: camera → base_link")
        lh_base = self.node.camera_to_base(lh_wrist, self.args.camera_frame, self.args.base_frame)
        rh_base = self.node.camera_to_base(rh_wrist, self.args.camera_frame, self.args.base_frame)
        self._lh_base, self._rh_base = lh_base, rh_base
        rgb_snap, intr_snap = self._captured_rgb, self._intrinsics
        lh_all_s, rh_all_s = lh_cam_all, rh_cam_all
        self.root.after(0, lambda: self._vis_win and self._vis_win.update_grasp(
            rgb_snap, lh_all_s, rh_all_s, intr_snap))
        self._update_coord_labels(lh_base, rh_base)

    def _do_grasp_demo(self):
        self._log("[Demo Step 3] Shape2Gesture シミュレーション...")
        time.sleep(1.5)
        lh_wrist = np.array([self._t[0] - 0.05, self._t[1] + 0.12, self._t[2]])
        rh_wrist = np.array([self._t[0] - 0.05, self._t[1] - 0.12, self._t[2]])
        self._finish_grasp(lh_wrist, rh_wrist)

    def _update_coord_labels(self, lh_base, rh_base):
        self._log(f"  左手首(base): [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]")
        self._log(f"  右手首(base): [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]")
        def _upd():
            self._lbl_lh.config(
                text=f"左手首: [{lh_base[0]:.3f}, {lh_base[1]:.3f}, {lh_base[2]:.3f}]")
            self._lbl_rh.config(
                text=f"右手首: [{rh_base[0]:.3f}, {rh_base[1]:.3f}, {rh_base[2]:.3f}]")
            self._set_state(S_DONE)
        self.root.after(0, _upd)

    # ──────────────────────── アーム移動 ───────────────────────────────────────

    def _on_move(self):
        if self._lh_base is None or self._rh_base is None:
            self._log("[エラー] 把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(
            self._do_move_demo if self.args.demo else self._do_move,
            on_error_state=S_DONE,
        )

    def _on_move_left(self):
        if self._lh_base is None:
            self._log("[エラー] 左手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(
            self._do_move_left_demo if self.args.demo else self._do_move_left,
            on_error_state=S_DONE,
        )

    def _on_move_right(self):
        if self._rh_base is None:
            self._log("[エラー] 右手把持座標が生成されていません")
            return
        self._set_state(S_MOVING)
        self._run_bg(
            self._do_move_right_demo if self.args.demo else self._do_move_right,
            on_error_state=S_DONE,
        )

    def _do_move(self):
        """両アームを把持位置へ移動する。"""
        robot = self._get_robot()
        log   = self._log
        l_arm     = robot.get_planning_component(self.args.l_arm_group)
        r_arm     = robot.get_planning_component(self.args.r_arm_group)
        l_gripper = robot.get_planning_component("l_gripper_group")
        r_gripper = robot.get_planning_component("r_gripper_group")
        robot_model = robot.get_robot_model()
        plan_p  = self._make_plan_params(robot, vel=0.1)
        g_plan_p = self._make_plan_params(robot, vel=1.0)
        OPEN_L  = math.radians(-40.0);  OPEN_R  = math.radians(40.0)
        GRASP_L = math.radians(-20.0);  GRASP_R = math.radians(20.0)
        q_l = Quaternion(x=-0.707, y=0.0, z=0.0, w=0.707)
        q_r = Quaternion(x=0.707,  y=0.0, z=0.0, w=0.707)

        def set_gripper(comp, group, angle):
            rs = RobotState(robot_model)
            rs.set_joint_group_positions(group, [angle])
            comp.set_start_state_to_current_state()
            comp.set_goal_state(robot_state=rs)
            _plan_and_execute(robot, comp, log, g_plan_p)

        log("[Step 5] 初期姿勢へ...")
        for arm, name in ((l_arm, "l_arm_init_pose"), (r_arm, "r_arm_init_pose")):
            arm.set_start_state_to_current_state()
            arm.set_goal_state(configuration_name=name)
            _plan_and_execute(robot, arm, log, plan_p)

        log("[Step 5] グリッパ開放...")
        set_gripper(l_gripper, "l_gripper_group", OPEN_L)
        set_gripper(r_gripper, "r_gripper_group", OPEN_R)

        Z = self.args.pre_grasp_z_offset
        lh_pre = self._lh_base.copy(); lh_pre[2] += Z
        rh_pre = self._rh_base.copy(); rh_pre[2] += Z
        log(f"[Step 5] プリグラスプへ移動 (z+{Z:.2f}m)...")
        self.node.move_arm(robot, l_arm, plan_p, lh_pre, self.args.l_pose_link, q_l)
        self.node.move_arm(robot, r_arm, plan_p, rh_pre, self.args.r_pose_link, q_r)

        log("[Step 5] グラスプ位置へ下降...")
        self.node.move_arm(robot, l_arm, plan_p, self._lh_base, self.args.l_pose_link, q_l)
        self.node.move_arm(robot, r_arm, plan_p, self._rh_base, self.args.r_pose_link, q_r)

        log("[Step 5] グリッパ閉鎖 (把持)...")
        set_gripper(l_gripper, "l_gripper_group", GRASP_L)
        set_gripper(r_gripper, "r_gripper_group", GRASP_R)
        log("[Step 5完了] 把持動作終了")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _move_single_arm(self, robot, side: str):
        """片腕のみを把持位置へ移動する共通ヘルパー (side='left' or 'right')。"""
        log = self._log
        if side == "left":
            arm_group    = self.args.l_arm_group
            gripper_name = "l_gripper_group"
            pose_link    = self.args.l_pose_link
            orientation  = Quaternion(x=-0.707, y=0.0, z=0.0, w=0.707)
            open_angle   = math.radians(-40.0)
            grasp_angle  = math.radians(-20.0)
            init_pose    = "l_arm_init_pose"
            wrist_base   = self._lh_base
        else:
            arm_group    = self.args.r_arm_group
            gripper_name = "r_gripper_group"
            pose_link    = self.args.r_pose_link
            orientation  = Quaternion(x=0.707, y=0.0, z=0.0, w=0.707)
            open_angle   = math.radians(40.0)
            grasp_angle  = math.radians(20.0)
            init_pose    = "r_arm_init_pose"
            wrist_base   = self._rh_base

        plan_p   = self._make_plan_params(robot, vel=0.1)
        g_plan_p = self._make_plan_params(robot, vel=1.0)
        arm      = robot.get_planning_component(arm_group)
        gripper  = robot.get_planning_component(gripper_name)
        model    = robot.get_robot_model()

        def set_gripper(angle):
            rs = RobotState(model)
            rs.set_joint_group_positions(gripper_name, [angle])
            gripper.set_start_state_to_current_state()
            gripper.set_goal_state(robot_state=rs)
            _plan_and_execute(robot, gripper, log, g_plan_p)

        log(f"[{side}アーム] 初期姿勢へ...")
        arm.set_start_state_to_current_state()
        arm.set_goal_state(configuration_name=init_pose)
        _plan_and_execute(robot, arm, log, plan_p)

        log(f"[{side}アーム] グリッパ開放...")
        set_gripper(open_angle)

        Z = self.args.pre_grasp_z_offset
        pre = wrist_base.copy(); pre[2] += Z
        log(f"[{side}アーム] プリグラスプへ移動 (z+{Z:.2f}m)...")
        self.node.move_arm(robot, arm, plan_p, pre, pose_link, orientation)

        log(f"[{side}アーム] グラスプ位置へ下降...")
        self.node.move_arm(robot, arm, plan_p, wrist_base, pose_link, orientation)

        log(f"[{side}アーム] グリッパ閉鎖...")
        set_gripper(grasp_angle)
        log(f"[{side}アーム完了]")

    def _do_move_left(self):
        robot = self._get_robot()
        self._move_single_arm(robot, "left")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_right(self):
        robot = self._get_robot()
        self._move_single_arm(robot, "right")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_demo(self):
        Z = self.args.pre_grasp_z_offset
        lh_pre = self._lh_base.copy(); lh_pre[2] += Z
        rh_pre = self._rh_base.copy(); rh_pre[2] += Z

        steps = [
            ("初期姿勢へ移動中",               1.0),
            ("グリッパ開放",                   0.5),
            (f"プリグラスプへ移動 (z+{Z:.2f}m)", 1.5),
            ("グラスプ位置へ下降",              1.2),
            ("グリッパ閉鎖 (把持)",             0.8),
        ]
        for msg, delay in steps:
            self._log(f"[Demo 両アーム] {msg}...")
            time.sleep(delay)

        self._log(f"[Demo] 左アーム目標: {self._lh_base}")
        self._log(f"[Demo] 右アーム目標: {self._rh_base}")
        self._log("[Mock 両アーム完了]")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_left_demo(self):
        Z = self.args.pre_grasp_z_offset
        steps = [
            ("初期姿勢へ移動中",               1.0),
            ("グリッパ開放",                   0.5),
            (f"プリグラスプへ移動 (z+{Z:.2f}m)", 1.5),
            ("グラスプ位置へ下降",              1.2),
            ("グリッパ閉鎖 (把持)",             0.8),
        ]
        for msg, delay in steps:
            self._log(f"[Demo 左アーム] {msg}...")
            time.sleep(delay)
        self._log(f"[Demo] 左アーム目標: {self._lh_base}")
        self._log("[Mock 左アーム完了]")
        self.root.after(0, lambda: self._set_state(S_DONE))

    def _do_move_right_demo(self):
        Z = self.args.pre_grasp_z_offset
        steps = [
            ("初期姿勢へ移動中",               1.0),
            ("グリッパ開放",                   0.5),
            (f"プリグラスプへ移動 (z+{Z:.2f}m)", 1.5),
            ("グラスプ位置へ下降",              1.2),
            ("グリッパ閉鎖 (把持)",             0.8),
        ]
        for msg, delay in steps:
            self._log(f"[Demo 右アーム] {msg}...")
            time.sleep(delay)
        self._log(f"[Demo] 右アーム目標: {self._rh_base}")
        self._log("[Mock 右アーム完了]")
        self.root.after(0, lambda: self._set_state(S_DONE))

    # ──────────────────────── グリッパ / 初期姿勢 ──────────────────────────────

    def _on_gripper_open(self):
        self._run_bg(self._do_gripper, "open")

    def _on_gripper_close(self):
        self._run_bg(self._do_gripper, "close")

    def _do_gripper(self, action: str):
        label = "開放" if action == "open" else "閉鎖"
        if self.args.demo:
            self._log(f"[Demo] グリッパ{label}シミュレーション...")
            time.sleep(0.8)
            self._log(f"[Demo] グリッパ{label}完了")
            return

        robot    = self._get_robot()
        g_plan_p = self._make_plan_params(robot, vel=1.0)
        model    = robot.get_robot_model()
        angles   = (math.radians(-40), math.radians(40)) if action == "open" \
                   else (math.radians(-20), math.radians(20))
        self._log(f"グリッパ{label}中...")
        for comp_name, group, angle in (
            ("l_gripper_group", "l_gripper_group", angles[0]),
            ("r_gripper_group", "r_gripper_group", angles[1]),
        ):
            comp = robot.get_planning_component(comp_name)
            rs   = RobotState(model)
            rs.set_joint_group_positions(group, [angle])
            comp.set_start_state_to_current_state()
            comp.set_goal_state(robot_state=rs)
            _plan_and_execute(robot, comp, self._log, g_plan_p)

    def _on_home(self):
        self._run_bg(self._do_home)

    def _do_home(self):
        if self.args.demo:
            self._log("[Demo] 初期姿勢へ移動シミュレーション...")
            time.sleep(1.5)
            self._log("[Demo] 初期姿勢完了")
            return
        robot  = self._get_robot()
        plan_p = self._make_plan_params(robot, vel=0.1)
        self._log("初期姿勢へ移動中...")
        for arm, name in (
            (robot.get_planning_component(self.args.l_arm_group), "l_arm_init_pose"),
            (robot.get_planning_component(self.args.r_arm_group), "r_arm_init_pose"),
        ):
            arm.set_start_state_to_current_state()
            arm.set_goal_state(configuration_name=name)
            _plan_and_execute(robot, arm, self._log, plan_p)
        self._log("初期姿勢完了")

    # ──────────────────────── MoveItPy ヘルパー ────────────────────────────────

    def _get_robot(self):
        if not _MOVEIT_OK:
            raise RuntimeError("MoveItPy が利用できません (ROS2 環境で実行してください)")
        if self._robot is None:
            self._log("MoveItPy 初期化中 (数秒かかります)...")
            self._robot = MoveItPy(node_name="sciurus17_gui_moveit")
            self._log("MoveItPy 初期化完了")
        return self._robot

    @staticmethod
    def _make_plan_params(robot, vel=0.1):
        p = PlanRequestParameters(robot, "ompl_rrtc_default")
        p.max_velocity_scaling_factor     = vel
        p.max_acceleration_scaling_factor = vel
        return p

    # ──────────────────────── 終了処理 ─────────────────────────────────────────

    def _on_close(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
        self.root.destroy()


# ── エントリポイント ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="sciurus17 把持コントロールパネル")

    parser.add_argument("--demo", action="store_true",
                        help="デモモード: ROS2/実機なしで動作 (保存画像を使用)")
    parser.add_argument("--demo-image", default=None,
                        help="デモ時に使用する画像フォルダ (rgb.png / depth.png / cam.json)")
    parser.add_argument("--neck-pitch", type=float, default=0.0,
                        help="首ピッチ角の初期値 [deg]。正=下向き。重力方向・高さ計測に使用")

    parser.add_argument("--server-url",  default="http://10.40.1.126:8080")
    parser.add_argument("--grasp-url",   default="",
                        help="server_grasp.py の URL (省略時: --server-url と同じホスト:8082)")
    parser.add_argument("--mesh-out",    default="meshes/object.ply")
    parser.add_argument("--mesh-method", default="knn",
                        choices=["bpa", "poisson", "knn"])
    parser.add_argument("--model-dir",   default="save_model")
    parser.add_argument("--model-epoch", type=int, default=69)

    parser.add_argument("--color-topic",
                        default="/sciurus17/camera/color/image_raw")
    parser.add_argument("--depth-topic",
                        default="/sciurus17/camera/aligned_depth_to_color/image_raw")
    parser.add_argument("--info-topic",
                        default="/sciurus17/camera/color/camera_info")
    parser.add_argument("--depth-scale", type=float, default=0.001)

    parser.add_argument("--camera-frame", default="camera_color_optical_frame")
    parser.add_argument("--base-frame",   default="base_link")
    parser.add_argument("--l-arm-group",  default="l_arm_group")
    parser.add_argument("--r-arm-group",  default="r_arm_group")
    parser.add_argument("--l-pose-link",  default="l_link7")
    parser.add_argument("--r-pose-link",  default="r_link7")
    parser.add_argument("--pre-grasp-z-offset", type=float, default=0.10)

    # 頭部制御
    parser.add_argument("--head-group",        default="head_group",
                        help="頭部 MoveIt planning group 名")
    parser.add_argument("--neck-yaw-joint",    default="neck_joint_1",
                        help="頭部ヨー関節名 (sciurus17 SRDF に合わせて設定)")
    parser.add_argument("--neck-pitch-joint",  default="neck_joint_2",
                        help="頭部ピッチ関節名 (sciurus17 SRDF に合わせて設定)")

    parser.add_argument("--launch-cmd", default="",
                        help="sciurus17 起動コマンド (例: ros2 launch ...)")

    args = parser.parse_args()

    if args.demo or not _ROS2_OK:
        if not args.demo:
            print("[情報] ROS2 が見つかりません。自動的にデモモードで起動します。")
            args.demo = True
        node = DemoRobotNode(args.demo_image)
        spin_thread = None
    else:
        rclpy.init()
        node = RobotNode(args.color_topic, args.depth_topic, args.info_topic)
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()

    root = tk.Tk()
    _app = SciurusGUI(root, node, args)
    try:
        root.mainloop()
    finally:
        if not args.demo and _ROS2_OK:
            rclpy.shutdown()


if __name__ == "__main__":
    main()
