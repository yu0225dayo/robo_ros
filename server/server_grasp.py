"""
Shape2Gesture 把持姿勢生成サーバ

GraspGenerator (Shape2Gesture) を HTTP サービスとして提供する。
SAM-6D サーバ (server.py) と独立して動作する。

起動方法:
python server_grasp.py \
    --grasp-model-dir /path/to/save_model \
    --grasp-client-dir /path/to/client \
    --host 0.0.0.0 --port 8082


エンドポイント:
    GET  /health          — 死活確認
    POST /generate_grasp  — PLY メッシュから把持姿勢生成
"""

import argparse
import base64
import csv
from datetime import datetime
import io
import inspect
import json
import os
import subprocess
import sys
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Shape2Gesture Grasp Generation Server")

_grasp_model_dir: str = ""
_grasp_client_dir: str = ""
_grasp_generator = None
_grasp_generator_lock = threading.Lock()
_pipeline_server = None
_csv_log_path: str = ""
_csv_lock = threading.Lock()

# /reconstruct_mesh が返す Docker パス → ホストパスのマッピング (server.py と共有 tmp)
_host_tmp: str   = os.path.join(_SERVER_DIR, "tmp")
_docker_tmp: str = "/workspace/tmp"


def _write_grasp_csv(mesh_path: str, grasps: list) -> str:
    """把持姿勢を CSV ファイルに追記し、今回分の CSV 文字列を返す"""
    fieldnames = ["timestamp", "mesh_path", "sample_idx", "hand"] + [
        f"j{i}_{ax}" for i in range(23) for ax in ("x", "y", "z")
    ]
    ts = datetime.now().isoformat(timespec="seconds")

    rows = []
    for idx, g in enumerate(grasps):
        for hand_name, joints in (("left", g["left_hand"]), ("right", g["right_hand"])):
            row = {"timestamp": ts, "mesh_path": mesh_path, "sample_idx": idx, "hand": hand_name}
            for i, (x, y, z) in enumerate(joints):
                row[f"j{i}_x"] = x
                row[f"j{i}_y"] = y
                row[f"j{i}_z"] = z
            rows.append(row)

    # CSV 文字列を生成 (レスポンス用)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    csv_text = buf.getvalue()

    # ファイルにも追記
    if _csv_log_path:
        os.makedirs(os.path.dirname(_csv_log_path), exist_ok=True)
        file_exists = os.path.exists(_csv_log_path)
        with _csv_lock:
            with open(_csv_log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    w.writeheader()
                w.writerows(rows)

    return csv_text


# 23-joint hand index constants (from Shape2Gesture visualization.py handinf)
# Thumb:  0→1→2→3→4→18   (j1=CMC, j18=TIP)
# Index:  0→5→6→7→19      (j5=MCP, j19=TIP)
# Middle: 0→8→9→10→20     (j8=MCP, j20=TIP)
# Ring:   0→11→12→13→21   (j11=MCP, j21=TIP)
# Pinky:  0→14→15→16→17→22 (j14=MCP, j22=TIP)
_J_WRIST      = 0
_J_THUMB_CMC  = 1   # 親指根元
_J_INDEX_MCP  = 5   # 人差し指付け根
_J_MIDDLE_MCP = 8   # 中指付け根
_J_PINKY_MCP  = 14  # 小指付け根

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 18),   # thumb
    (0, 5), (5, 6), (6, 7), (7, 19),             # index
    (0, 8), (8, 9), (9, 10), (10, 20),           # middle
    (0, 11), (11, 12), (12, 13), (13, 21),       # ring
    (0, 14), (14, 15), (15, 16), (16, 17), (17, 22),  # pinky
]


def _compute_palm_frame_core(joints: np.ndarray, flip_z: bool):
    """
    23-joint 手姿勢から手の平基準座標系を計算する内部ヘルパー。

    基準座標系:
      X軸: 手首(j0) → 中指MCP(j8) を手の平平面に射影・正規化
      Z軸: cross(j0→j5, j0→j14) を正規化。flip_z=True のとき符号反転
           左手: flip_z=False, 右手: flip_z=True
           (右手では外積が手のひら側を向くため反転して手の甲=+Z に揃える)
      Y軸: Z × X (右手系)
      原点: 手首 j0

    Args:
        joints: (23, 3) 関節座標配列
        flip_z: True のとき Z軸を反転 (右手用)

    Returns:
        R:     (3, 3) 回転行列。各列が [x_axis, y_axis, z_axis]
               物体座標 p を手の平基準系へ変換: p_palm = R.T @ (p - wrist)
        wrist: (3,) 手首座標 (j0)
        axes:  dict {"x", "y", "z"} — 各軸の単位ベクトル (物体座標系)
    """
    joints = np.asarray(joints, dtype=np.float64)

    wrist      = joints[_J_WRIST]
    index_mcp  = joints[_J_INDEX_MCP]
    middle_mcp = joints[_J_MIDDLE_MCP]
    pinky_mcp  = joints[_J_PINKY_MCP]

    v_index  = index_mcp  - wrist
    v_pinky  = pinky_mcp  - wrist
    v_middle = middle_mcp - wrist

    # Z軸: cross(j0→j5, j0→j14) → j5, j14 が XY 平面に乗る
    z_raw = np.cross(v_index, v_pinky)
    norm_z = np.linalg.norm(z_raw)
    if norm_z < 1e-9:
        raise ValueError("手の平法線が計算できません (3点が直線上)")
    z_axis = z_raw / norm_z
    if flip_z:
        z_axis = -z_axis

    # X軸: 中指方向を手の平平面に射影 (Gram–Schmidt)
    norm_m = np.linalg.norm(v_middle)
    if norm_m < 1e-9:
        raise ValueError("wrist と middle_mcp が同一座標です")
    x_raw  = v_middle / norm_m
    x_axis = x_raw - np.dot(x_raw, z_axis) * z_axis
    norm_x = np.linalg.norm(x_axis)
    if norm_x < 1e-9:
        raise ValueError("中指方向が手の平法線と平行です")
    x_axis /= norm_x

    # Y軸: 右手系
    y_axis = np.cross(z_axis, x_axis)

    R = np.column_stack([x_axis, y_axis, z_axis])
    return R, wrist, {"x": x_axis, "y": y_axis, "z": z_axis}


def compute_palm_frame_right(joints: np.ndarray):
    """
    右手の23-joint姿勢から手の平基準座標系を求める。

    右手では cross(j0→j5, j0→j14) が手のひら側を向くため flip_z=True で反転。

    Returns:
        R:     (3, 3) 回転行列 (列 = [x, y, z])
        wrist: (3,) 手首座標
        axes:  dict {"x", "y", "z"}
    """
    return _compute_palm_frame_core(joints, flip_z=True)


def compute_palm_frame_left(joints: np.ndarray):
    """
    左手の23-joint姿勢から手の平基準座標系を求める。

    左手では cross(j0→j5, j0→j14) がそのまま手の甲方向を向く。

    Returns:
        R:     (3, 3) 回転行列 (列 = [x, y, z])
        wrist: (3,) 手首座標
        axes:  dict {"x", "y", "z"}
    """
    return _compute_palm_frame_core(joints, flip_z=False)


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, _SERVER_DIR)
    except ValueError:
        return path


def _get_docker_workspace_host(container: str = "sam6d_service") -> str:
    """sam6d Docker コンテナの /workspace に対応するホストパスを返す"""
    try:
        r = subprocess.run(
            ["docker", "inspect", "--format", "{{json .Mounts}}", container],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for m in json.loads(r.stdout.strip()):
                dst = m.get("Destination", "")
                src = m.get("Source", "")
                if dst == "/workspace":
                    return src
                if dst == "/workspace/tmp" and src.endswith("/tmp"):
                    return src[:-4]  # /workspace/tmp → strip /tmp → host workspace root
    except Exception:
        pass
    return ""


def _align_from_gravity(pts: np.ndarray, gravity_cam=None):
    """点群を重力方向に揃える (Y-down)。整列済み点群と回転行列 R_corr を返す。"""
    if gravity_cam is None or np.linalg.norm(gravity_cam) < 1e-6:
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


def _visualize_zflip_pem(
    pts_before: np.ndarray,
    pts_after: np.ndarray,
    mesh_host_path: str,
) -> str:
    """
    Z軸反転 before/after の PEM スタイル画像を生成し base64 文字列を返す。

    detection_pem.json + rgb.png + camera_custom.json が揃う場合は RGB 上に投影。
    揃わない場合は直交3視点スキャッタで代替。
    保存先: <mesh_dir>/pem_zflip_before_after.png

    Returns:
        base64 エンコードされた PNG 文字列。失敗時は ""。
    """
    # detection_pem.json のパスを mesh_host_path から導出
    # e.g. .../object_seed42_mesh_scaled.ply → .../object_seed42_mesh_templates/sam6d_results/detection_pem.json
    stem = mesh_host_path
    if stem.endswith("_scaled.ply"):
        stem = stem[: -len("_scaled.ply")]   # → ...object_seed42_mesh
    elif stem.endswith(".ply"):
        stem = stem[: -len(".ply")]           # → ...object_seed42_mesh (or similar)
    pem_json_path = os.path.join(stem + "_templates", "sam6d_results", "detection_pem.json")
    rgb_path      = os.path.join(_host_tmp, "rgb.png")
    cam_path      = os.path.join(_host_tmp, "camera_custom.json")

    sam6d_results_dir = os.path.join(stem + "_templates", "sam6d_results")
    os.makedirs(sam6d_results_dir, exist_ok=True)
    out_path = os.path.join(sam6d_results_dir, "pem_zflip_before_after.png")

    try:
        if (os.path.exists(pem_json_path)
                and os.path.exists(rgb_path)
                and os.path.exists(cam_path)):
            img = _pem_projection(pts_before, pts_after, pem_json_path, rgb_path, cam_path)
            mode = "projection"
        else:
            img = _pem_orthographic(pts_before, pts_after)
            mode = "orthographic"

        cv2.imwrite(out_path, img)
        print(f"[GraspServer] z-flip before/after [{mode}]: {_rel(out_path)}")
        _, buf = cv2.imencode(".png", img)
        return base64.b64encode(buf).decode()
    except Exception as e:
        print(f"[GraspServer] z-flip 可視化失敗: {e}")
        return ""


def _pem_projection(
    pts_before: np.ndarray,
    pts_after: np.ndarray,
    pem_json_path: str,
    rgb_path: str,
    cam_path: str,
) -> np.ndarray:
    """detection_pem.json の R・t を使って RGB 画像上に点群を投影した before/after 画像を返す。

    server.py の条件付き Z 反転ロジックを可視化:
      BEFORE: R_raw をそのまま使って投影 (反転処理なし)
      AFTER:  R[1,2]<0 なら Z 列を反転して DOWN に揃えた R を使って投影 (反転処理あり)
    → R[1,2]>0 の場合は反転処理が走らないので BEFORE=AFTER (同一画像)
    """
    bgr = cv2.imread(rgb_path)
    h, w = bgr.shape[:2]

    with open(cam_path) as f:
        cam = json.load(f)
    K = np.array(cam["cam_K"], dtype=np.float32).reshape(3, 3)

    with open(pem_json_path) as f:
        dets = json.load(f)
    if not dets:
        return _pem_orthographic(pts_before, pts_after)
    best  = max(dets, key=lambda d: d["score"])
    R_raw = np.array(best["R"], dtype=np.float32)
    t_mm  = np.array(best["t"], dtype=np.float32)

    # BEFORE: PEM出力 R_raw をそのまま表示 (表示用フリップなし)
    R_vis_before = R_raw.copy()

    # AFTER: server.py の条件付き Z 反転のみ適用 (表示用フリップなし)
    R_after = R_raw.copy()
    flip_applied = R_after[1, 2] > 0
    if flip_applied:
        R_after[:, 2] *= -1
    R_vis_after = R_after.copy()

    pts = pts_before   # 両パネルとも元のメッシュ点群を使用

    def _project_pts(R_vis):
        np.random.seed(42)
        idx = np.random.choice(len(pts), min(len(pts), 1000), replace=False)
        cam_pts = R_vis @ pts[idx].T + t_mm[:, None]
        p = K @ cam_pts
        return (p[:2] / p[2]).T.astype(np.int32)

    def _project_bbox(R_vis):
        mn, mx = pts.min(0), pts.max(0)
        corners = (np.array([[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],
                               [-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1]], dtype=np.float32)
                   * ((mx - mn) / 2) + (mn + mx) / 2)
        cam_c = R_vis @ corners.T + t_mm[:, None]
        p = K @ cam_c
        return (p[:2] / p[2]).T.astype(np.int32)

    def _draw_axes(panel, R_axes):
        diag = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        ax_len = max(diag * 0.30, 20.0)

        def proj_cam(cam_pt):
            cam = np.asarray(cam_pt, np.float64)
            if cam[2] <= 0:
                return None
            p = K.astype(np.float64) @ cam
            return (int(p[0] / p[2]), int(p[1] / p[2]))

        t_d = t_mm.astype(np.float64)
        R_d = R_axes.astype(np.float64)
        o2d = proj_cam(t_d)
        if o2d is None:
            return
        for i, (color, lbl) in enumerate([
            ((0,   0, 220), "X"),
            ((0, 200,   0), "Y"),
            ((220,  0,   0), "Z"),
        ]):
            e2d = proj_cam(t_d + R_d[:, i] * ax_len)
            if e2d is None:
                continue
            if i == 2 and e2d[1] > o2d[1]:
                e2d = (e2d[0], 2 * o2d[1] - e2d[1])
            cv2.arrowedLine(panel, o2d, e2d, color, 2, tipLength=0.25)
            ex = max(0, min(w - 16, e2d[0] + 4))
            ey = max(14, min(h - 4, e2d[1]))
            cv2.putText(panel, lbl, (ex, ey), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    flip_label = "Z反転あり (DOWN→UP)" if flip_applied else "Z反転なし (すでにUP)"
    panels = []
    for label, dot_color, bbox_color, R_vis in [
        ("BEFORE  (Z反転処理前)",  (0, 220,   0), (0, 255, 200), R_vis_before),
        (f"AFTER   ({flip_label})", (0, 100, 255), (80, 80, 255), R_vis_after),
    ]:
        panel = bgr.copy()
        p2d = _project_pts(R_vis)
        in_b = (p2d[:,0] >= 0) & (p2d[:,0] < w) & (p2d[:,1] >= 0) & (p2d[:,1] < h)
        for u, v in p2d[in_b]:
            cv2.circle(panel, (int(u), int(v)), 2, dot_color, -1)

        p2d_b = _project_bbox(R_vis)
        c_dark = tuple(int(c * 0.45) for c in bbox_color)
        for i, j in [(0,1),(1,3),(3,2),(2,0)]:
            cv2.line(panel, tuple(p2d_b[i]), tuple(p2d_b[j]), bbox_color, 2)
        for i, j in [(4,5),(5,7),(7,6),(6,4)]:
            cv2.line(panel, tuple(p2d_b[i]), tuple(p2d_b[j]), c_dark, 1)
        for i, j in [(0,4),(1,5),(2,6),(3,7)]:
            cv2.line(panel, tuple(p2d_b[i]), tuple(p2d_b[j]), c_dark, 1)

        _draw_axes(panel, R_vis)

        cv2.putText(panel, label, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        panels.append(panel)

    combined = np.hstack(panels)
    bar = np.zeros((26, combined.shape[1], 3), dtype=np.uint8)
    info = (f"R[1,2]_raw={R_raw[1,2]:.3f}  "
            f"{flip_label}  "
            f"t=[{t_mm[0]:.0f},{t_mm[1]:.0f},{t_mm[2]:.0f}]mm  "
            f"score={best['score']:.3f}")
    cv2.putText(bar, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 255), 1)
    return np.vstack([bar, combined])


def _pem_orthographic(pts_before: np.ndarray, pts_after: np.ndarray) -> np.ndarray:
    """投影情報がない場合の代替: XZ / YZ / XY の3視点スキャッタ before/after。"""
    SIZE, MARGIN = 280, 18

    def _to_img(pts, ax0, ax1):
        p2 = pts[:, [ax0, ax1]].copy()
        mn, mx = p2.min(0), p2.max(0)
        rng = max((mx - mn).max(), 1e-6)
        p2 = (p2 - mn) / rng * (SIZE - 2 * MARGIN) + MARGIN
        return p2.astype(np.int32)

    views = [("XZ (front)", 0, 2), ("YZ (side)", 1, 2), ("XY (top)", 0, 1)]
    rows  = []
    for view_label, a0, a1 in views:
        col = []
        for pts, title, color in [
            (pts_before, f"BEFORE  {view_label}", (0, 220,   0)),
            (pts_after,  f"AFTER   {view_label}", (0, 100, 255)),
        ]:
            panel = np.full((SIZE, SIZE, 3), 25, dtype=np.uint8)
            p2d = _to_img(pts, a0, a1)
            for u, v in p2d:
                cv2.circle(panel, (int(u), SIZE - int(v)), 1, color, -1)
            cv2.putText(panel, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 0), 1)
            col.append(panel)
        rows.append(np.hstack(col))
    return np.vstack(rows)


def _get_grasp_generator():
    """GraspGenerator を遅延ロードして返す。model_dir 未指定なら None。"""
    global _grasp_generator
    if _grasp_generator is not None:
        return _grasp_generator
    if not _grasp_model_dir:
        return None
    with _grasp_generator_lock:
        if _grasp_generator is None:
            if _grasp_client_dir and _grasp_client_dir not in sys.path:
                sys.path.insert(0, _grasp_client_dir)
            from pipeline.grasp_generator import GraspGenerator
            gen = GraspGenerator(model_dir=_grasp_model_dir)
            gen.load_models()
            _grasp_generator = gen
            print(f"[GraspServer] GraspGenerator ロード完了 (model_dir={_grasp_model_dir})")
    return _grasp_generator


def _json_body(resp: JSONResponse) -> dict:
    return json.loads(resp.body.decode("utf-8"))


def _accepted_kwargs(fn, kwargs: dict) -> dict:
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def _ensure_sam3d_import_paths():
    if _pipeline_server is None:
        return
    sam3d_repo = getattr(_pipeline_server, "_sam3d_repo", "")
    if not sam3d_repo:
        return
    sam3d_repo = os.path.abspath(os.path.expanduser(sam3d_repo))
    notebook_path = os.path.join(sam3d_repo, "notebook")
    inference_path = os.path.join(notebook_path, "inference.py")
    if not os.path.exists(inference_path):
        raise HTTPException(
            500,
            "sam-3d-objects notebook/inference.py was not found. "
            f"Check --sam3d-repo: {sam3d_repo}",
        )
    for path in (sam3d_repo, notebook_path):
        if path not in sys.path:
            sys.path.insert(0, path)


@app.get("/health")
def health():
    gen = _grasp_generator
    pipeline_loaded = (
        _pipeline_server is not None
        and getattr(_pipeline_server, "sam_predictor", None) is not None
    )
    return {"status": "ok", "model_loaded": gen is not None,
            "pipeline_loaded": pipeline_loaded,
            "model_dir": _grasp_model_dir}


@app.post("/generate_grasp")
async def generate_grasp(
    mesh_path: str = Form(...),
    gravity_x: float = Form(0.0),
    gravity_y: float = Form(0.0),
    gravity_z: float = Form(0.0),
    num_samples: int = Form(1),
):
    """
    Shape2Gesture で把持姿勢を生成する

    Args:
        mesh_path:       サーバ側の PLY ファイルパス
                         (/reconstruct_mesh が返す mesh_path をそのまま渡す)
        gravity_x/y/z:   カメラ座標系の重力方向ベクトル (0,0,0=未指定→固定補正)
        num_samples:     生成する把持候補数

    Returns:
        {
          "grasps": [
            {"left_hand": [[x,y,z],...23関節],
             "right_hand": [[x,y,z],...23関節]}
          ],
          "mesh_scale_m": float,   # 正規化座標→メートルのスケール係数
          "R_corr": [[...]]        # 重力アライメント回転行列 (3x3)
        }

    クライアント側での座標変換:
        R = (R_from_sam6d @ R_corr.T)
        pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)
        wrist_cam = normalized_to_camera(left_hand[0], pose)
    """
    gen = _get_grasp_generator()
    if gen is None:
        raise HTTPException(503, "GraspGenerator が未ロードです。--grasp-model-dir を指定して起動してください。")

    # Docker パス → ホストパスに変換 (server.py の tmp を共有)
    mesh_host = mesh_path.replace(_docker_tmp, _host_tmp)
    if not os.path.exists(mesh_host):
        raise HTTPException(404, f"メッシュが見つかりません: {mesh_host}")

    # pose_estimate が生成した Z軸スケール済みメッシュがあれば優先して使用
    scaled_host = mesh_host.replace(".ply", "_scaled.ply")
    load_path = scaled_host if os.path.exists(scaled_host) else mesh_host

    # PLY 点群読み込み (open3d があれば使用、なければ plyfile)
    try:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(load_path)
        mesh_pts = np.asarray(pcd.points, dtype=np.float32)
        if len(mesh_pts) == 0:
            raise ValueError("open3d で点群が空")
    except Exception:
        from plyfile import PlyData
        ply_data = PlyData.read(load_path)
        v = ply_data["vertex"]
        mesh_pts = np.stack([v["x"], v["y"], v["z"]], axis=-1).astype(np.float32)

    if len(mesh_pts) == 0:
        raise HTTPException(500, "点群が空です")

    print(f"[GraspServer] PLY 読み込み: {len(mesh_pts)} pts ({_rel(load_path)})")

    # detection_pem.json から R[1,2] を読み、server.py と同じ条件でZ反転を判定
    stem = load_path
    if stem.endswith("_scaled.ply"):
        stem = stem[:-len("_scaled.ply")]
    elif stem.endswith(".ply"):
        stem = stem[:-len(".ply")]
    pem_json_path = os.path.join(stem + "_templates", "sam6d_results", "detection_pem.json")

    flip_z = False
    if os.path.exists(pem_json_path):
        with open(pem_json_path) as f:
            dets = json.load(f)
        if dets:
            best_det = max(dets, key=lambda d: d["score"])
            R_pem = np.array(best_det["R"], dtype=np.float32)
            flip_z = bool(R_pem[1, 2] > 0)
            print(f"[GraspServer] R[1,2]={R_pem[1,2]:.3f} → Z反転{'あり (DOWN→UP)' if flip_z else 'なし (すでにUP)'}")
    else:
        print(f"[GraspServer] detection_pem.json なし → Z反転なし")

    # 条件付きZ反転: server.py の R[:,2] 反転と対応させる
    if flip_z:
        mesh_pts_aligned = mesh_pts.copy()
        mesh_pts_aligned[:, 2] *= -1
        mesh_pts_aligned = mesh_pts_aligned.astype(mesh_pts.dtype)
        R_corr = np.diag([1.0, 1.0, -1.0]).astype(np.float64)
    else:
        mesh_pts_aligned = mesh_pts.copy()
        R_corr = np.eye(3, dtype=np.float64)

    # Z軸反転 before/after の PEM 可視化
    img_zflip_b64 = _visualize_zflip_pem(mesh_pts, mesh_pts_aligned, load_path)

    # メッシュスケール: Z軸長（高さ）[m]
    bbox_ext = mesh_pts_aligned.max(axis=0) - mesh_pts_aligned.min(axis=0)
    mesh_scale_m = float(bbox_ext[2]) / 1000.0

    # 把持姿勢生成
    print(f"[GraspServer] 生成中 (num_samples={num_samples}, scale={mesh_scale_m:.4f} m, "
          f"scaled={'yes' if load_path == scaled_host else 'no'})...")
    results = gen.generate(mesh_pts_aligned, num_samples=num_samples)

    grasps = []
    for lh, rh in results:
        entry = {"left_hand": lh.tolist(), "right_hand": rh.tolist()}

        # 手の平基準座標系 (palm frame) を各手に付加
        for key, joints_np, frame_fn in (
            ("left_hand",  lh, compute_palm_frame_left),
            ("right_hand", rh, compute_palm_frame_right),
        ):
            try:
                R_palm, wrist_pos, axes = frame_fn(joints_np)
                entry[key.replace("_hand", "_palm_frame")] = {
                    "R":      R_palm.tolist(),      # 3×3: 列が [x_axis, y_axis, z_axis]
                    "wrist":  wrist_pos.tolist(),
                    "x_axis": axes["x"].tolist(),   # 手首→親指方向
                    "y_axis": axes["y"].tolist(),
                    "z_axis": axes["z"].tolist(),   # 手の甲方向 (左右共通)
                }
            except Exception as e:
                entry[key.replace("_hand", "_palm_frame")] = {"error": str(e)}

        grasps.append(entry)

    print(f"[GraspServer] 完了: {len(grasps)} grasps")
    csv_text = _write_grasp_csv(load_path, grasps)
    if _csv_log_path:
        print(f"[GraspServer] CSV 追記: {_csv_log_path}")

    return JSONResponse({
        "grasps":         grasps,
        "mesh_scale_m":   mesh_scale_m,
        "R_corr":         R_corr.tolist(),
        "grasps_csv":     csv_text,
        "img_zflip_b64":  img_zflip_b64,
    })


@app.get("/download_grasps_csv")
async def download_grasps_csv():
    """蓄積した把持姿勢 CSV をダウンロードする"""
    if not _csv_log_path or not os.path.exists(_csv_log_path):
        raise HTTPException(404, "grasps.csv がまだ存在しません。先に /generate_grasp を実行してください。")
    return FileResponse(
        _csv_log_path,
        media_type="text/csv",
        filename="grasps.csv",
    )


@app.post("/estimate_and_generate_grasp")
async def estimate_and_generate_grasp(
    rgb_image: UploadFile = File(...),
    depth_image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    mesh_method: str = Form("knn"),
    object_size_mm: float = Form(0.0),
    det_score_thresh: float = Form(0.2),
    gravity_x: float = Form(0.0),
    gravity_y: float = Form(0.0),
    gravity_z: float = Form(0.0),
    num_samples: int = Form(1),
):
    if _pipeline_server is None:
        raise HTTPException(
            503,
            "SAM/SAM-6D pipeline is not loaded. Start server_grasp.py with "
            "--sam-checkpoint, --sam3d-config and --sam3d-repo.",
        )

    rgb_bytes = await rgb_image.read()
    depth_bytes = await depth_image.read()
    _ensure_sam3d_import_paths()

    recon_resp = await _pipeline_server.reconstruct_mesh(
        image=UploadFile(filename="frame.jpg", file=io.BytesIO(rgb_bytes)),
        click_x=click_x,
        click_y=click_y,
        seed=seed,
        target_points=2048,
        output_dir="",
        mesh_method=mesh_method,
        object_size_mm=object_size_mm,
    )
    recon = _json_body(recon_resp)
    mesh_path = recon["mesh_path"]
    template_dir = recon["template_dir"]

    pose_kwargs = _accepted_kwargs(_pipeline_server.pose_estimate, {
        "rgb_image": UploadFile(filename="frame.jpg", file=io.BytesIO(rgb_bytes)),
        "depth_image": UploadFile(filename="depth.bin", file=io.BytesIO(depth_bytes)),
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "mesh_path": mesh_path,
        "template_dir": template_dir,
        "det_score_thresh": det_score_thresh,
        "click_x": click_x,
        "click_y": click_y,
        "object_size_mm": object_size_mm,
        "gravity_x": gravity_x,
        "gravity_y": gravity_y,
        "gravity_z": gravity_z,
    })
    pose_resp = await _pipeline_server.pose_estimate(**pose_kwargs)
    pose = _json_body(pose_resp)
    if not pose.get("success"):
        raise HTTPException(500, f"pose estimate failed: {pose}")

    grasp_resp = await generate_grasp(
        mesh_path=mesh_path,
        gravity_x=gravity_x,
        gravity_y=gravity_y,
        gravity_z=gravity_z,
        num_samples=num_samples,
    )
    grasp = _json_body(grasp_resp)

    return JSONResponse({
        "success": True,
        "mesh_path": mesh_path,
        "template_dir": template_dir,
        "mask_center_u": recon.get("mask_center_u"),
        "mask_center_v": recon.get("mask_center_v"),
        "scores": recon.get("scores", []),
        "best_idx": recon.get("best_idx", 0),
        "R": pose["R"],
        "t": pose["t"],
        "mask_area": pose.get("mask_area", 0),
        "img_pose": pose.get("img_pose", ""),
        "img_mesh": pose.get("img_mesh", ""),
        "grasps":        grasp["grasps"],
        "mesh_scale_m":  grasp["mesh_scale_m"],
        "R_corr":        grasp["R_corr"],
        "grasps_csv":    grasp.get("grasps_csv", ""),
        "img_zflip_b64": grasp.get("img_zflip_b64", ""),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grasp-model-dir",
                        default=os.path.join(_SERVER_DIR, "client", "save_model"),
                        help="Shape2Gesture の save_model ディレクトリ")
    parser.add_argument("--grasp-client-dir",
                        default=os.path.join(_SERVER_DIR, "client"),
                        help="client/ ディレクトリ (pipeline/models のインポート元)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    parser.add_argument("--host-tmp", default=os.path.join(_SERVER_DIR, "tmp"))
    parser.add_argument("--docker-tmp", default="/workspace/tmp")
    parser.add_argument("--sam-checkpoint",
                        default=os.path.join(_SERVER_DIR, "sam2_checkpoints", "sam2.1_hiera_large.pt"),
                        help="SAM2 checkpoint. Required for /estimate_and_generate_grasp.")
    parser.add_argument("--sam3d-config",
                        default=os.path.join(_SERVER_DIR, "sam-3d-objects", "checkpoints", "hf", "pipeline.yaml"),
                        help="sam-3d-objects pipeline.yaml. Required for /estimate_and_generate_grasp.")
    parser.add_argument("--sam3d-repo",
                        default=os.path.join(_SERVER_DIR, "sam-3d-objects"),
                        help="sam-3d-objects repo path. Required for /estimate_and_generate_grasp.")
    parser.add_argument("--sam6d-service", default="http://localhost:8081",
                        help="SAM-6D service URL used by the imported pipeline server.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv-log", default=os.path.join(_SERVER_DIR, "tmp", "grasps.csv"),
                        help="把持姿勢を追記するCSVファイルパス (省略時は server/tmp/grasps.csv)")
    args = parser.parse_args()

    _grasp_model_dir = args.grasp_model_dir
    _grasp_client_dir = args.grasp_client_dir
    _csv_log_path = args.csv_log
    _host_tmp   = args.host_tmp
    _docker_tmp = args.docker_tmp

    print("=" * 50)
    print("  Shape2Gesture Grasp Generation Server")
    print(f"  host:port:       {args.host}:{args.port}")
    print(f"  grasp_model_dir: {_grasp_model_dir}")
    print(f"  grasp_client_dir:{_grasp_client_dir}")
    print(f"  host_tmp:        {_host_tmp}")
    print(f"  pipeline:        {'enabled' if args.sam_checkpoint and args.sam3d_config and args.sam3d_repo else 'disabled'}")
    print("=" * 50)

    _get_grasp_generator()

    if args.sam_checkpoint and args.sam3d_config and args.sam3d_repo:
        import server as pipeline_server

        args.sam_checkpoint = os.path.abspath(os.path.expanduser(args.sam_checkpoint))
        args.sam3d_config = os.path.abspath(os.path.expanduser(args.sam3d_config))
        args.sam3d_repo = os.path.abspath(os.path.expanduser(args.sam3d_repo))
        notebook_path = os.path.join(args.sam3d_repo, "notebook")
        for path in (args.sam3d_repo, notebook_path):
            if path not in sys.path:
                sys.path.insert(0, path)

        # Docker の /workspace マウント元から正しい _host_tmp を導出
        _ws_host = _get_docker_workspace_host()
        if _ws_host:
            pipeline_server._host_tmp = os.path.join(_ws_host, "tmp")
            print(f"  [docker workspace] {_ws_host}  → _host_tmp={pipeline_server._host_tmp}")
        else:
            pipeline_server._host_tmp = _host_tmp
            print(f"  [warning] Docker workspace 取得失敗。_host_tmp={_host_tmp}")
        pipeline_server._docker_tmp = _docker_tmp
        pipeline_server._sam6d_url = args.sam6d_service.rstrip("/")
        pipeline_server.load_models(
            sam_checkpoint=args.sam_checkpoint,
            sam3d_config=args.sam3d_config,
            sam3d_repo=args.sam3d_repo,
            device=args.device,
        )
        _pipeline_server = pipeline_server

    uvicorn.run(app, host=args.host, port=args.port)
