"""
Shape2Gesture 把持姿勢生成サーバ

GraspGenerator (Shape2Gesture) を HTTP サービスとして提供する。
SAM-6D サーバ (server.py) と独立して動作する。

起動方法:
python server_grasp.py \
  --sam6d-service http://localhost:8081 \
  --host 0.0.0.0 \
  --port 8082

# デフォルト値はすべて server/ ディレクトリ内で解決される:
#   client/save_model   … server/client -> ../client (symlink)
#   sam2_checkpoints/   … server/sam2_checkpoints/ (コピー)
#   sam-3d-objects/     … server/sam-3d-objects/ (既存)


エンドポイント:
    GET  /health          — 死活確認
    POST /generate_grasp  — PLY メッシュから把持姿勢生成
"""

import argparse
import csv
from datetime import datetime
import io
import inspect
import json
import os
import subprocess
import sys
import threading

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
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


def _write_grasp_csv(mesh_path: str, grasps: list):
    if not _csv_log_path:
        return
    fieldnames = ["timestamp", "mesh_path", "sample_idx", "hand"] + [
        f"j{i}_{ax}" for i in range(23) for ax in ("x", "y", "z")
    ]
    ts = datetime.now().isoformat(timespec="seconds")
    os.makedirs(os.path.dirname(_csv_log_path), exist_ok=True)
    file_exists = os.path.exists(_csv_log_path)
    with _csv_lock:
        with open(_csv_log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for idx, g in enumerate(grasps):
                for hand_name, joints in (("left", g["left_hand"]), ("right", g["right_hand"])):
                    row = {"timestamp": ts, "mesh_path": mesh_path, "sample_idx": idx, "hand": hand_name}
                    for i, (x, y, z) in enumerate(joints):
                        row[f"j{i}_x"] = x
                        row[f"j{i}_y"] = y
                        row[f"j{i}_z"] = z
                    writer.writerow(row)


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

    # 把持姿勢生成入力: Z軸反転 (server.pyのZ列反転と一致させる)
    R_corr = np.diag([1.0, 1.0, -1.0]).astype(np.float64)
    mesh_pts_aligned = (R_corr @ mesh_pts.T).T.astype(mesh_pts.dtype)

    # メッシュスケール: Z軸長（高さ）[m]
    bbox_ext = mesh_pts_aligned.max(axis=0) - mesh_pts_aligned.min(axis=0)
    mesh_scale_m = float(bbox_ext[2]) / 1000.0

    # 把持姿勢生成
    print(f"[GraspServer] 生成中 (num_samples={num_samples}, scale={mesh_scale_m:.4f} m, "
          f"scaled={'yes' if load_path == scaled_host else 'no'})...")
    results = gen.generate(mesh_pts_aligned, num_samples=num_samples)

    grasps = [
        {"left_hand": lh.tolist(), "right_hand": rh.tolist()}
        for lh, rh in results
    ]
    print(f"[GraspServer] 完了: {len(grasps)} grasps")
    _write_grasp_csv(load_path, grasps)
    if _csv_log_path:
        print(f"[GraspServer] CSV 追記: {_csv_log_path}")

    return JSONResponse({
        "grasps":       grasps,
        "mesh_scale_m": mesh_scale_m,
        "R_corr":       R_corr.tolist(),
    })


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
        "grasps": grasp["grasps"],
        "mesh_scale_m": grasp["mesh_scale_m"],
        "R_corr": grasp["R_corr"],
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
