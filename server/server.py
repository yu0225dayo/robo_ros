"""
SAM 3D Objects + SAM-6D パイプラインサーバ (Linux + A6000 で実行)

クライアント (ローカルPC) から RGB+深度画像を受け取り、以下を行う:
    1. SAM + sam-3d-objects で完全3Dモデル (PLY) を生成
    2. SAM-6D Docker サービス (port 8081) へプロキシして6DoF姿勢推定

起動方法 (Linux サーバ上で):
    python server.py \
        --sam-checkpoint ~/ws/sam_vit_h_4b8939.pth \
        --sam3d-config   ~/ws/sam-3d-objects/checkpoints/hf/pipeline.yaml \
        --sam3d-repo     ~/ws/sam-3d-objects \
        --sam6d-service  http://localhost:8081 \
        --host 0.0.0.0 --port 8080
"""

import argparse
import sys
import os
import json
import numpy as np
import cv2
import torch
import httpx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

_SERVER_DIR = os.path.dirname(os.path.abspath(__file__))


def _rel(path: str) -> str:
    """絶対パスをサーバディレクトリからの相対パスに変換する"""
    try:
        return os.path.relpath(path, _SERVER_DIR)
    except ValueError:
        return path


app = FastAPI(title="SAM 3D + SAM-6D Pipeline Server")

# SAM のみ起動時にロード (軽量なのでキープ)
sam_predictor = None
args_global = None

# SAM-3D はオンデマンドロード (推論後に削除してGPUを解放)
# モデル自体は保持せず、設定パスだけ保持する
_sam3d_config: str = ""
_sam3d_repo: str   = ""
_sam3d_device: str = "cuda"

# SAM-6D サービス URL (Docker コンテナ)
_sam6d_url: str = "http://localhost:8081"

# ホスト↔Dockerコンテナ間の共有tmpディレクトリパスマッピング
_host_tmp: str   = os.path.join(_SERVER_DIR, "tmp")
_docker_tmp: str = "/workspace/tmp"


def to_docker_path(host_path: str) -> str:
    """ホスト側の絶対パスをDockerコンテナ内のパスに変換する"""
    abs_path = os.path.abspath(host_path)
    if abs_path.startswith(_host_tmp):
        return _docker_tmp + abs_path[len(_host_tmp):]
    return abs_path


def load_models(sam_checkpoint: str, sam3d_config: str, sam3d_repo: str,
                device: str = "cuda"):
    """SAM のみ起動時にロード。SAM-3D はオンデマンドロード。"""
    global sam_predictor, _sam3d_config, _sam3d_repo, _sam3d_device

    # SAM-3D の設定パスを保存 (モデル本体はロードしない)
    _sam3d_config = sam3d_config
    _sam3d_repo   = sam3d_repo
    _sam3d_device = device

    # sam-3d-objects をパスに追加 (notebook/ に inference.py がある)
    notebook_path = os.path.join(sam3d_repo, "notebook")
    for p in [sam3d_repo, notebook_path]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # SAM2
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    # sam_checkpoint 引数をSAM2チェックポイントとして使用
    sam2_ckpt = sam_checkpoint
    sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=device)
    sam_predictor = SAM2ImagePredictor(sam2_model)
    print(f"[Server] SAM2 ロード完了 ({device})")
    print("[Server] SAM-3D はリクエスト時にオンデマンドロードします")


def _load_sam3d_and_run(rgb, best_mask, seed):
    """SAM-3D をロード → 推論 → 即削除してGPUを解放する"""
    import gc
    import torch
    from inference import Inference

    print("[Server] SAM-3D ロード中...")
    sam3d = Inference(_sam3d_config, compile=False)
    try:
        print("[Server] SAM-3D 推論中...")
        output = sam3d(rgb, best_mask, seed=seed)
        return output
    finally:
        # 推論後に即削除してGPUを解放
        del sam3d
        gc.collect()
        torch.cuda.empty_cache()
        print("[Server] SAM-3D 削除・GPU解放完了")


@app.get("/health")
def health():
    """サーバの死活確認"""
    return {"status": "ok", "models_loaded": sam_predictor is not None}


@app.post("/reconstruct")
async def reconstruct(
    image: UploadFile = File(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    target_points: int = Form(2048),
    output_dir: str = Form("tmp/server_reconstructions"),
):
    """
    RGB画像から物体の完全3D点群を生成して返す

    Args:
        image:         RGB画像ファイル (JPEG/PNG)
        click_x, click_y: SAMプロンプト座標 (-1,-1 で画像中央を使用)
        seed:          sam-3d-objects のランダムシード
        target_points: 返す点群の点数

    Returns:
        JSON: {"points": [[x,y,z], ...], "num_points": N,
               "mask_center_u": int, "mask_center_v": int}
    """
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="モデルがロードされていません")

    # 画像をデコード
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="画像のデコードに失敗しました")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    # SAMプロンプト設定
    if click_x < 0 or click_y < 0:
        prompt_point = np.array([[w // 2, h // 2]])
    else:
        prompt_point = np.array([[click_x, click_y]])

    # Step 1: SAM2 で2Dマスク生成
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(rgb)
        masks, scores, _ = sam_predictor.predict(
            point_coords=prompt_point,
            point_labels=np.array([1]),
            multimask_output=True,
        )
    best_mask = masks[np.argmax(scores)]  # スコア最大のマスクを使用
    print(f"[Server] SAM マスク生成完了 (面積: {best_mask.sum()} px, "
          f"プロンプト: ({prompt_point[0][0]}, {prompt_point[0][1]}))")

    # SAM2 マスクを保存 (3枚 + スコア表示)
    os.makedirs(output_dir, exist_ok=True)
    for _i in range(3):
        _m = cv2.cvtColor(masks[_i].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        cv2.putText(_m, f"mask_{_i+1}  score={scores[_i]:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_dir, f"mask_sam2_{_i+1}.png"), _m)
    mask_sam2_path = os.path.join(output_dir, "mask_sam2.png")
    cv2.imwrite(mask_sam2_path, (best_mask.astype(np.uint8) * 255))
    print(f"[Server] SAM2 マスク保存: {_rel(output_dir)}/mask_sam2_{{1,2,3}}.png")

    # Step 2: SAM-3D でモデル生成 (推論後に即削除)
    output = _load_sam3d_and_run(rgb, best_mask, seed)

    # Step 3: PLY保存 → XYZ抽出
    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"object_seed{seed}.ply")
    output["gs"].save_ply(ply_path)
    print(f"[Server] PLY保存: {_rel(ply_path)}")

    # Gaussian splat PLY から XYZ 座標を抽出
    from plyfile import PlyData
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    points = np.stack([
        vertex["x"].astype(np.float32),
        vertex["y"].astype(np.float32),
        vertex["z"].astype(np.float32),
    ], axis=-1)
    print(f"[Server] Gaussian数: {len(points)}")

    # リサンプリング
    n = len(points)
    choice = np.random.choice(n, target_points, replace=(n < target_points))
    points = points[choice]

    # マスク重心ピクセル (Windows側でカメラ座標変換に使用)
    ys, xs = np.where(best_mask)
    mask_center_u = int(xs.mean())
    mask_center_v = int(ys.mean())

    print(f"[Server] 点群送信: {len(points)} points, mask_center=({mask_center_u},{mask_center_v})")
    return JSONResponse({
        "points": points.tolist(),
        "num_points": len(points),
        "ply_path": ply_path,
        "mask_center_u": mask_center_u,
        "mask_center_v": mask_center_v,
    })


def _sam6d_post(endpoint: str, payload: dict, timeout: float = 300.0) -> dict:
    """SAM-6D Docker サービスへ JSON POST し、レスポンスを返す"""
    url = f"{_sam6d_url}/{endpoint.lstrip('/')}"
    try:
        resp = httpx.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except httpx.ConnectError:
        raise HTTPException(503, f"SAM-6D サービスに接続できません ({url}). "
                                 "docker compose up sam6d を確認してください。")
    except httpx.HTTPStatusError as e:
        raise HTTPException(e.response.status_code,
                            f"SAM-6D サービスエラー: {e.response.text}")


@app.post("/reconstruct_mesh")
async def reconstruct_mesh(
    image: UploadFile = File(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    target_points: int = Form(2048),
    output_dir: str = Form(""),
    mesh_method: str = Form("bpa"),  # "bpa" or "poisson"
    object_size_mm: float = Form(0.0),  # 0=自動(200mm), >0=指定サイズ
):
    """
    クライアント互換エンドポイント: SAM-3D でメッシュ生成 + SAM-6D テンプレートレンダリング

    レスポンス: PLY バイナリ
    ヘッダ:
        X-Mesh-Path:      サーバ側の .ply パス
        X-Template-Dir:   SAM-6D テンプレートディレクトリ
        X-Mask-Center-U:  マスク重心 U 座標
        X-Mask-Center-V:  マスク重心 V 座標
    """
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="モデルがロードされていません")

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(status_code=400, detail="画像のデコードに失敗しました")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    prompt_point = np.array([[click_x if click_x >= 0 else w // 2,
                               click_y if click_y >= 0 else h // 2]])

    import time
    t0 = time.time()

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(rgb)
        masks, scores, _ = sam_predictor.predict(
            point_coords=prompt_point,
            point_labels=np.array([1]),
            multimask_output=True,
        )
    best_mask = masks[np.argmax(scores)]  # スコア最大のマスクを使用
    t_sam = time.time()
    print(f"[Server] SAM マスク完了 (面積:{best_mask.sum()}px, score={scores[np.argmax(scores)]:.3f}) [{t_sam - t0:.1f}s]")

    # SAM2 マスクを保存 (3枚 + スコア表示)
    save_dir = output_dir if output_dir else os.path.join(_host_tmp, "server_reconstructions")
    os.makedirs(save_dir, exist_ok=True)
    for _i in range(3):
        _m = cv2.cvtColor(masks[_i].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        cv2.putText(_m, f"mask_{_i+1}  score={scores[_i]:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_dir, f"mask_sam2_{_i+1}.png"), _m)
    mask_sam2_path = os.path.join(save_dir, "mask_sam2.png")
    cv2.imwrite(mask_sam2_path, (best_mask.astype(np.uint8) * 255))
    print(f"[Server] SAM2 マスク保存: {_rel(save_dir)}/mask_sam2_{{1,2,3}}.png")

    # SAM-3D をロード → 推論 → 即削除してGPUを解放
    output = _load_sam3d_and_run(rgb, best_mask, seed)
    t_sam3d = time.time()
    print(f"[Server] SAM-3D 推論完了 [{t_sam3d - t_sam:.1f}s]")

    # 共有tmpに保存してDockerからアクセスできるようにする
    os.makedirs(save_dir, exist_ok=True)
    ply_path = os.path.join(save_dir, f"object_seed{seed}.ply")
    # GS点群をPLYに保存
    import open3d as o3d
    output["gs"].save_ply(ply_path)
    print(f"[Server] GS PLY 保存: {_rel(ply_path)}")

    # 点群 → メッシュ変換
    print("[Server] 点群をメッシュに変換中 (BPA)...")
    t_bpa_start = time.time()
    gs_ply = o3d.io.read_point_cloud(ply_path)
    n_pts = len(gs_ply.points)
    print(f"[Server] 元の点数: {n_pts}")

    # GS PLY の f_dc_* (球面調和DC成分) から RGB 色を復元して点群に付与
    if not gs_ply.has_colors():
        try:
            from plyfile import PlyData
            ply_data = PlyData.read(ply_path)
            verts = ply_data['vertex']
            if 'f_dc_0' in verts.data.dtype.names:
                f_dc = np.stack([verts['f_dc_0'], verts['f_dc_1'], verts['f_dc_2']], axis=1)
                colors = f_dc / (2 * np.sqrt(np.pi)) + 0.5  # SH DC → linear RGB
                colors = np.clip(colors, 0.0, 1.0)
                gs_ply.colors = o3d.utility.Vector3dVector(colors)
                print(f"[Server] GS f_dc から色復元完了 ({n_pts} 点)")
        except Exception as e:
            print(f"[Server] 色復元スキップ: {e}")

    # 全点群を保存
    pcd_full_path = ply_path.replace(".ply", "_pcd_full.ply")
    o3d.io.write_point_cloud(pcd_full_path, gs_ply)
    print(f"[Server] 全点群保存: {_rel(pcd_full_path)}")

    # 10000点にダウンサンプリング
    if n_pts > 10000:
        gs_ply = gs_ply.random_down_sample(10000 / n_pts)
    pcd_path = ply_path.replace(".ply", "_pcd.ply")
    o3d.io.write_point_cloud(pcd_path, gs_ply)
    print(f"[Server] ダウンサンプル後: {len(gs_ply.points)} points → {_rel(pcd_path)}")

    gs_ply.estimate_normals()
    gs_ply.orient_normals_consistent_tangent_plane(k=15)

    print(f"[Server] メッシュ生成方法: {mesh_method}")
    if mesh_method == "knn":
        from sklearn.neighbors import NearestNeighbors
        k = 10
        pts = np.asarray(gs_ply.points)
        normals = np.asarray(gs_ply.normals)
        print(f"[Server] KNN メッシュ構築 (k={k}, points={len(pts)})...")
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(pts)
        _, indices = nbrs.kneighbors(pts)
        faces_set = set()
        for i, neighbors in enumerate(indices):
            nn = neighbors[1:]
            for j in range(len(nn)):
                for k_idx in range(j + 1, len(nn)):
                    face = tuple(sorted([i, nn[j], nn[k_idx]]))
                    faces_set.add(face)
        faces = list(faces_set)
        print(f"[Server] KNN 三角形数: {len(faces)}")
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(pts)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(np.array(faces, dtype=np.int32))
        mesh_o3d.vertex_normals = o3d.utility.Vector3dVector(normals)
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_unreferenced_vertices()
        t_mesh_end = time.time()
        print(f"[Server] KNN 完了 [{t_mesh_end - t_bpa_start:.1f}s]")
    elif mesh_method == "poisson":
        mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            gs_ply, depth=8
        )
        # 密度の低い頂点 (外れ値) を除去
        density_thresh = np.quantile(np.asarray(densities), 0.1)
        mesh_o3d = mesh_o3d.select_by_index(
            np.where(np.asarray(densities) >= density_thresh)[0]
        )
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_unreferenced_vertices()
        t_mesh_end = time.time()
        print(f"[Server] Poisson 完了 [{t_mesh_end - t_bpa_start:.1f}s]")
    else:
        # BPA
        bbox = gs_ply.get_axis_aligned_bounding_box()
        r = max(bbox.get_extent()) * 0.2
        radii = [r, r * 2]
        mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            gs_ply, o3d.utility.DoubleVector(radii)
        )
        mesh_o3d.remove_degenerate_triangles()
        mesh_o3d.remove_unreferenced_vertices()
        t_mesh_end = time.time()
        print(f"[Server] BPA 完了 [{t_mesh_end - t_bpa_start:.1f}s]")

    # 点群から頂点への色転送 (KNN最近傍)
    if gs_ply.has_colors():
        from sklearn.neighbors import NearestNeighbors
        pcd_pts = np.asarray(gs_ply.points)
        pcd_colors = np.asarray(gs_ply.colors)  # [0,1] float
        mesh_pts = np.asarray(mesh_o3d.vertices)
        nbrs = NearestNeighbors(n_neighbors=1).fit(pcd_pts)
        _, indices = nbrs.kneighbors(mesh_pts)
        mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(pcd_colors[indices.flatten()])
        print(f"[Server] 点群から頂点色転送完了 ({len(mesh_pts)} 頂点)")
    else:
        print("[Server] 点群に色情報なし。頂点色なしでメッシュ保存")

    # SAM-3D はmm単位ではなく正規化座標で出力するため、mm単位にスケール変換
    # SAM-6D の get_test_data は /1000 して mm→m を仮定している
    bbox = mesh_o3d.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()  # [x, y, z]
    # 常に最長辺=200mmに正規化 (object_size_mmはpose_estimate側でのみ使用)
    ref_extent = max(extent)
    if ref_extent > 0:
        scale = 200.0 / ref_extent
        mesh_o3d.scale(scale, center=bbox.get_center())
    # SAM-6D はモデル原点=物体中心を前提とするため原点に移動
    center = mesh_o3d.get_axis_aligned_bounding_box().get_center()
    mesh_o3d.translate(-center)
    print(f"[Server] メッシュスケール変換 (最長辺): {ref_extent:.4f} → 200mm (原点中心化済み)")

    mesh_path = ply_path.replace(".ply", "_mesh.ply")
    o3d.io.write_triangle_mesh(mesh_path, mesh_o3d)
    t_mesh = time.time()
    print(f"[Server] メッシュ保存: {_rel(mesh_path)} ({len(mesh_o3d.triangles)} triangles) [合計: {t_mesh - t0:.1f}s]")

    ys, xs = np.where(best_mask)
    mask_center_u = int(xs.mean())
    mask_center_v = int(ys.mean())

    # SAM-6D テンプレートレンダリング (点群直接投影, Blenderproc不要)
    print("[Server] SAM-6D テンプレートレンダリング中 (点群直接投影)...")
    tdir_resp = _sam6d_post("render_templates", {
        "cad_path": to_docker_path(mesh_path),
        "output_dir": None,
        "num_templates": 42,
        "pcd_path": to_docker_path(pcd_full_path),
    }, timeout=600.0)
    template_dir = tdir_resp["template_dir"]
    print(f"[Server] テンプレート完了: {template_dir}")

    import base64
    with open(mesh_path, "rb") as f:
        ply_b64 = base64.b64encode(f.read()).decode()

    # 3マスク全部を PNG エンコードして返す (確認用)
    masks_b64 = []
    for m in masks:
        _, buf = cv2.imencode(".png", (m.astype(np.uint8) * 255))
        masks_b64.append(base64.b64encode(buf).decode())

    return JSONResponse({
        "ply_b64":        ply_b64,
        "masks_b64":      masks_b64,          # [mask0, mask1, mask2]
        "scores":         scores.tolist(),     # [score0, score1, score2]
        "best_idx":       2,
        "mesh_path":      mesh_path,
        "template_dir":   template_dir,
        "mask_center_u":  mask_center_u,
        "mask_center_v":  mask_center_v,
    })


@app.post("/pose_estimate")
async def pose_estimate(
    rgb_image: UploadFile = File(...),
    depth_image: UploadFile = File(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...),
    mesh_path: str = Form(...),
    template_dir: str = Form(...),
    det_score_thresh: float = Form(0.2),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    object_size_mm: float = Form(0.0),  # 0=深度から自動推定, >0=指定サイズ
):
    """
    クライアント互換エンドポイント: 6DoF 姿勢推定

    depth_image: float32 生バイト列 (H×W×4 bytes, メートル単位)
    """
    import subprocess

    rgb_bytes   = await rgb_image.read()
    depth_bytes = await depth_image.read()

    # RGB デコード
    nparr = np.frombuffer(rgb_bytes, np.uint8)
    bgr   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "RGB画像のデコードに失敗しました")
    h, w = bgr.shape[:2]

    # depth: float32 生バイト → uint16 mm
    depth_f32 = np.frombuffer(depth_bytes, dtype=np.float32).reshape(h, w)
    depth_mm  = (depth_f32 * 1000.0).astype(np.uint16)

    # 共有 tmp ディレクトリに保存 (Docker から /workspace/tmp/ として見える)
    rgb_host   = os.path.join(_host_tmp, "rgb.png")
    depth_host = os.path.join(_host_tmp, "depth.png")
    cam_host   = os.path.join(_host_tmp, "camera_custom.json")

    cv2.imwrite(rgb_host, bgr)
    cv2.imwrite(depth_host, depth_mm)
    cam_json_data = {
        "cam_K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "depth_scale": 1.0,
    }
    with open(cam_host, "w") as f:
        json.dump(cam_json_data, f)

    # ---- SAM2 でマスク生成 → detection_ism.json として保存 ----
    rgb_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    px = click_x if click_x >= 0 else w // 2
    py = click_y if click_y >= 0 else h // 2
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(rgb_np)
        sam2_masks, sam2_scores, _ = sam_predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )
    best_idx = int(np.argmax(sam2_scores))
    best_sam2_mask = sam2_masks[best_idx]
    print(f"[pose_estimate] SAM2 マスク完了 (面積:{best_sam2_mask.sum()}px, score={sam2_scores[best_idx]:.3f})")


    # template_dir (Docker path) → output_dir (テンプレートの親ディレクトリ)
    tdir = template_dir.rstrip("/")
    output_dir_docker = tdir[:-len("/templates")] if tdir.endswith("/templates") else tdir
    output_dir_host = output_dir_docker.replace(_docker_tmp, _host_tmp)
    sam6d_results_dir = os.path.join(output_dir_host, "sam6d_results")
    os.makedirs(sam6d_results_dir, exist_ok=True)

    # テンプレート存在チェック
    templates_host = os.path.join(output_dir_host, "templates")
    if not os.path.isdir(templates_host):
        raise HTTPException(
            500,
            f"テンプレートが見つかりません: {_rel(templates_host)}\n"
            "/reconstruct_mesh を再実行してください。"
        )
    try:
        os.chmod(sam6d_results_dir, 0o777)
    except Exception:
        pass

    # SAM2マスク3枚をスコア付きで保存
    for _i in range(3):
        _m = cv2.cvtColor(sam2_masks[_i].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        cv2.putText(_m, f"sam2_mask_{_i+1}  score={sam2_scores[_i]:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(sam6d_results_dir, f"mask_sam2_{_i+1}.png"), _m)

    # detection_ism.json として保存 (run_demo_custom.sh のSAM1をスキップ)
    import pycocotools.mask as cocomask
    rle = cocomask.encode(np.asfortranarray(best_sam2_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    ys, xs = np.where(best_sam2_mask)
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    detection_ism = [{
        "scene_id": 0, "image_id": 0, "category_id": 0,
        "bbox": [x1, y1, x2 - x1, y2 - y1],
        "score": float(sam2_scores[best_idx]),
        "segmentation": rle,
        "time": 0.0,
    }]
    detection_ism_path = os.path.join(sam6d_results_dir, "detection_ism.json")
    with open(detection_ism_path, "w") as f:
        json.dump(detection_ism, f)
    print(f"[pose_estimate] detection_ism.json 保存 (SAM2): {_rel(detection_ism_path)}")

    # vis_mask.png 生成 (マスクオーバーレイ + クリック点 + bbox)
    vis = rgb_np.copy()
    bool_mask = best_sam2_mask.astype(bool)
    vis[bool_mask] = (vis[bool_mask] * 0.5 + np.array([0, 255, 0]) * 0.5).astype(np.uint8)
    cv2.circle(vis, (px, py), 6, (255, 0, 0), -1)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(os.path.join(sam6d_results_dir, "vis_mask.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[pose_estimate] vis_mask.png 保存: {_rel(sam6d_results_dir)}")

    # Docker パス
    rgb_docker   = f"{_docker_tmp}/rgb.png"
    depth_docker = f"{_docker_tmp}/depth.png"
    cam_docker   = f"{_docker_tmp}/camera_custom.json"

    # RGBDから実スケール推定 → メッシュをスケーリング (model_points/radius に影響)
    # テンプレートxyzはNOCS形式のためスケーリング不要
    # マスク境界ノイズを除くため内側のみ使用（エロージョン半径=マスクサイズに比例）
    _mask_u8 = best_sam2_mask.astype(np.uint8)
    _mask_area = int(_mask_u8.sum())
    _kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # 縁2px除外
    _mask_inner = cv2.erode(_mask_u8, _kernel)
    _ys_i, _xs_i = np.where(_mask_inner.astype(bool))
    if len(_ys_i) < 10:  # エロージョンしすぎた場合はフルマスクにフォールバック
        _mask_inner = _mask_u8
        _ys_i, _xs_i = np.where(_mask_inner.astype(bool))
    print(f"[pose_estimate] マスク内側: 全{_mask_area}px → 内側{len(_ys_i)}px (erode=2px)")
    _Z = depth_f32[_ys_i, _xs_i]
    _valid = _Z > 0
    estimated_size_mm = None
    if _valid.sum() > 10:
        _ys_v = _ys_i[_valid]
        _xs_v = _xs_i[_valid]
        _Zv = _Z[_valid]
        _Xv = (_xs_v - cx) * _Zv / fx
        _Yv = (_ys_v - cy) * _Zv / fy

        # 深度 Z の外れ値除去（中央値±2σ）
        _Z_med = float(np.median(_Zv))
        _Z_std = float(_Zv.std())
        _inlier = np.abs(_Zv - _Z_med) < 2.0 * _Z_std
        if _inlier.sum() >= 10:
            _ys_v, _xs_v = _ys_v[_inlier], _xs_v[_inlier]
            _Zv, _Xv, _Yv = _Zv[_inlier], _Xv[_inlier], _Yv[_inlier]
        print(f"[pose_estimate] 深度外れ値除去: {_valid.sum()}px → {len(_Zv)}px "
              f"(Z_med={_Z_med*1000:.0f}mm ±{_Z_std*1000:.0f}mm×2)")

        # 高さ（Y方向）はパーセンタイルで推定（端点の外れ値に頑健）
        _Y_lo = float(np.percentile(_Yv, 2))
        _Y_hi = float(np.percentile(_Yv, 98))
        estimated_size_mm = (_Y_hi - _Y_lo) * 1000.0
        print(f"[pose_estimate] 推定物体サイズ: {estimated_size_mm:.1f}mm (高さY, pct2-98) "
              f"Y:[{_Y_lo*1000:.0f}, {_Y_hi*1000:.0f}]mm")

        # 高さ推定に用いた最高・最低点を可視化して保存
        _idx_top = int(np.argmin(np.abs(_Yv - _Y_lo)))  # 2パーセンタイルに最も近い点
        _idx_bot = int(np.argmin(np.abs(_Yv - _Y_hi)))  # 98パーセンタイルに最も近い点
        _pt_top = (int(_xs_v[_idx_top]), int(_ys_v[_idx_top]))
        _pt_bot = (int(_xs_v[_idx_bot]), int(_ys_v[_idx_bot]))
        vis_h = rgb_np.copy()
        _inner_bool = _mask_inner.astype(bool)
        vis_h[_inner_bool] = (vis_h[_inner_bool] * 0.6 + np.array([80, 180, 255]) * 0.4).astype(np.uint8)
        cv2.circle(vis_h, _pt_top, 10, (0, 255, 0), -1)
        cv2.putText(vis_h, f"top {_Yv[_idx_top]*1000:.0f}mm",
                    (_pt_top[0] + 8, _pt_top[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.circle(vis_h, _pt_bot, 10, (0, 50, 255), -1)
        cv2.putText(vis_h, f"bot {_Yv[_idx_bot]*1000:.0f}mm",
                    (_pt_bot[0] + 8, _pt_bot[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 50, 255), 2)
        cv2.putText(vis_h, f"height={estimated_size_mm:.1f}mm  pct2-98  erode=2px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 220, 0), 2)
        cv2.imwrite(os.path.join(sam6d_results_dir, "vis_height.png"),
                    cv2.cvtColor(vis_h, cv2.COLOR_RGB2BGR))
        print(f"[pose_estimate] vis_height.png 保存")

    # object_size_mm が指定されている場合は深度推定より優先
    if object_size_mm > 0:
        estimated_size_mm = object_size_mm
        print(f"[pose_estimate] 指定サイズ使用: {estimated_size_mm:.1f}mm")

    mesh_host_path = mesh_path.replace(_docker_tmp, _host_tmp)
    if estimated_size_mm is not None:
        import trimesh as _trimesh
        import glob as _glob
        # メッシュのZ軸長を取得してscale_factorを計算 (メッシュZ=物体高さ)
        _m = _trimesh.load_mesh(mesh_host_path)
        _z_extent = _m.bounding_box.extents[2]  # Z軸方向の長さ [mm]
        if _z_extent > 0:
            scale_factor = estimated_size_mm / _z_extent
        else:
            scale_factor = estimated_size_mm / 200.0
        # メッシュをスケーリングして保存
        scaled_mesh_host = mesh_host_path.replace(".ply", "_scaled.ply")
        _m.apply_scale(scale_factor)
        _m.export(scaled_mesh_host)
        mesh_path_for_pem = scaled_mesh_host.replace(_host_tmp, _docker_tmp)
        # テンプレートxyz を元ディレクトリに直接上書き (新規ファイルとして書き込む)
        tem_dir = os.path.join(output_dir_host, "templates")
        for _xyz_path in _glob.glob(os.path.join(tem_dir, "xyz_*.npy")):
            _xyz = np.load(_xyz_path).astype(np.float32)
            _out = os.path.join(tem_dir, os.path.basename(_xyz_path))
            try:
                os.remove(_out)
            except OSError:
                pass
            np.save(_out, _xyz * scale_factor)
        output_dir_docker_for_pem = output_dir_docker
        print(f"[pose_estimate] スケール (カメラY→メッシュZ): {_z_extent:.1f}mm → {estimated_size_mm:.1f}mm (factor={scale_factor:.3f})")
    else:
        mesh_path_for_pem = mesh_host_path.replace(_host_tmp, _docker_tmp)
        output_dir_docker_for_pem = output_dir_docker

    script_docker = "/workspace/SAM-6D/SAM-6D/run_demo_custom.sh"
    cmd = [
        "docker", "exec", "sam6d_service",
        "bash", script_docker,
        mesh_path_for_pem,
        output_dir_docker_for_pem,
        str(click_x),
        str(click_y),
        rgb_docker,
        depth_docker,
        cam_docker,
    ]
    print(f"[pose_estimate] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HTTPException(500, f"run_demo_custom.sh 失敗:\n{proc.stderr[-2000:]}")

    # 結果 JSON 読み込み
    result_json_path = os.path.join(sam6d_results_dir, "detection_pem.json")
    if not os.path.exists(result_json_path):
        raise HTTPException(
            500,
            f"pose 推定結果が見つかりません: {_rel(result_json_path)}\n"
            f"stdout: {proc.stdout[-1000:]}\nstderr: {proc.stderr[-1000:]}"
        )
    with open(result_json_path, "r") as f:
        detections = json.load(f)

    if not detections:
        raise HTTPException(500, "pose 推定失敗: detection が空です")

    best  = max(detections, key=lambda d: d["score"])
    R_list = best["R"]                               # 3×3 list
    t_m    = [v / 1000.0 for v in best["t"]]        # mm → m

    # ---- 画像生成 ----
    import base64
    import open3d as o3d

    # vis_pem と同じ描画スタイルのヘルパー関数 (draw_utils.py 移植)
    def _proj(pts_mm, R, t_mm, K):
        """pts_mm: (3,N) mm座標 → (N,2) 画像座標"""
        cam = R @ pts_mm + t_mm[:, np.newaxis]
        p = K @ cam
        if np.any(p[2] == 0) or np.any(~np.isfinite(p)):
            import os
            print(f"[_proj] 無効な深度値を検出 (Z=0/NaN/Inf)。プロセスを終了します。")
            os._exit(1)
        return (p[:2] / p[2]).T.astype(np.int32)

    def _draw_bbox(img, pts8x2, color, size=2):
        """8頂点bboxを3段階の明度で描画 (vis_pem スタイル)"""
        c_dark = tuple(int(c * 0.3) for c in color)
        c_mid  = tuple(int(c * 0.6) for c in color)
        for i, j in [(4,5),(5,7),(7,4),(4,6)]:  # ground
            cv2.line(img, tuple(pts8x2[i]), tuple(pts8x2[j]), c_dark, size)
        for i, j in [(0,4),(1,5),(2,6),(3,7)]:  # pillars
            cv2.line(img, tuple(pts8x2[i]), tuple(pts8x2[j]), c_mid, size)
        for i, j in [(0,1),(1,3),(3,2),(2,0)]:  # top
            cv2.line(img, tuple(pts8x2[i]), tuple(pts8x2[j]), color, size)
        return img

    def _draw_axes(img, R, t_mm, K, length_mm):
        """座標軸を赤(X)緑(Y)青(Z)の矢印で描画。Z(青)軸が画像下向きなら表示のみ上向きに反転。"""
        origin = _proj(t_mm[:, np.newaxis], np.eye(3), np.zeros(3), K)[0]
        for i, c in enumerate([(0,0,255),(0,255,0),(255,0,0)]):  # BGR: x=赤,y=緑,z=青
            end_mm = t_mm + R[:, i] * length_mm
            ep = _proj(end_mm[:, np.newaxis], np.eye(3), np.zeros(3), K)[0]
            if i == 2 and ep[1] > origin[1]:  # Z(青)軸が画像下向き → 表示のみ反転
                ep = np.array([ep[0], 2 * origin[1] - ep[1]], dtype=np.int32)
            cv2.arrowedLine(img, tuple(origin), tuple(ep), c, 2, tipLength=0.3)
        return img

    def _make_vis(rgb_bgr, R, t_mm, pts_mm, K, pcd_color=(0,0,255), bbox_color=(0,0,255), with_axes=True):
        """vis_pem スタイル: 左=元画像 右=可視化 横並び (pts_mm: (N,3))"""
        vis = rgb_bgr.copy()
        # 点群投影 (1000点にダウンサンプル)
        choose = np.random.choice(len(pts_mm), min(len(pts_mm), 1000), replace=False)
        p2d = _proj(pts_mm[choose].T, R, t_mm, K)
        in_b = (p2d[:,0]>=0)&(p2d[:,0]<w)&(p2d[:,1]>=0)&(p2d[:,1]<h)
        for u, v in p2d[in_b]:
            cv2.circle(vis, (int(u), int(v)), 1, pcd_color, -1)
        # bbox
        mins, maxs = pts_mm.min(0), pts_mm.max(0)
        shift = (mins + maxs) / 2
        scale = maxs - mins
        corners = np.array([[-1,-1,-1],[ 1,-1,-1],[-1, 1,-1],[ 1, 1,-1],
                             [-1,-1, 1],[ 1,-1, 1],[-1, 1, 1],[ 1, 1, 1]],
                            dtype=np.float32) * (scale / 2) + shift
        bbox2d = _proj(corners.T, R, t_mm, K)
        _draw_bbox(vis, bbox2d, bbox_color)
        # 座標軸
        if with_axes:
            _draw_axes(vis, R, t_mm, K, length_mm=np.max(scale) * 0.6)
        # 左:元画像 右:可視化 横並び (vis_pem と同レイアウト)
        concat = np.concatenate([rgb_bgr, vis], axis=1)
        return concat

    R_np = np.array(R_list, dtype=np.float32)
    # Z軸を上向き(R[1,2]<0)に統一 — mesh Z = 物体高さ軸 → カメラ-Y(画像上)に揃える
    # 下向き(R[1,2]>0)のときだけ反転してUPに揃える
    r12_raw = float(R_np[1, 2])
    if R_np[1, 2] > 0:
        R_np[:, 2] *= -1
        print(f"[pose_estimate] Z軸反転 (DOWN→UP): R[1,2]={r12_raw:.3f} → {R_np[1,2]:.3f}")
    else:
        print(f"[pose_estimate] Z軸反転なし (すでにUP): R[1,2]={r12_raw:.3f}")
    R_list = R_np.tolist()

    # 可視化用: R_npは常にZ上向き(R[1,2]<0) → そのまま表示 (↑向きなので反転不要)
    R_vis = R_np.copy()
    print(f"[pose_estimate] 可視化用: R_vis[1,2]={R_vis[1,2]:.3f}")

    t_mm_np = np.array(best["t"], dtype=np.float32)   # mm単位 (vis_pemと同じ)
    K_np = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    mesh_host = mesh_path_for_pem.replace(_docker_tmp, _host_tmp)
    img1_b64 = ""
    try:
        mesh_o3d = o3d.io.read_triangle_mesh(mesh_host)
        if len(mesh_o3d.triangles) > 0:
            pcd = mesh_o3d.sample_points_uniformly(number_of_points=5000)
        else:
            pcd = o3d.io.read_point_cloud(mesh_host)
        pts_mm = np.asarray(pcd.points, dtype=np.float32)  # mm単位
        concat1 = _make_vis(bgr, R_vis, t_mm_np, pts_mm, K_np, pcd_color=(0,255,0), bbox_color=(0,255,255), with_axes=True)
        _, buf1 = cv2.imencode(".png", concat1)
        img1_b64 = base64.b64encode(buf1).decode()
        print("[pose_estimate] 画像1 (vis_pemスタイル) 生成完了")
    except Exception as e:
        print(f"[pose_estimate] 画像1 生成失敗: {e}")

    img2_b64 = ""
    try:
        mesh_o3d2 = o3d.io.read_triangle_mesh(mesh_host)
        if len(mesh_o3d2.triangles) > 0:
            pcd2 = mesh_o3d2.sample_points_uniformly(number_of_points=20000)
        else:
            pcd2 = o3d.io.read_point_cloud(mesh_host)
        pts_mm2 = np.asarray(pcd2.points, dtype=np.float32)
        concat2 = _make_vis(bgr, R_vis, t_mm_np, pts_mm2, K_np, pcd_color=(0,255,0), bbox_color=(0,255,255), with_axes=False)
        _, buf2 = cv2.imencode(".png", concat2)
        img2_b64 = base64.b64encode(buf2).decode()
        print("[pose_estimate] 画像2 (高密度メッシュ) 生成完了")
    except Exception as e:
        print(f"[pose_estimate] 画像2 生成失敗: {e}")

    return JSONResponse({
        "success":    True,
        "R":          R_list,
        "t":          t_m,
        "mask_area":  0,
        "img_pose":   img1_b64,   # 点群+bbox投影
        "img_mesh":   img2_b64,   # メッシュ面投影
    })


@app.get("/sam6d/health")
def sam6d_health():
    """SAM-6D Docker サービスの死活確認"""
    return _sam6d_post("health", {})


@app.post("/render_templates")
async def render_templates(
    cad_path: str = Form(...),
    output_dir: str = Form(""),
    num_templates: int = Form(42),
):
    """
    CADモデル (.ply) からテンプレートをレンダリングする (物体ごとに一度)

    Args:
        cad_path:      サーバ上の PLY ファイルパス [mm単位]
        output_dir:    テンプレート保存先 (空文字で自動生成)
        num_templates: テンプレート数 (デフォルト 42)

    Returns:
        {"template_dir": "/path/to/templates"}
    """
    payload = {
        "cad_path": cad_path,
        "output_dir": output_dir if output_dir else None,
        "num_templates": num_templates,
    }
    return JSONResponse(_sam6d_post("render_templates", payload))


@app.post("/estimate_pose")
async def estimate_pose(
    rgb: UploadFile = File(...),
    depth: UploadFile = File(...),
    intrinsics_json: str = Form(...),   # JSON文字列: {"fx","fy","cx","cy"}
    cad_path: str = Form(...),
    template_dir: str = Form(...),
    det_score_thresh: float = Form(0.2),
):
    """
    RGB + 深度画像から 6DoF 姿勢推定

    Args:
        rgb:              RGB 画像 (PNG/JPEG)
        depth:            深度画像 (uint16 PNG, mm 単位)
        intrinsics_json:  カメラ内部パラメータ JSON
        cad_path:         サーバ上の CAD (.ply) パス [mm]
        template_dir:     render_templates() の出力ディレクトリ
        det_score_thresh: 検出スコア閾値

    Returns:
        {"R": [[...]], "t": [...], "mask_area": int}
    """
    import tempfile, shutil

    tmpdir = tempfile.mkdtemp(dir=_host_tmp)
    try:
        # アップロードファイルを一時保存
        rgb_path = os.path.join(tmpdir, "rgb.png")
        depth_path = os.path.join(tmpdir, "depth.png")
        cam_path = os.path.join(tmpdir, "camera.json")

        rgb_bytes = await rgb.read()
        with open(rgb_path, "wb") as f:
            f.write(rgb_bytes)

        depth_bytes = await depth.read()
        with open(depth_path, "wb") as f:
            f.write(depth_bytes)

        intrinsics = json.loads(intrinsics_json)
        cam_json = {
            "cam_K": [intrinsics["fx"], 0.0, intrinsics["cx"],
                      0.0, intrinsics["fy"], intrinsics["cy"],
                      0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        with open(cam_path, "w") as f:
            json.dump(cam_json, f)

        payload = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
            "cam_json_path": cam_path,
            "cad_path": cad_path,
            "template_dir": template_dir,
            "det_score_thresh": det_score_thresh,
        }
        result = _sam6d_post("estimate_pose", payload, timeout=300.0)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return JSONResponse(result)


@app.post("/full_pipeline")
async def full_pipeline(
    image: UploadFile = File(...),
    depth: UploadFile = File(...),
    intrinsics_json: str = Form(...),   # {"fx","fy","cx","cy"}
    click_x: int = Form(-1),
    click_y: int = Form(-1),
    seed: int = Form(42),
    target_points: int = Form(2048),
    det_score_thresh: float = Form(0.2),
    output_dir: str = Form("tmp/pipeline"),
):
    """
    フルパイプライン: SAM-3D 再構成 → SAM-6D 姿勢推定 を一括実行

    Args:
        image:           RGB 画像 (PNG/JPEG)
        depth:           深度画像 (uint16 PNG, mm 単位)
        intrinsics_json: カメラ内部パラメータ JSON {"fx","fy","cx","cy"}
        click_x, click_y: SAM プロンプト座標 (-1,-1 で中央)
        seed:            sam-3d-objects シード
        target_points:   点群点数
        det_score_thresh: SAM-6D 検出閾値

    Returns:
        {
            "points":       [[x,y,z], ...],   # SAM-3D 点群 (object 座標系)
            "ply_path":     "/path/to/obj.ply",
            "template_dir": "/path/to/templates",
            "R":            [[...], [...], [...]],
            "t":            [x, y, z],          # カメラ座標系 [m]
            "mask_center_u": int,
            "mask_center_v": int,
        }
    """
    if sam_predictor is None:
        raise HTTPException(503, "SAM モデルがロードされていません")

    import tempfile, shutil

    # ---- Step 1: SAM-3D で PLY 生成 ----
    image_bytes = await image.read()
    depth_bytes = await depth.read()

    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise HTTPException(400, "RGB画像のデコードに失敗しました")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    prompt_point = np.array([[click_x if click_x >= 0 else w // 2,
                               click_y if click_y >= 0 else h // 2]])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(rgb)
        masks, scores, _ = sam_predictor.predict(
            point_coords=prompt_point,
            point_labels=np.array([1]),
            multimask_output=True,
        )
    best_mask = masks[np.argmax(scores)]  # スコア最大のマスクを使用
    print(f"[Pipeline] SAM2 マスク完了 (面積:{best_mask.sum()}px, score={scores[np.argmax(scores)]:.3f})")

    # SAM2 マスクを保存 (3枚 + スコア表示)
    os.makedirs(output_dir, exist_ok=True)
    for _i in range(3):
        _m = cv2.cvtColor(masks[_i].astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
        cv2.putText(_m, f"mask_{_i+1}  score={scores[_i]:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imwrite(os.path.join(output_dir, f"mask_sam2_{_i+1}.png"), _m)
    mask_sam2_path = os.path.join(output_dir, "mask_sam2.png")
    cv2.imwrite(mask_sam2_path, (best_mask.astype(np.uint8) * 255))
    print(f"[Pipeline] SAM2 マスク保存: {output_dir}/mask_sam2_{{1,2,3}}.png")

    # SAM-3D をロード → 推論 → 即削除してGPUを解放
    recon_output = _load_sam3d_and_run(rgb, best_mask, seed)

    os.makedirs(output_dir, exist_ok=True)
    ply_path = os.path.join(output_dir, f"object_seed{seed}.ply")
    recon_output["gs"].save_ply(ply_path)
    print(f"[Pipeline] PLY 保存: {_rel(ply_path)}")

    from plyfile import PlyData
    ply_data = PlyData.read(ply_path)
    vertex = ply_data["vertex"]
    points = np.stack([vertex["x"], vertex["y"], vertex["z"]], axis=-1).astype(np.float32)
    n = len(points)
    choice = np.random.choice(n, target_points, replace=(n < target_points))
    points = points[choice]

    ys, xs = np.where(best_mask)
    mask_center_u = int(xs.mean())
    mask_center_v = int(ys.mean())

    # ---- Step 2: 深度・カメラパラメータを一時保存 ----
    tmpdir = tempfile.mkdtemp(dir=output_dir)
    try:
        rgb_path_tmp   = os.path.join(tmpdir, "rgb.png")
        depth_path_tmp = os.path.join(tmpdir, "depth.png")
        cam_path_tmp   = os.path.join(tmpdir, "camera.json")

        cv2.imwrite(rgb_path_tmp, bgr)

        depth_arr = np.frombuffer(depth_bytes, np.uint8)
        depth_img = cv2.imdecode(depth_arr, cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            raise HTTPException(400, "深度画像のデコードに失敗しました")
        cv2.imwrite(depth_path_tmp, depth_img)

        intrinsics = json.loads(intrinsics_json)
        cam_json = {
            "cam_K": [intrinsics["fx"], 0.0, intrinsics["cx"],
                      0.0, intrinsics["fy"], intrinsics["cy"],
                      0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        with open(cam_path_tmp, "w") as f:
            json.dump(cam_json, f)

        # ---- Step 3: テンプレートレンダリング (キャッシュ済みなら省略) ----
        tdir_resp = _sam6d_post("render_templates", {
            "cad_path": ply_path,
            "output_dir": None,
            "num_templates": 42,
        }, timeout=600.0)
        template_dir = tdir_resp["template_dir"]
        print(f"[Pipeline] テンプレートディレクトリ: {template_dir}")

        # ---- Step 4: SAM-6D 姿勢推定 ----
        pose_resp = _sam6d_post("estimate_pose", {
            "rgb_path": rgb_path_tmp,
            "depth_path": depth_path_tmp,
            "cam_json_path": cam_path_tmp,
            "cad_path": ply_path,
            "template_dir": template_dir,
            "det_score_thresh": det_score_thresh,
            "click_x": mask_center_u,
            "click_y": mask_center_v,
        }, timeout=300.0)
        print(f"[Pipeline] 姿勢推定完了")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return JSONResponse({
        "points":        points.tolist(),
        "num_points":    len(points),
        "ply_path":      ply_path,
        "template_dir":  template_dir,
        "R":             pose_resp["R"],
        "t":             pose_resp["t"],
        "mask_area":     pose_resp["mask_area"],
        "mask_center_u": mask_center_u,
        "mask_center_v": mask_center_v,
    })


@app.post("/segment_only")
async def segment_only(
    image: UploadFile = File(...),
    click_x: int = Form(-1),
    click_y: int = Form(-1),
):
    """
    SAMのマスクのみ返す (デバッグ用)
    """
    if sam_predictor is None:
        raise HTTPException(status_code=503, detail="SAMがロードされていません")

    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    prompt_point = np.array([[click_x if click_x >= 0 else w // 2,
                               click_y if click_y >= 0 else h // 2]])
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        sam_predictor.set_image(rgb)
        masks, scores, _ = sam_predictor.predict(
            point_coords=prompt_point,
            point_labels=np.array([1]),
            multimask_output=True,
        )
    best_mask = masks[1]  # mask index 1 (中スケール) を使用
    return JSONResponse({
        "mask_area": int(best_mask.sum()),
        "score": float(scores[1]),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam-checkpoint", default=None,
                        help="SAM2 モデル重みパス (.pt) "
                             "(省略時: {sam3d-repo}/../../sam2_checkpoints/sam2.1_hiera_large.pt)")
    parser.add_argument("--sam3d-repo",
                        default=os.path.join(_SERVER_DIR, "sam-3d-objects"),
                        help="sam-3d-objects リポジトリのパス (省略時: server/sam-3d-objects)")
    parser.add_argument("--sam3d-config", default=None,
                        help="sam-3d-objects の pipeline.yaml パス (省略時: {sam3d-repo}/checkpoints/hf/pipeline.yaml)")
    parser.add_argument("--sam6d-service", default="http://localhost:8081")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host-tmp", default=os.path.join(_SERVER_DIR, "tmp"))
    parser.add_argument("--docker-tmp", default="/workspace/tmp")
    args = parser.parse_args()

    # 省略引数を sam3d-repo から自動導出
    if args.sam3d_config is None:
        args.sam3d_config = os.path.join(args.sam3d_repo, "checkpoints", "hf", "pipeline.yaml")
    if args.sam_checkpoint is None:
        args.sam_checkpoint = os.path.join(_SERVER_DIR, "sam2_checkpoints", "sam2.1_hiera_large.pt")

    args_global = args
    _sam6d_url  = args.sam6d_service
    _host_tmp   = args.host_tmp
    _docker_tmp = args.docker_tmp

    print("=" * 50)
    print(f"  SAM 3D + SAM-6D Pipeline Server")
    print(f"  device:        {args.device}")
    print(f"  host:port:     {args.host}:{args.port}")
    print(f"  sam6d_service: {_sam6d_url}")
    print(f"  sam_checkpoint:{args.sam_checkpoint}")
    print(f"  sam3d_repo:    {args.sam3d_repo}")
    print(f"  sam3d_config:  {args.sam3d_config}")
    print(f"  host_tmp:      {_host_tmp}")
    print(f"  docker_tmp:    {_docker_tmp}")
    print("=" * 50)

    load_models(args.sam_checkpoint, args.sam3d_config, args.sam3d_repo, args.device)

    uvicorn.run(app, host=args.host, port=args.port)
