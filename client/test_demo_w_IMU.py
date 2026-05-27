"""
IMU を用いた自動高さ推定付きパイプラインテスト

test_demo.py の拡張版。--object-size の手入力の代わりに、
RealSense IMU（加速度センサー）で重力方向を取得し、
SAM マスク + 深度画像 + カメラパラメータから物体高さを自動計算する。

使用方法:
    # フルモード（IMU自動取得）
    python test_demo_w_IMU.py --mode full \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --click-x 400 --click-y 280

    # 重力ベクトルを手動指定（IMU不使用 / カメラなし環境）
    python test_demo_w_IMU.py --mode full \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --gravity 0 -1 0

    # 高さ推定をスキップ（従来の --object-size 指定）
    python test_demo_w_IMU.py --mode full \
        --rgb test_data/rgb.png \
        --depth test_data/depth.png \
        --object-size 15.5
"""

import argparse
import os
import sys
import yaml
import numpy as np
import cv2

os.chdir(os.path.dirname(os.path.abspath(__file__)))


# ===========================================================================
# 設定ロード
# ===========================================================================

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_intrinsics(args, img_w: int, img_h: int):
    """intrinsics を読み込む。cam.json に gravity が含まれていれば args.gravity に設定する。"""
    import json as _json
    from utils.coord_transform import CameraIntrinsics

    if args.cam_json:
        with open(args.cam_json, "r") as f:
            cam = _json.load(f)
        K = cam["cam_K"]
        if len(K) == 9:
            fx, cx = K[0], K[2]
            fy, cy = K[4], K[5]
        else:
            raise ValueError(f"cam_K の形式が不正: {K}")
        if "depth_scale" in cam and args.depth_scale == 0.001:
            args.depth_scale = cam["depth_scale"]
        # gravity が JSON に含まれていて --gravity 未指定なら自動設定
        if "gravity" in cam and args.gravity is None:
            args.gravity = cam["gravity"]
            print(f"[intrinsics] cam_json から gravity 読み込み: {args.gravity}")
        print(f"[intrinsics] cam_json から読み込み: fx={fx} fy={fy} cx={cx} cy={cy}  depth_scale={args.depth_scale}")
    else:
        fx = args.fx
        fy = args.fy
        cx = args.cx if args.cx > 0 else img_w / 2
        cy = args.cy if args.cy > 0 else img_h / 2

    return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=img_w, height=img_h)


def load_depth(depth_path: str, depth_scale: float = 1.0) -> np.ndarray:
    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise FileNotFoundError(f"深度画像が読み込めません: {depth_path}")
    depth_m = depth_raw.astype(np.float32) * depth_scale
    print(f"[test] 深度画像ロード: {depth_path}  shape={depth_raw.shape}  dtype={depth_raw.dtype}  "
          f"range=[{depth_m.min():.3f}, {depth_m.max():.3f}] m")
    return depth_m


# ===========================================================================
# IMU・高さ計算ユーティリティ
# ===========================================================================

def get_gravity_imu(n_samples: int = 30) -> np.ndarray:
    """
    RealSense 加速度センサーから重力方向の単位ベクトルを取得する。

    Returns:
        gravity_vec: (3,) 重力方向の単位ベクトル（RealSense カメラ座標系）
    """
    try:
        import pyrealsense2 as rs
    except ImportError:
        raise RuntimeError(
            "pyrealsense2 がインストールされていません。\n"
            "  pip install pyrealsense2\n"
            "または --gravity で重力ベクトルを手動指定してください。"
        )

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 100)

    print(f"[IMU] RealSense IMU を起動して重力ベクトルを取得中 ({n_samples} サンプル)...")
    profile = pipeline.start(cfg)
    samples = []
    collected = 0
    try:
        while collected < n_samples:
            frames = pipeline.wait_for_frames()
            accel_frame = frames.first_or_default(rs.stream.accel)
            if not accel_frame:
                continue
            motion = accel_frame.as_motion_frame().get_motion_data()
            samples.append([motion.x, motion.y, motion.z])
            collected += 1
    finally:
        pipeline.stop()

    g = np.mean(samples, axis=0)
    g = g / np.linalg.norm(g)
    print(f"[IMU] 重力ベクトル g = [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}]")
    return g.astype(np.float64)


def get_points_3d_from_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    min_depth: float = 0.05,
    max_depth: float = 5.0,
) -> np.ndarray:
    """
    マスク領域内の深度画像を 3D 点群に変換する（ベクトル化・高速）。

    Args:
        depth_m: (H, W) float32, メートル単位の深度画像
        mask:    (H, W) bool または uint8 (>0 が物体領域)
        fx, fy, cx, cy: カメラ内部パラメータ

    Returns:
        points_3d: (N, 3) float64, カメラ座標系の 3D 点群 [m]
                   有効点がない場合は shape (0, 3)
    """
    mask_bool = mask.astype(bool)
    vs, us = np.where(mask_bool)
    if len(vs) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    d = depth_m[vs, us].astype(np.float64)
    valid = (d > min_depth) & (d < max_depth)
    vs, us, d = vs[valid], us[valid], d[valid]

    if len(d) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    X = (us - cx) * d / fx
    Y = (vs - cy) * d / fy
    Z = d
    return np.stack([X, Y, Z], axis=1)


def calc_height_from_points(points_3d: np.ndarray, gravity_vec: np.ndarray) -> float:
    """
    3D 点群を重力方向に射影して物体の高さを計算する。

    Args:
        points_3d:   (N, 3) カメラ座標系の 3D 点群 [m]
        gravity_vec: (3,)   重力方向の単位ベクトル

    Returns:
        height_m: 高さ [m]
    """
    projections = points_3d @ gravity_vec  # (N,)
    return float(projections.max() - projections.min())


def estimate_height_from_depth_mask(
    depth_m: np.ndarray,
    mask: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    gravity_vec: np.ndarray,
):
    """
    深度画像 + マスク + 重力ベクトル → 物体高さ [m] + 点群

    Returns:
        height_m: 高さ [m]。有効点が不足している場合は 0.0。
        pts:      (N, 3) 3D 点群（可視化用）
    """
    pts = get_points_3d_from_mask(depth_m, mask, fx, fy, cx, cy)
    if len(pts) < 10:
        print(f"[高さ推定] 有効な深度点が不足 ({len(pts)} 点)。高さ推定をスキップ。")
        return 0.0, pts
    h = calc_height_from_points(pts, gravity_vec)
    print(f"[高さ推定] 点数={len(pts)}  高さ={h:.4f} m ({h*100:.1f} cm)")
    return h, pts


def draw_height_pcd(
    rgb_bgr: np.ndarray,
    pts: np.ndarray,
    gravity_vec: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    height_m: float,
) -> np.ndarray:
    """
    高さで色付けした点群を RGB 画像に ★ マーカーで描画する。

    低い点 → 青、高い点 → 赤
    """
    if len(pts) == 0:
        return rgb_bgr.copy()

    img = rgb_bgr.copy()
    H, W = img.shape[:2]

    # 重力方向に射影してスカラー高さを取得
    proj = pts @ gravity_vec  # (N,)
    p_min, p_max = proj.min(), proj.max()
    span = p_max - p_min if p_max > p_min else 1.0
    norm = (proj - p_min) / span  # 0=低, 1=高

    # 2D 投影
    Z = pts[:, 2]
    valid = Z > 0
    u = np.where(valid, (pts[:, 0] * fx / Z + cx).astype(np.int32), -1)
    v = np.where(valid, (pts[:, 1] * fy / Z + cy).astype(np.int32), -1)

    # 最高点（赤）・最低点（青）の2点のみ描画
    for target_idx, color in [(np.argmax(proj), (0, 0, 255)),   # 最高点: 赤
                               (np.argmin(proj), (255, 0, 0))]: # 最低点: 青
        if not valid[target_idx]:
            continue
        ui, vi = int(u[target_idx]), int(v[target_idx])
        if not (0 <= ui < W and 0 <= vi < H):
            continue
        cv2.drawMarker(img, (ui, vi), color,
                       markerType=cv2.MARKER_STAR,
                       markerSize=12, thickness=1, line_type=cv2.LINE_AA)

    # 高さテキスト
    cv2.putText(img, f"Height: {height_m*100:.1f} cm",
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "HIGH", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(img, "LOW",  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return img


# ===========================================================================
# モード別実行関数
# ===========================================================================

def run_offline_mesh(args, config):
    """RGB ファイル → サーバ → reference mesh 保存"""
    from pipeline.sam6d_detector import SAM6DClient

    sam_cfg = config["sam3d"]
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
    )

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[test] RGB ロード: {args.rgb}  shape={rgb.shape}")

    mesh_path = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")

    if args.click_x >= 0 and args.click_y >= 0:
        client.save_reference_mesh(rgb, mesh_path,
                                   click_x=args.click_x, click_y=args.click_y,
                                   mesh_method=mesh_method)
    elif args.interactive:
        client.save_reference_mesh_interactive(rgb, mesh_path)
    else:
        client.save_reference_mesh(rgb, mesh_path, mesh_method=mesh_method)

    print(f"\n[完了] mesh: {mesh_path}")
    print(f"       サーバ mesh: {client._server_mesh_path}")
    print(f"       テンプレート: {client._template_dir}")
    print(f"\n次のコマンド:")
    print(f"  python test_demo_w_IMU.py --mode online \\")
    print(f"    --rgb {args.rgb} \\")
    print(f"    --depth <depth_path> \\")
    print(f"    --mesh {mesh_path} \\")
    print(f"    --server-mesh-path \"{client._server_mesh_path}\" \\")
    print(f"    --template-dir \"{client._template_dir}\"")


def _align_y_up(mesh_pts: np.ndarray):
    """PEM は Y 軸下向きで出力するため、X 軸周り 180° 回転（Y→-Y, Z→-Z）で常に補正する。"""
    R_corr = np.diag([1.0, -1.0, -1.0])
    print("[Y-up補正] Y軸反転適用")
    return (R_corr @ mesh_pts.T).T.astype(mesh_pts.dtype), R_corr


def run_full(args, config):
    """
    RGB + 深度ファイル → SAM マスク取得 → IMU/重力で高さ自動推定
    → SAM-6D pose → 可視化
    """
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from utils.coord_transform import CameraIntrinsics, ObjectPose
    from utils.visualization import project_hands_on_image, save_grasp_figure
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]

    # ---- RGB ロード ----
    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")
    print(f"[test] RGB ロード: {args.rgb}  shape={rgb.shape}")

    # ---- 深度ロード ----
    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    # ---- Step 0: 重力ベクトル取得 ----
    if args.gravity is not None:
        # --gravity で手動指定（cam.json からの自動設定も含む）
        gravity_vec = np.array(args.gravity, dtype=np.float64)
        gravity_vec = gravity_vec / np.linalg.norm(gravity_vec)
        print(f"[高さ] 重力ベクトル: {gravity_vec}")
    else:
        # RealSense IMU から自動取得
        gravity_vec = get_gravity_imu(n_samples=args.imu_samples)
    object_size_mm = 0.0  # マスク取得後に計算

    # ---- Step 1: SAM-3D でメッシュ生成（マスクも取得） ----
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mesh_path = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")
    click_x, click_y = args.click_x, args.click_y

    print("\n[Step 1] SAM-3D でメッシュ生成中 (マスク取得)...")
    if args.click_x >= 0 and args.click_y >= 0:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            click_x=click_x, click_y=click_y,
            mesh_method=mesh_method,
            object_size_mm=0.0,  # スケールはここでは設定しない
        )
    elif args.interactive:
        _, click_x, click_y, masks, scores = client.save_reference_mesh_interactive(
            rgb, mesh_path, mesh_method=mesh_method,
        )
    else:
        _, masks, scores = client.save_reference_mesh(
            rgb, mesh_path,
            mesh_method=mesh_method,
            object_size_mm=0.0,
        )

    print(f"[Step 1完了] mesh: {mesh_path}")

    # ---- Step 2: マスク + 深度 → 高さ推定 ----
    if gravity_vec is not None and masks:
        # スコアが最大のマスクを使用
        best_idx = int(np.argmax(scores)) if scores else 0
        best_mask = masks[best_idx]
        print(f"\n[Step 2] マスクから高さ推定中 (mask_idx={best_idx}, score={scores[best_idx] if scores else '?':.3f})...")

        # マスクを深度と同サイズにリサイズ
        if best_mask.shape[:2] != depth.shape[:2]:
            best_mask = cv2.resize(best_mask, (depth.shape[1], depth.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)

        # マスク面積に応じて erosion を適用（大きいマスクのみ5px縮小）
        mask_area = int(np.count_nonzero(best_mask))
        img_area  = best_mask.shape[0] * best_mask.shape[1]
        mask_ratio = mask_area / img_area * 100
        print(f"[Step 2] マスク面積: {mask_area}px / {img_area}px ({mask_ratio:.1f}%)")
        if mask_area > img_area * 0.02:  # 画像の2%以上なら大きいとみなす
            erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            eroded_mask = cv2.erode(best_mask, erode_kernel, iterations=1)
            print(f"[Step 2] → erosion 適用 (3px)")
        else:
            eroded_mask = best_mask
            print(f"[Step 2] → 小さいため erosion スキップ")

        height_m, pts_3d = estimate_height_from_depth_mask(
            depth, eroded_mask,
            intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
            gravity_vec,
        )

        if height_m > 0.005:  # 5mm 以上なら使用
            object_size_mm = height_m * 1000.0
            print(f"[Step 2完了] 推定高さ: {height_m*100:.1f} cm = {object_size_mm:.0f} mm")

            os.makedirs("output/test", exist_ok=True)

            # SAM マスクをオーバーレイして保存
            mask_vis = cv2.applyColorMap(best_mask, cv2.COLORMAP_JET)
            mask_overlay = cv2.addWeighted(rgb, 0.6, mask_vis, 0.4, 0)
            cv2.imwrite("output/test/sam_mask.png", mask_overlay)
            print("[Step 2] SAM マスク保存: output/test/sam_mask.png")

            # 高さ色付き点群画像を保存
            height_vis = draw_height_pcd(
                rgb, pts_3d, gravity_vec,
                intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy,
                height_m,
            )
            cv2.imwrite("output/test/server_calc_height.png", height_vis)
            print("[Step 2] 高さ推定点群画像保存: output/test/server_calc_height.png")
        else:
            print("[Step 2] 高さ推定失敗。object_size_mm=0 (サーバ深度推定に委譲)。")
            object_size_mm = 0.0
    elif not masks:
        print("[Step 2] SAM マスクが空。高さ推定をスキップ。")

    # ---- Step 3: 6DoF pose 推定 ----
    # object_size_mm を使ってスケールを設定してから推定
    client._object_size_mm = object_size_mm

    print(f"\n[Step 3] SAM-6D で 6DoF pose 推定中 (object_size_mm={object_size_mm:.0f})...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics,
        click_x=click_x, click_y=click_y,
    )
    print(f"[Step 3完了] t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print(f"  R=\n{R}")

    # ---- 軸方向を表示 ----
    _ax_labels = ["X(赤)", "Y(緑)", "Z(青)"]
    print("\n[軸方向] 物体の各軸が画像のどちらを向いているか (X=赤/Y=緑/Z=青)")
    print(f"  {'軸':<8} {'cam_X':>7} {'cam_Y':>7} {'cam_Z':>7}  画像上下")
    for i, label in enumerate(_ax_labels):
        ax = R[:, i]
        if abs(ax[1]) > 0.5:
            updown = "画像↑ 上向き" if ax[1] < 0 else "画像↓ 下向き"
        else:
            updown = "(左右 or 奥行き方向)"
        print(f"  {label:<8} {ax[0]:>+7.3f} {ax[1]:>+7.3f} {ax[2]:>+7.3f}  {updown}")
    print(f"  ※cam_Y が負 → 画像の上方向")

    os.makedirs("output/test", exist_ok=True)
    if img_pose is not None:
        cv2.imwrite("output/test/server_pointcloud.png", img_pose)
        print("[Step 3] 点群投影画像保存: output/test/server_pointcloud.png")
        if not args.no_show:
            cv2.imshow("Pose: pointcloud", img_pose)
    if img_mesh is not None:
        cv2.imwrite("output/test/server_mesh.png", img_mesh)
        print("[Step 3] メッシュ投影画像保存: output/test/server_mesh.png")
        if not args.no_show:
            cv2.imshow("Pose: mesh", img_mesh)
    if not args.no_show:
        print("何かキーを押すと続行...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("\n" + "=" * 50)
    if object_size_mm > 0:
        print(f"  推定物体高さ: {object_size_mm/10:.1f} cm  ({object_size_mm:.0f} mm)")
    else:
        print(f"  推定物体高さ: 自動推定 (高さ入力なし)")
    print(f"  pose t: [{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] m")
    print("=" * 50)

    if args.skip_grasp:
        print("[完了] --skip-grasp が指定されたため把持姿勢生成をスキップします。")
        return

    # ---- Step 4: Shape2Gesture ----
    mesh_pts = load_pointcloud_ply(mesh_path, target_points=2048)

    mesh_pts, R_corr = _align_y_up(mesh_pts)
    R = (R.astype(np.float64) @ R_corr.T).astype(np.float32)

    _centered = mesh_pts - mesh_pts.mean(axis=0)
    _bbox_ext = _centered.max(axis=0) - _centered.min(axis=0)
    mesh_scale_m = float(np.max(_bbox_ext)) / 1000.0
    print(f"[CoordTransform] mesh_scale_m={mesh_scale_m:.4f} m")
    pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)

    print("\n[Step 4] Shape2Gesture で把持姿勢を生成中...")
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()
    grasp_results = generator.generate(mesh_pts, num_samples=model_cfg["num_samples"])
    norm_pts, seg_labels = generator.get_segmentation(mesh_pts)

    print(f"\n[Step 5] {len(grasp_results)} 件の把持姿勢を保存中...")
    for i, (lh_norm, rh_norm) in enumerate(grasp_results):
        sd = f"output/test/sample_{i:02d}"
        os.makedirs(sd, exist_ok=True)
        save_grasp_figure(
            norm_pts, lh_norm, rh_norm,
            labels=seg_labels,
            save_path=os.path.join(sd, "grasp_3d.png"),
            title=f"Sample {i}",
        )
        result_img = project_hands_on_image(
            rgb, lh_norm, rh_norm,
            object_pose=pose,
            intrinsics=intrinsics,
        )
        cv2.imwrite(os.path.join(sd, "rgb_w_grasp.png"), result_img)
        print(f"  保存: {sd}/")

    if not args.no_show:
        cv2.imshow("Grasp Result", cv2.imread("output/test/sample_00/rgb_w_grasp.png"))
        print("\n何かキーを押すと終了...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_online(args, config):
    """RGB + 深度ファイル → SAM-6D pose → Shape2Gesture → 画像投影"""
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from utils.coord_transform import CameraIntrinsics, ObjectPose
    from utils.visualization import project_hands_on_image, project_pointcloud_on_image, render_mesh_on_image, save_grasp_figure
    from utils.pointcloud_utils import load_pointcloud_ply

    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]

    rgb = cv2.imread(args.rgb)
    if rgb is None:
        raise FileNotFoundError(f"RGB 画像が読み込めません: {args.rgb}")

    depth = load_depth(args.depth, depth_scale=args.depth_scale)
    if depth.shape[:2] != rgb.shape[:2]:
        depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]),
                           interpolation=cv2.INTER_NEAREST)

    h, w = rgb.shape[:2]
    intrinsics = load_intrinsics(args, w, h)

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )
    client.load_reference_mesh(
        args.mesh,
        server_mesh_path=args.server_mesh_path or "",
        template_dir=args.template_dir or "",
    )

    print("\n[Step 1] SAM-6D で 6DoF pose 推定中...")
    R, t, img_pose, img_mesh = client.estimate_pose(
        rgb, depth, intrinsics,
        click_x=args.click_x, click_y=args.click_y,
    )
    print(f"  R=\n{R}")
    print(f"  t={t}")

    # ---- 軸方向を表示 ----
    _ax_labels = ["X(赤)", "Y(緑)", "Z(青)"]
    print("\n[軸方向] 物体の各軸が画像のどちらを向いているか (X=赤/Y=緑/Z=青)")
    print(f"  {'軸':<8} {'cam_X':>7} {'cam_Y':>7} {'cam_Z':>7}  画像上下")
    for i, label in enumerate(_ax_labels):
        ax = R[:, i]
        if abs(ax[1]) > 0.5:
            updown = "画像↑ 上向き" if ax[1] < 0 else "画像↓ 下向き"
        else:
            updown = "(左右 or 奥行き方向)"
        print(f"  {label:<8} {ax[0]:>+7.3f} {ax[1]:>+7.3f} {ax[2]:>+7.3f}  {updown}")
    print(f"  ※cam_Y が負 → 画像の上方向")

    mesh_pts = load_pointcloud_ply(args.mesh, target_points=2048)
    os.makedirs("output/test", exist_ok=True)

    if img_pose is not None:
        cv2.imwrite("output/test/server_pointcloud.png", img_pose)
    if img_mesh is not None:
        cv2.imwrite("output/test/server_mesh.png", img_mesh)

    vis_img = render_mesh_on_image(rgb, args.mesh, R, t, intrinsics, mesh_unit="mm")
    cv2.imwrite("output/test/pose_check_mesh.png", vis_img)

    vis_pts_img = project_pointcloud_on_image(rgb, mesh_pts, R, t, intrinsics, points_unit="mm")
    cv2.imwrite("output/test/pose_check_pts.png", vis_pts_img)

    if not args.no_show:
        cv2.imshow("Pose Check: mesh render", vis_img)
        cv2.imshow("Pose Check: pointcloud + bbox", vis_pts_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.skip_grasp:
        print("\n[完了] --skip-grasp が指定されたため把持姿勢生成をスキップします。")
        return

    gravity_vec = None
    if args.gravity is not None:
        gravity_vec = np.array(args.gravity, dtype=np.float64)
        gravity_vec /= np.linalg.norm(gravity_vec)

    mesh_pts, R_corr = _align_y_up(mesh_pts)
    R = (R.astype(np.float64) @ R_corr.T).astype(np.float32)

    _centered = mesh_pts - mesh_pts.mean(axis=0)
    _bbox_ext = _centered.max(axis=0) - _centered.min(axis=0)
    mesh_scale_m = float(np.max(_bbox_ext)) / 1000.0
    print(f"[CoordTransform] mesh_scale_m={mesh_scale_m:.4f} m")
    pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)

    print("\n[Step 2] Shape2Gesture で把持姿勢を生成中...")
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()
    grasp_results = generator.generate(mesh_pts, num_samples=model_cfg["num_samples"])
    norm_pts, seg_labels = generator.get_segmentation(mesh_pts)

    print(f"\n[Step 3] {len(grasp_results)} 件の把持姿勢を保存中...")
    for i, (lh_norm, rh_norm) in enumerate(grasp_results):
        sd = f"output/test/sample_{i:02d}"
        os.makedirs(sd, exist_ok=True)
        save_grasp_figure(
            norm_pts, lh_norm, rh_norm,
            labels=seg_labels,
            save_path=os.path.join(sd, "grasp_3d.png"),
            title=f"Sample {i}",
        )
        result_img = project_hands_on_image(
            rgb, lh_norm, rh_norm,
            object_pose=pose,
            intrinsics=intrinsics,
        )
        cv2.imwrite(os.path.join(sd, "rgb_w_grasp.png"), result_img)
        print(f"  保存: {sd}/")

    if not args.no_show:
        cv2.imshow("Grasp Result", cv2.imread("output/test/sample_00/rgb_w_grasp.png"))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ===========================================================================
# エントリポイント
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="IMU 自動高さ推定付きパイプラインテスト")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["offline-mesh", "online", "full"], default="full")
    parser.add_argument("--no-show", action="store_true",
                        help="cv2.imshow を使わない (ヘッドレス環境用)")
    parser.add_argument("--skip-grasp", action="store_true",
                        help="Shape2Gesture をスキップして pose 可視化のみ実行")

    # フォルダ指定（rgb.png / depth.png / cam.json を自動解決）
    parser.add_argument("--data-dir", default=None,
                        help="save_data_IMU.py で保存したフォルダ。指定すると --rgb/--depth/--cam-json を自動設定。")

    # 個別ファイル指定（--data-dir より低優先）
    parser.add_argument("--rgb",   default=None,  help="RGB 画像パス (.png/.jpg)")
    parser.add_argument("--depth", default=None,  help="深度画像パス (.png)")
    parser.add_argument("--depth-scale", type=float, default=0.001,
                        help="深度のスケール係数 (0.001: mm→m, デフォルト)")
    parser.add_argument("--cam-json", default=None,
                        help="camera JSON ファイル ({cam_K:[...], depth_scale, gravity})")
    parser.add_argument("--fx", type=float, default=591.0)
    parser.add_argument("--fy", type=float, default=590.0)
    parser.add_argument("--cx", type=float, default=-1)
    parser.add_argument("--cy", type=float, default=-1)

    # mesh パス
    parser.add_argument("--mesh",     default=None, help="[online] ローカル mesh (.ply)")
    parser.add_argument("--mesh-out", default="meshes/test_object.ply",
                        help="[offline-mesh/full] 保存先")
    parser.add_argument("--server-mesh-path", default=None)
    parser.add_argument("--template-dir",     default=None)

    # クリック・インタラクティブ
    parser.add_argument("--click-x",    type=int, default=-1)
    parser.add_argument("--click-y",    type=int, default=-1)
    parser.add_argument("--interactive", action="store_true", default=True)

    # 高さ指定（優先順位: --gravity > cam.json の gravity > IMU 自動）
    parser.add_argument("--gravity", type=float, nargs=3, default=None,
                        metavar=("GX", "GY", "GZ"),
                        help="重力方向ベクトル手動指定。cam.json の gravity より優先。")
    parser.add_argument("--imu-samples", type=int, default=30,
                        help="IMU サンプル数 (デフォルト: 30)")

    args = parser.parse_args()

    # --data-dir が指定されていれば rgb/depth/cam-json を自動設定
    if args.data_dir:
        if args.rgb is None:
            args.rgb = os.path.join(args.data_dir, "rgb.png")
        if args.depth is None:
            args.depth = os.path.join(args.data_dir, "depth.png")
        if args.cam_json is None:
            args.cam_json = os.path.join(args.data_dir, "cam.json")
        print(f"[data-dir] {args.data_dir} → rgb/depth/cam.json を自動設定")

    config = load_config(args.config)

    if args.mode == "offline-mesh":
        if not args.rgb:
            print("エラー: --rgb または --data-dir を指定してください。")
            sys.exit(1)
        run_offline_mesh(args, config)
    elif args.mode == "full":
        if not args.rgb or not args.depth:
            print("エラー: --data-dir または --rgb/--depth を指定してください。")
            sys.exit(1)
        run_full(args, config)
    else:
        if not args.rgb or not args.depth:
            print("エラー: --data-dir または --rgb/--depth を指定してください。")
            sys.exit(1)
        if not args.mesh:
            print("エラー: --mesh を指定してください。")
            sys.exit(1)
        run_online(args, config)


if __name__ == "__main__":
    main()
