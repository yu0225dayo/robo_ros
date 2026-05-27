"""
real_world_demo メインエントリーポイント

パイプライン:
    [オフライン] 物体ごとに1回:
        RealSense RGB → サーバ (SAM-3D) → reference mesh (.ply) 保存
        サーバが SAM-6D でテンプレートをレンダリングし保存

    [オンライン] 毎回:
        RealSense RGBD
            → サーバ (SAM-6D) → 6DoF pose (R, t)
            → reference mesh × (R, t, scale) → カメラ座標系の完全点群
            → Shape2Gesture → 把持姿勢 (正規化座標系)
            → (R, t, scale) でカメラ座標系へ変換
            → 画像に投影して保存 / ロボットへ送信

使用方法:
    # Step1: reference mesh を生成 (物体ごとに1回)
    python main.py --mode offline-mesh --mesh-out meshes/cup.ply

    # Step2: 把持姿勢生成 (毎回)
    python main.py --mesh meshes/cup.ply --no-robot
    python main.py --mesh meshes/cup.ply
"""

import argparse
import json
import os
import sys
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def put_text_jp(img: np.ndarray, text: str, pos, font_size: int = 24, color=(0, 255, 0)) -> np.ndarray:
    """PIL を使って日本語テキストをBGR画像に描画する"""
    from PIL import Image, ImageDraw, ImageFont
    import cv2 as _cv2
    img_pil = Image.fromarray(_cv2.cvtColor(img, _cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = None
    for fp in [
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/msgothic.ttc",
        "C:/Windows/Fonts/YuGothM.ttc",
    ]:
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except (IOError, OSError):
            continue
    if font is None:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=(color[2], color[1], color[0]))  # BGR→RGB
    return _cv2.cvtColor(np.array(img_pil), _cv2.COLOR_RGB2BGR)


# ==============================================================
# オフラインフェーズ: reference mesh の生成・保存
# ==============================================================

def run_offline_mesh(config: dict, args):
    """
    RealSense RGB → サーバ (SAM-3D + SAM-6D テンプレート) → mesh 保存

    物体ごとに1回だけ実行する。
    生成した mesh は --mesh-out で指定したパスに保存される。
    """
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient

    cam_cfg = config["camera"]
    sam_cfg = config["sam3d"]

    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
    )

    print("\n[操作方法]")
    print("  [c] この画像で reference mesh を生成")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            key = camera.show_preview(rgb, depth)

            if key == ord("q"):
                break
            elif key == ord("c"):
                mesh_path = args.mesh_out
                print(f"\n[Step] reference mesh 生成中 → {mesh_path}")
                if sam_cfg.get("interactive", True):
                    client.save_reference_mesh_interactive(rgb, mesh_path)  # (path, cx, cy, mask)
                else:
                    client.save_reference_mesh(rgb, mesh_path)  # (path, mask)
                print(f"[完了] {mesh_path} に保存しました。")
                print("       次回: python main.py --mesh", mesh_path)
                break
    finally:
        camera.stop()


# ==============================================================
# オンラインフェーズ: RGBD → 6DoF pose → 把持姿勢 → ロボット
# ==============================================================

def run_online(config: dict, args):
    """
    毎フレーム: RGBD → SAM-6D pose → Shape2Gesture → ロボット
    """
    import cv2 as _cv2
    from datetime import datetime
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from pipeline.robot_interface import RobotInterface, GraspPose
    from utils.visualization import (
        live_visualize_setup, live_visualize_update,
        visualize_multiple_grasps, project_hands_on_image, show_figure,
    )
    from utils.coord_transform import (
        CameraIntrinsics, ObjectPose, normalized_to_camera,
    )
    from utils.pointcloud_utils import load_pointcloud_ply

    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]
    robot_cfg = config["robot"]
    vis_cfg   = config["visualization"]

    # ---- 初期化 ----
    print("=" * 50)
    print("[Step 1] 初期化中...")
    print("=" * 50)

    # Shape2Gesture モデルのロード
    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()

    # SAM-6D クライアント
    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 120.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )
    # reference mesh のパスをセット (サーバ側パスは別途指定 or 初回 offline-mesh で取得)
    server_mesh_path = args.server_mesh_path or ""
    template_dir     = args.template_dir or ""
    client.load_reference_mesh(args.mesh, server_mesh_path, template_dir)

    # reference mesh の点群を読み込み + スケール計算
    mesh_pts = load_pointcloud_ply(args.mesh, target_points=2048)
    # PLY は mm 単位。正規化半径 (mm) / 1000 → メートルスケール
    _centered = mesh_pts - mesh_pts.mean(axis=0)
    _bbox_ext = _centered.max(axis=0) - _centered.min(axis=0)
    mesh_scale_m = float(np.max(_bbox_ext)) / 1000.0
    print(f"[CoordTransform] mesh_scale_m={mesh_scale_m:.4f} m")

    # ロボット
    mode = robot_cfg["mode"]
    robot_kwargs = robot_cfg.get(mode, {})
    robot = RobotInterface(mode=mode, **robot_kwargs)
    if not args.no_robot:
        robot.connect()

    # カメラ
    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    # カメラ内部パラメータを camera.json として保存
    cam_json = {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "width": cam_cfg["width"],
        "height": cam_cfg["height"],
    }
    os.makedirs("output", exist_ok=True)
    with open("camera.json", "w") as f:
        json.dump(cam_json, f, indent=2)
    print("[Camera] 内部パラメータ保存: camera.json")

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    fig, ax = live_visualize_setup()

    print("\n[操作方法]")
    print("  [g] 把持姿勢を生成してロボットに送信")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            key = camera.show_preview(rgb, depth)

            if key == ord("q"):
                break

            elif key == ord("g"):
                out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)

                # ---- Step 2: SAM-6D で 6DoF pose 推定 ----
                print("\n" + "=" * 50)
                print("[Step 2] SAM-6D で 6DoF pose 推定中...")
                print("=" * 50)

                R, t, img_pose, img_mesh = client.estimate_pose(
                    rgb, depth, intrinsics,
                    click_x=args.click_x, click_y=args.click_y,
                )

                # サーバから受信した投影画像を保存・表示
                if img_mesh is not None:
                    path = os.path.join(out_dir, "pose.png")
                    _cv2.imwrite(path, img_mesh)
                    _cv2.imshow("pose", img_mesh)
                    _cv2.waitKey(1)
                    print(f"[投影] メッシュ投影: {path}")

                if img_pose is not None:
                    path = os.path.join(out_dir, "pose_pointcloud.png")
                    _cv2.imwrite(path, img_pose)
                    _cv2.imshow("pose_pointcloud", img_pose)
                    _cv2.waitKey(1)
                    print(f"[投影] 点群投影: {path}")

                # 6DoF pose オブジェクト (R, t, scale)
                pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)

                # ---- Step 3: Shape2Gesture で把持姿勢生成 ----
                print("\n" + "=" * 50)
                print("[Step 3] Shape2Gesture で把持姿勢を生成中...")
                print("=" * 50)

                grasp_results = generator.generate(
                    mesh_pts,
                    num_samples=model_cfg["num_samples"],
                )

                norm_pts, seg_labels = generator.get_segmentation(mesh_pts)

                if vis_cfg["show_all_samples"] and len(grasp_results) > 1:
                    visualize_multiple_grasps(
                        norm_pts, grasp_results,
                        labels=seg_labels if vis_cfg["show_segmentation"] else None,
                    )

                left_hand_norm, right_hand_norm = grasp_results[0]
                live_visualize_update(
                    fig, ax, norm_pts, left_hand_norm, right_hand_norm,
                    labels=seg_labels if vis_cfg["show_segmentation"] else None,
                )

                # ---- Step 4: 点群画像の上に把持姿勢を投影して保存・表示 ----
                base = img_pose if img_pose is not None else rgb
                grasp_img = project_hands_on_image(
                    base, left_hand_norm, right_hand_norm,
                    object_pose=pose,
                    intrinsics=intrinsics,
                )
                grasp_path = os.path.join(out_dir, "pointcloud_w_grasp.png")
                _cv2.imwrite(grasp_path, grasp_img)
                _cv2.imshow("pointcloud_w_grasp", grasp_img)
                _cv2.waitKey(1)
                print(f"[投影結果] {out_dir} に保存しました。")

                # ---- Step 5: ロボットへ送信 ----
                print("\n" + "=" * 50)
                print("[Step 5] ロボットに把持姿勢を送信中...")
                print("=" * 50)

                left_hand_cam  = normalized_to_camera(left_hand_norm, pose)
                right_hand_cam = normalized_to_camera(right_hand_norm, pose)

                if not args.no_robot:
                    grasp_pose = GraspPose(
                        left_hand=left_hand_cam,
                        right_hand=right_hand_cam,
                        object_scale=mesh_scale_m,
                        object_center=t,
                    )
                    robot.send_grasp_pose(grasp_pose, execute=robot_cfg["execute"])
                    result = robot.wait_for_result(timeout=robot_cfg["timeout"])
                    print(f"[Robot] 結果: {result}")
                else:
                    print("[main] --no-robot: ロボット送信をスキップ")

    except KeyboardInterrupt:
        print("\n[main] 中断されました。")
    finally:
        camera.stop()
        if not args.no_robot:
            robot.disconnect()


# ==============================================================
# フルパイプライン: mesh生成 → pose推定 → 把持姿勢生成
# ==============================================================

def run_full(config: dict, args):
    """
    フルパイプライン (カメラ起動から把持まで一連で実行)

    操作:
        [c] 物体をクリック選択 → mesh生成 → 3D点群表示
        [g] pose推定 → 把持生成 → 画像保存 → ロボット送信 (自動, [c]後に有効)
        [q] 終了
    """
    import cv2 as _cv2
    from datetime import datetime
    from pipeline.camera import RealSenseCamera
    from pipeline.sam6d_detector import SAM6DClient
    from pipeline.grasp_generator import GraspGenerator
    from pipeline.robot_interface import RobotInterface, GraspPose
    from utils.visualization import (
        MatplotlibGraspVisualizer, save_grasp_figure,
        live_visualize_setup, show_figure,
        visualize_multiple_grasps, project_hands_on_image,
    )
    from utils.coord_transform import (
        CameraIntrinsics, ObjectPose, normalized_to_camera,
    )
    from utils.pointcloud_utils import load_pointcloud_ply, normalize_pointcloud

    cam_cfg   = config["camera"]
    sam_cfg   = config["sam3d"]
    sam6d_cfg = config.get("sam6d", {})
    model_cfg = config["grasp_model"]
    robot_cfg = config["robot"]
    vis_cfg   = config["visualization"]

    mesh_path  = args.mesh_out
    mesh_method = sam_cfg.get("mesh_method", "bpa")

    # ---- 初期化 ----
    print("=" * 50)
    print("[初期化] Shape2Gesture モデルをロード中...")
    print("=" * 50)

    generator = GraspGenerator(
        model_dir=model_cfg["model_dir"],
        epoch=model_cfg["epoch"],
    )
    generator.load_models()

    client = SAM6DClient(
        server_url=sam_cfg["server_url"],
        timeout_mesh=sam_cfg.get("timeout", 300.0),
        timeout_pose=sam6d_cfg.get("timeout", 30.0),
    )

    mode = robot_cfg["mode"]
    robot_kwargs = robot_cfg.get(mode, {})
    robot = RobotInterface(mode=mode, **robot_kwargs)
    if not args.no_robot:
        robot.connect()

    camera = RealSenseCamera(
        width=cam_cfg["width"],
        height=cam_cfg["height"],
        fps=cam_cfg["fps"],
    )
    camera.start()

    # カメラ内部パラメータを camera.json として保存
    cam_json = {
        "fx": float(camera.fx),
        "fy": float(camera.fy),
        "cx": float(camera.cx),
        "cy": float(camera.cy),
        "width": cam_cfg["width"],
        "height": cam_cfg["height"],
    }
    os.makedirs("output", exist_ok=True)
    with open("camera.json", "w") as f:
        json.dump(cam_json, f, indent=2)
    print("[Camera] 内部パラメータ保存: camera.json")

    intrinsics = CameraIntrinsics(
        fx=camera.fx, fy=camera.fy,
        cx=camera.cx, cy=camera.cy,
        width=cam_cfg["width"], height=cam_cfg["height"],
    )

    fig, ax = live_visualize_setup()
    o3d_vis = MatplotlibGraspVisualizer()

    # 状態変数
    mesh_pts      = None
    mesh_pts_norm = None
    click_x = click_y = -1
    rgb_frozen   = None
    depth_frozen = None

    print("\n[操作方法]")
    print("  [c] 物体をクリック選択 → mesh生成 → 3D点群表示")
    print("  [g] pose推定 → 把持生成 → 画像保存 (自動, [c]後に有効)")
    print("  [q] 終了")
    print("=" * 50)

    try:
        while True:
            rgb, depth, _ = camera.capture()
            o3d_vis.poll()  # open3d ウィンドウのイベント処理

            status = "[g]で把持生成 / [c]で物体選択" if mesh_pts is not None else "[g]で把持生成（自動でmesh生成）/ [c]で物体選択のみ"
            preview = put_text_jp(rgb.copy(), status, (10, 10))
            key = camera.show_preview(preview, depth)

            if key == ord("q"):
                break

            elif key == ord("c"):
                # ---- フレーム固定 + mesh 生成 ----
                print("\n" + "=" * 50)
                print("[mesh生成] 物体をクリックして選択してください...")
                print("=" * 50)
                rgb_frozen   = rgb.copy()
                depth_frozen = depth.copy()
                os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)
                _, click_x, click_y, sam_masks, sam_scores = client.save_reference_mesh_interactive(
                    rgb_frozen, mesh_path, mesh_method=mesh_method
                )
                mesh_pts      = load_pointcloud_ply(mesh_path, target_points=2048)
                mesh_pts_norm = normalize_pointcloud(mesh_pts)
                _centered = mesh_pts - mesh_pts.mean(axis=0)
                _bbox_ext = _centered.max(axis=0) - _centered.min(axis=0)
                mesh_scale_m = float(np.max(_bbox_ext)) / 1000.0
                print(f"[mesh生成完了] {mesh_path}  scale={mesh_scale_m:.4f} m")

                # ---- 3マスクを横並びで out_dir に保存・表示 ----
                out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)
                if sam_masks:
                    panels = []
                    for i, (mask, score) in enumerate(zip(sam_masks, sam_scores)):
                        panel = rgb_frozen.copy()
                        colored = np.zeros_like(panel)
                        colored[mask > 127] = (0, 255, 0)
                        panel = _cv2.addWeighted(panel, 0.7, colored, 0.3, 0)
                        # bbox
                        ys, xs = np.where(mask > 127)
                        if len(xs) > 0:
                            x1, y1 = int(xs.min()), int(ys.min())
                            x2, y2 = int(xs.max()), int(ys.max())
                            _cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        label = f"mask{i} score={score:.3f}"
                        if i == int(np.argmax(sam_scores)):
                            label += " [BEST]"
                            _cv2.rectangle(panel, (0, 0), (panel.shape[1]-1, panel.shape[0]-1),
                                           (0, 0, 255), 4)
                        _cv2.putText(panel, label, (10, 30),
                                     _cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        _cv2.imwrite(os.path.join(out_dir, f"mask{i}.png"), panel)
                        panels.append(panel)
                    # 横並び1枚に結合して表示
                    compare = np.hstack(panels)
                    _cv2.imwrite(os.path.join(out_dir, "masks_compare.png"), compare)
                    _cv2.imshow("SAM masks (0=small / 1=mid / 2=large)", compare)
                    _cv2.waitKey(1)
                    print(f"[マスク] 保存: {out_dir}/mask0.png ~ mask2.png, masks_compare.png")

                # 3D 点群のみ表示 (matplotlib, ダウンサンプリング)
                _vis_n = min(512, len(mesh_pts_norm))
                _vis_idx = np.random.choice(len(mesh_pts_norm), _vis_n, replace=False)
                ax.cla()
                ax.set_title("Reference Mesh")
                ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
                ax.set_axis_off()
                ax.scatter(mesh_pts_norm[_vis_idx, 0], mesh_pts_norm[_vis_idx, 1],
                           mesh_pts_norm[_vis_idx, 2], c="green", s=3)
                show_figure(fig)
                _cv2.waitKey(1)

                # ---- pose 推定 → RGB に 3D 点群を投影 ----
                print("[Step] pose 推定 → RGB に点群投影中...")
                try:
                    R, t, img_pose, img_mesh = client.estimate_pose(
                        rgb_frozen, depth_frozen, intrinsics,
                        click_x=click_x, click_y=click_y,
                    )
                    from utils.visualization import project_pointcloud_on_image
                    proj_img = project_pointcloud_on_image(
                        rgb_frozen, mesh_pts, R, t, intrinsics,
                        points_unit="mm",
                    )
                    _cv2.imshow("mesh_projection", proj_img)
                    _cv2.waitKey(1)
                    print("[mesh投影] RGB 画像に点群を投影しました。")
                except Exception as e:
                    print(f"[警告] pose 推定失敗のため投影をスキップ: {e}")

                print("[次のステップ] [g] を押してください。")

            elif key == ord("g"):
                # mesh 未生成の場合は自動でメッシュ生成を先に実行
                if mesh_pts is None or rgb_frozen is None:
                    print("\n" + "=" * 50)
                    print("[自動] mesh 未生成: 物体をクリックして選択してください...")
                    print("=" * 50)
                    rgb_frozen   = rgb.copy()
                    depth_frozen = depth.copy()
                    os.makedirs(os.path.dirname(os.path.abspath(mesh_path)), exist_ok=True)
                    try:
                        _, click_x, click_y, sam_masks, sam_scores = client.save_reference_mesh_interactive(
                            rgb_frozen, mesh_path, mesh_method=mesh_method
                        )
                    except KeyboardInterrupt:
                        print("[キャンセル] 物体選択をキャンセルしました。")
                        rgb_frozen = depth_frozen = None
                        continue
                    mesh_pts      = load_pointcloud_ply(mesh_path, target_points=2048)
                    mesh_pts_norm = normalize_pointcloud(mesh_pts)
                    _centered = mesh_pts - mesh_pts.mean(axis=0)
                    _bbox_ext = _centered.max(axis=0) - _centered.min(axis=0)
                    mesh_scale_m = float(np.max(_bbox_ext)) / 2.0 / 1000.0
                    print(f"[mesh生成完了] {mesh_path}  scale={mesh_scale_m:.4f} m")

                    # 3D 点群を表示 (matplotlib, ダウンサンプリング)
                    _vis_n = min(512, len(mesh_pts_norm))
                    _vis_idx = np.random.choice(len(mesh_pts_norm), _vis_n, replace=False)
                    ax.cla()
                    ax.set_title("Reference Mesh")
                    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_zlim(-1.2, 1.2)
                    ax.set_axis_off()
                    ax.scatter(mesh_pts_norm[_vis_idx, 0], mesh_pts_norm[_vis_idx, 1],
                               mesh_pts_norm[_vis_idx, 2], c="green", s=3)
                    show_figure(fig)
                    _cv2.waitKey(1)

                out_dir = os.path.join("output", datetime.now().strftime("%Y%m%d_%H%M%S"))
                os.makedirs(out_dir, exist_ok=True)

                # ---- pose 推定 ----
                print("\n" + "=" * 50)
                print("[Step 1] SAM-6D で 6DoF pose 推定中...")
                print("=" * 50)
                try:
                    R, t, img_pose, img_mesh = client.estimate_pose(
                        rgb_frozen, depth_frozen, intrinsics,
                        click_x=click_x, click_y=click_y,
                    )
                except Exception as e:
                    print(f"[エラー] pose 推定失敗: {e}")
                    print("  → サーバログを確認してください。[g] で再試行できます。")
                    rgb_frozen = depth_frozen = None
                    mesh_pts = mesh_pts_norm = None
                    continue
                if img_mesh is not None:
                    _cv2.imwrite(os.path.join(out_dir, "pose.png"), img_mesh)
                    _cv2.imshow("pose", img_mesh); _cv2.waitKey(1)
                if img_pose is not None:
                    _cv2.imwrite(os.path.join(out_dir, "pose_pointcloud.png"), img_pose)
                    _cv2.imshow("pose_pointcloud", img_pose); _cv2.waitKey(1)

                pose = ObjectPose(center_3d=t, scale=mesh_scale_m, R=R)

                # ---- 把持姿勢生成 ----
                print("\n" + "=" * 50)
                print("[Step 2] Shape2Gesture で把持姿勢を生成中...")
                print("=" * 50)
                grasp_results = generator.generate(
                    mesh_pts, num_samples=model_cfg["num_samples"]
                )
                norm_pts, seg_labels = generator.get_segmentation(mesh_pts)

                # ダウンサンプリング (可視化・保存共通)
                _vis_n = min(512, len(norm_pts))
                _vis_idx = np.random.choice(len(norm_pts), _vis_n, replace=False)
                _vis_pts = norm_pts[_vis_idx]
                _vis_labels = (seg_labels[_vis_idx]
                               if seg_labels is not None and vis_cfg["show_segmentation"]
                               else None)

                # ---- 全サンプルをサブフォルダに保存 ----
                for i, (lh, rh) in enumerate(grasp_results):
                    sd = os.path.join(out_dir, f"sample_{i}")
                    os.makedirs(sd, exist_ok=True)
                    save_grasp_figure(
                        _vis_pts, lh, rh,
                        labels=_vis_labels,
                        save_path=os.path.join(sd, "grasp_3d.png"),
                        title=f"Sample {i}",
                    )
                    grasp_img_i = project_hands_on_image(
                        rgb_frozen, lh, rh,
                        object_pose=pose, intrinsics=intrinsics,
                    )
                    _cv2.imwrite(os.path.join(sd, "rgb_w_grasp.png"), grasp_img_i)
                print(f"[完了] {len(grasp_results)} サンプル保存: {out_dir}")

                # ---- sample_0 のみ表示 ----
                lh0, rh0 = grasp_results[0]
                o3d_vis.update(_vis_pts, lh0, rh0, labels=_vis_labels)
                grasp_img0 = project_hands_on_image(
                    rgb_frozen, lh0, rh0,
                    object_pose=pose, intrinsics=intrinsics,
                )
                _cv2.imwrite(os.path.join(out_dir, "rgb_w_grasp.png"), grasp_img0)
                _cv2.imshow("rgb_w_grasp", grasp_img0)
                _cv2.waitKey(1)

                # ---- ロボットへ送信 ----
                left_hand_cam  = normalized_to_camera(lh0, pose)
                right_hand_cam = normalized_to_camera(rh0, pose)
                if not args.no_robot:
                    grasp_pose = GraspPose(
                        left_hand=left_hand_cam,
                        right_hand=right_hand_cam,
                        object_scale=mesh_scale_m,
                        object_center=t,
                    )
                    robot.send_grasp_pose(grasp_pose, execute=robot_cfg["execute"])
                    result = robot.wait_for_result(timeout=robot_cfg["timeout"])
                    print(f"[Robot] 結果: {result}")
                else:
                    print("[main] --no-robot: ロボット送信をスキップ")

                # リセット → 次の物体選択へ
                rgb_frozen = depth_frozen = None
                mesh_pts = mesh_pts_norm = None
                print("\n[完了] 次の把持を行うには [g] を押してください（自動でmesh生成します）。")

    except KeyboardInterrupt:
        print("\n[main] 中断されました。")
    finally:
        camera.stop()
        o3d_vis.destroy()
        if not args.no_robot:
            robot.disconnect()


# ==============================================================
# エントリーポイント
# ==============================================================

def main():
    parser = argparse.ArgumentParser(
        description="real_world_demo: RealSense + SAM-3D + SAM-6D + Shape2Gesture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # Step1: reference mesh を生成 (物体ごとに1回)
  python main.py --mode offline-mesh --mesh-out meshes/cup.ply

  # Step2: 把持姿勢生成 (毎回)
  python main.py --mesh meshes/cup.ply --no-robot
  python main.py --mesh meshes/cup.ply
        """
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--mode", choices=["full", "online", "offline-mesh"], default="full",
        help="full: 物体選択からすべて一括実行 (デフォルト) / "
             "online: mesh 指定済みで pose 推定のみ / "
             "offline-mesh: mesh 生成のみ",
    )
    parser.add_argument(
        "--mesh", default=None,
        help="[online] ローカルの reference mesh (.ply) パス",
    )
    parser.add_argument(
        "--mesh-out", default="meshes/object.ply",
        help="[full/offline-mesh] mesh 保存先 .ply パス",
    )
    parser.add_argument(
        "--server-mesh-path", default=None,
        help="[online] サーバ側の mesh パス",
    )
    parser.add_argument(
        "--template-dir", default=None,
        help="[online] サーバ側のテンプレートディレクトリ",
    )
    parser.add_argument("--no-robot", action="store_true")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--epoch",       type=int, default=None)
    parser.add_argument(
        "--click-x", type=int, default=-1,
        help="[online] 物体クリック座標 X (-1: 画像中央)",
    )
    parser.add_argument(
        "--click-y", type=int, default=-1,
        help="[online] 物体クリック座標 Y (-1: 画像中央)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"エラー: 設定ファイルが見つかりません: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    if args.num_samples is not None:
        config["grasp_model"]["num_samples"] = args.num_samples
    if args.epoch is not None:
        config["grasp_model"]["epoch"] = args.epoch

    print("=" * 50)
    print("  real_world_demo")
    print("  RealSense + SAM-3D + SAM-6D + Shape2Gesture")
    print(f"  モード: {args.mode}")
    print("=" * 50)

    if args.mode == "full":
        run_full(config, args)

    elif args.mode == "offline-mesh":
        run_offline_mesh(config, args)

    else:  # online
        if args.mesh is None:
            print("エラー: --mesh で reference mesh (.ply) を指定してください。")
            sys.exit(1)
        if not os.path.exists(args.mesh):
            print(f"エラー: mesh が見つかりません: {args.mesh}")
            sys.exit(1)
        run_online(config, args)


if __name__ == "__main__":
    main()
