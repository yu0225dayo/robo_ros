# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

**sciurus17**（双腕人型ロボット）を動かすプロジェクト。
RealSense カメラで撮影した RGBD から物体の姿勢推定と両手把持姿勢を生成し、アームを制御する。

## システム構成

```
sciurus17 (ROS2 PC)                GPU 計算機 (10.40.1.126)
  RealSense RGBD カメラ
    └─ client/grasp_pipeline.py
         │  RGBD ──HTTP:8080──────→  server/server.py       (SAM2 + SAM3D + SAM6D)
         │                                ↓ HTTP:8081
         │                           SAM-6D Docker (sam6d_service)
         │  ←── R, t, points ──────────────────────────────
         │
         │  points ─HTTP:8082─────→  server/server_grasp.py (Shape2Gesture)
         │  ←── grasps (23関節×2手) ──────────────────────
         │
         └─ normalized_to_camera(R, t, scale) → 手首座標 [m]
```

## 起動手順（GPU 計算機上）

### ステップ 1: SAM-6D Docker

```bash
cd ~/ws/project/server
docker compose up -d sam6d
curl http://localhost:8081/health   # "models_loaded" が出るまで待つ
```

### ステップ 2: server.py（SAM2 + SAM3D + SAM6D）

```bash
conda activate sam3d-objects
cd /home/okada/ws/robo_ros/server
python server.py \
  --sam-checkpoint /ws/okada/project/sam2_checkpoints/sam2.1_hiera_large.pt \
  --sam3d-config   /ws/okada/SAM3D_6DoF/server/sam-3d-objects/checkpoints/hf/pipeline.yaml \
  --sam3d-repo     /ws/okada/SAM3D_6DoF/server/sam-3d-objects \
  --sam6d-service  http://localhost:8081 \
  --host 0.0.0.0 --port 8080
```

### ステップ 3: server_grasp.py（Shape2Gesture）

```bash
conda activate sam3d-objects
cd /home/okada/ws/robo_ros/server
python server_grasp.py \
  --grasp-model-dir ../client/save_model \
  --grasp-client-dir ../client \
  --sam-checkpoint /ws/okada/project/sam2_checkpoints/sam2.1_hiera_large.pt \
  --sam3d-config /ws/okada/SAM3D_6DoF/server/sam-3d-objects/checkpoints/hf/pipeline.yaml \
  --sam3d-repo /ws/okada/SAM3D_6DoF/server/sam-3d-objects \
  --sam6d-service http://localhost:8081 \
  --host 0.0.0.0 --port 8082
```

## ディレクトリ構成

| ディレクトリ | 内容 |
|---|---|
| `server/server.py` | 姿勢推定サーバ (SAM2 + SAM3D + SAM6D, port 8080) |
| `server/server_grasp.py` | 把持姿勢生成サーバ (Shape2Gesture, port 8082) |
| `client/grasp_pipeline.py` | クライアントパイプライン（4ステップ） |
| `client/pipeline/grasp_generator.py` | GraspGenerator クラス (Shape2Gesture ラッパー) |
| `client/utils/coord_transform.py` | 座標変換・ObjectPose |
| `client/models/` | Shape2Gesture モデル定義 |
| `client/save_model/` | 学習済みモデル重み |

## API エンドポイント

### server.py (port 8080)

| エンドポイント | 内容 |
|---|---|
| `GET /health` | 死活確認 |
| `POST /reconstruct_mesh` | RGB → SAM2マスク → SAM3D → PLY 返却 |
| `POST /pose_estimate` | RGB + depth + mesh → SAM2 + SAM6D → R, t, points 返却 |
| `POST /full_pipeline` | reconstruct_mesh + pose_estimate を一括実行 |

### server_grasp.py (port 8082)

| エンドポイント | 内容 |
|---|---|
| `GET /health` | 死活確認 |
| `POST /generate_grasp` | PLY → Shape2Gesture → 把持姿勢 (23関節×左右) 返却 |

## 座標変換の詳細

### 正規化座標 → カメラ座標

```python
p_cam = R @ (p_norm * mesh_scale_m) + t
```

- `p_norm`: Shape2Gesture 出力 (正規化物体座標系、[-1,1] 程度)
- `mesh_scale_m`: メッシュの Z軸長 [m]（`server_grasp.py` が計算）
- `R`: 物体座標系 → カメラ座標系の回転行列（server.py が返す、Z軸補正済み）
- `t`: 物体中心 カメラ座標系 [m]

### Z 軸補正（server.py）

物体の Z 軸（高さ方向）をカメラ Y+ 方向（画像下向き）に統一する。

```python
if R_np[1, 2] > 0:          # Z軸が画像下向き（世界の「下」方向）の場合
    R_np[:, 2] *= -1        # 反転して人間から見て↑向きに統一
```

- カメラ Y+ = 画像下方向 = 世界の「下」、カメラ Y- = 画像上方向 = 世界の「上」
- 正しい状態: `R[1,2] < 0`（Z軸が画像上向き＝人間から見て↑方向）
- Y 軸は補正しない

### メッシュ座標補正（server_grasp.py）

GraspGenerator 入力前にメッシュ点群の Z 軸を反転（server.py の Z 補正と対応）。

```python
R_corr = np.diag([1.0, 1.0, -1.0])
mesh_pts_aligned = (R_corr @ mesh_pts.T).T
```

### スケール推定（server.py）

SAM2 マスク内の深度点群から物体の高さを推定してメッシュをスケーリング。

- マスク境界 2px をエロージョンして除外（深度ノイズ対策）
- カメラ Y 方向の幅 = 物体高さ [mm] として `estimated_size_mm` を計算
- メッシュの Z 軸長に合わせてスケール係数を算出

デバッグ用に `vis_height.png` を保存（内側マスク・最高点(緑)・最低点(赤) を可視化）。

### hscale（ScalingNet 出力）

```
hand_size_real = mesh_scale_m × hscale
```

- `hscale ≈ 0.24`: 正規化空間（物体サイズ=1）での手のサイズ比
- GraspGenerator 内では `pred_ges / hscale` で正規化空間に変換している（`/ hscale` = 拡大）

## 既知の問題・注意事項

- **手のサイズ**: `grasp_generator.py` の `/ hscale`（Line 179-180）により手が大きすぎる（スパン約1.23m）。正しくは `× hscale` で約70mm になるはずだが、Shape2Gesture の学習時の実装に依存するため要検証。
- **Z 補正と R_corr の一貫性**: server.py が Z 列を反転した場合のみ server_grasp.py の R_corr（Z 反転）と整合する。server.py が補正しない場合は微妙にずれる（実用上ほぼ常に補正が入る）。
- **スケール不一致**: SAM-3D が生成する PLY のスケールが実寸と合わない場合がある。深度推定（カメラ Y 方向の高さ）でスケール補正しているが根本解決ではない。

## 共有ディレクトリ

- `server/tmp/` — サーバ・Docker 間の中間ファイル共有（Docker 内は `/workspace/tmp/`）
- `server/SAM-6D/` — SAM-6D リポジトリ（submodule）
- `server/sam-3d-objects/` — SAM-3D リポジトリ（submodule）
