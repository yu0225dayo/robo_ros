"""
Shape2Gesture モデルを用いた把持姿勢生成モジュール

学習済みモデルをロードし、物体点群から両手の把持姿勢 (23関節 × 3次元) を生成する。

パイプライン:
    点群入力
    → PointNetDenseCls (接触領域セグメンテーション)
    → ScalingNet (手のスケール予測)
    → PartsEncoder_w_TNet (部位特徴エンコード)
    → HandVAE.finetune (手形状デコード)
    → Position_Generater_VAE (手首位置・回転角生成)
    → 座標変換 → 把持姿勢 (23関節座標)
"""

import os
import numpy as np
import torch
from torch.autograd import Variable

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model import HandVAE, PartsEncoder_w_TNet, Position_Generater_VAE
from models.model_pointnet import PointNetDenseCls, ScalingNet
from models.functions_pointnet import get_patseg_wo_target
from models.calculate_method import z_rotation_matrix, normalize_pointcloud


class GraspGenerator:
    """
    Shape2Gesture 把持姿勢生成クラス

    使用例:
        generator = GraspGenerator(model_dir="save_model", epoch=69)
        generator.load_models()
        results = generator.generate(pointcloud_np, num_samples=3)
        for left_hand, right_hand in results:
            print(left_hand.shape)  # (23, 3)
    """

    def __init__(self, model_dir: str = "save_model", epoch: int = 69):
        """
        Args:
            model_dir: 学習済みモデルが格納されたディレクトリ
            epoch:     使用するPositionVAEのエポック番号
        """
        self.model_dir = model_dir
        self.epoch = epoch
        self._loaded = False

    def load_models(self):
        """全学習済みモデルをロードする"""
        model_dir = self.model_dir

        # --- PointNet パーツセグメンテーション ---
        pointnet_path = os.path.join(model_dir, "pointnet", "pointnet_acc_partseg_best.pth")
        self._check_model_path(pointnet_path, "PointNet")
        self.pointnet = PointNetDenseCls(k=3, feature_transform=None)
        self.pointnet.load_state_dict(torch.load(pointnet_path, weights_only=True))
        self.pointnet.eval()

        # --- ScalingNet 手スケール予測 ---
        scale_path = os.path.join(model_dir, "ScalingNet", "scaleNet_best.pth")
        self._check_model_path(scale_path, "ScalingNet")
        self.scaleNet = ScalingNet()
        self.scaleNet.load_state_dict(torch.load(scale_path, weights_only=True))
        self.scaleNet.eval()

        # --- PartsEncoder 部位エンコーダ (左・右) ---
        enc_l_path = os.path.join(model_dir, "pretrained_PartsEncoder", "parts_encoder_l_best.pth")
        enc_r_path = os.path.join(model_dir, "pretrained_PartsEncoder", "parts_encoder_r_best.pth")
        self._check_model_path(enc_l_path, "PartsEncoder Left")
        self._check_model_path(enc_r_path, "PartsEncoder Right")
        self.parts_encoder_l = PartsEncoder_w_TNet()
        self.parts_encoder_r = PartsEncoder_w_TNet()
        self.parts_encoder_l.load_state_dict(torch.load(enc_l_path, weights_only=True))
        self.parts_encoder_r.load_state_dict(torch.load(enc_r_path, weights_only=True))
        self.parts_encoder_l.eval()
        self.parts_encoder_r.eval()

        # --- HandVAE 手形状デコーダ (左・右) ---
        vae_l_path = os.path.join(model_dir, "pretrained_HnadVAE_formatxy", "vae_l_best.pth")
        vae_r_path = os.path.join(model_dir, "pretrained_HnadVAE_formatxy", "vae_r_best.pth")
        self._check_model_path(vae_l_path, "HandVAE Left")
        self._check_model_path(vae_r_path, "HandVAE Right")
        self.handvae_l = HandVAE()
        self.handvae_r = HandVAE()
        self.handvae_l.load_state_dict(torch.load(vae_l_path, weights_only=True))
        self.handvae_r.load_state_dict(torch.load(vae_r_path, weights_only=True))
        self.handvae_l.eval()
        self.handvae_r.eval()

        # --- Position_Generater_VAE 手首位置・回転角生成 (左・右) ---
        pos_l_path = os.path.join(model_dir, "worst30_sampler", "position_generater_l",
                                  "Epoch", f"{self.epoch}_epoch.pth")
        pos_r_path = os.path.join(model_dir, "worst30_sampler", "position_generater_r",
                                  "Epoch", f"{self.epoch}_epoch.pth")
        self._check_model_path(pos_l_path, "PositionVAE Left")
        self._check_model_path(pos_r_path, "PositionVAE Right")
        self.position_generater_l = Position_Generater_VAE()
        self.position_generater_r = Position_Generater_VAE()
        self.position_generater_l.load_state_dict(torch.load(pos_l_path, weights_only=True))
        self.position_generater_r.load_state_dict(torch.load(pos_r_path, weights_only=True))
        self.position_generater_l.eval()
        self.position_generater_r.eval()

        self._loaded = True
        print("[GraspGenerator] 全モデルロード完了")

    def generate(
        self,
        pointcloud: np.ndarray,
        num_samples: int = 1,
    ) -> list:
        """
        物体点群から把持姿勢を生成する

        Args:
            pointcloud:  (N, 3) numpy array の物体点群 (正規化済みでなくてもよい)
            num_samples: 生成する把持候補数 (VAEなので毎回異なる結果が得られる)

        Returns:
            list of (left_hand, right_hand) タプル
                left_hand:  (23, 3) numpy array — 左手関節座標 (物体座標系)
                right_hand: (23, 3) numpy array — 右手関節座標 (物体座標系)
        """
        if not self._loaded:
            raise RuntimeError("モデルがロードされていません。load_models() を先に呼んでください。")

        # --- 前処理: 正規化 → torch Tensor ---
        point_set = normalize_pointcloud(pointcloud.copy())
        n = len(point_set)
        choice = np.random.choice(n, 2048, replace=(n < 2048))
        point_set = point_set[choice, :].astype(np.float32)
        point_tensor = torch.from_numpy(point_set)  # (2048, 3)

        with torch.no_grad():
            # --- パーツセグメンテーション ---
            pt_input = point_tensor.view(1, point_tensor.size(0), point_tensor.size(1))
            pl, pr, all_feat, plout, prout = get_patseg_wo_target(self.pointnet, pt_input)

            # --- 手スケール予測 ---
            pts = pt_input.transpose(2, 1)
            hscale_l, hscale_r = self.scaleNet(pts)[0]
            print(f"[GraspGenerator] hand scale (正規化空間 scale=1 に対する手サイズ): "
                  f"left={hscale_l.item():.4f}, right={hscale_r.item():.4f}")

            # --- 部位特徴エンコード ---
            pf_l, mu_l, logvar_l = self.parts_encoder_l(pl, all_feat)
            pf_r, mu_r, logvar_r = self.parts_encoder_r(pr, all_feat)

            # --- 手形状デコード ---
            pred_handl = self.handvae_l.finetune(pf_l)
            pred_handr = self.handvae_r.finetune(pf_r)

            wrist_format = torch.tensor([0.5, 0.5, 0.5])
            pred_handl_base = pred_handl.view(1, -1, 3) - wrist_format
            pred_handr_base = pred_handr.view(1, -1, 3) - wrist_format

            # --- 複数サンプル生成 ---
            results = []
            for i in range(num_samples):
                print(f"[GraspGenerator] サンプル {i+1}/{num_samples} 生成中...")

                # 手首位置・回転角を生成 (VAE: 毎回異なる確率的サンプリング)
                R_l, wrist_l, kld_Rl, zl = self.position_generater_l(plout, all_feat, N=1)
                R_r, wrist_r, kld_Rr, zr = self.position_generater_r(prout, all_feat, N=1)

                R_l = z_rotation_matrix(R_l).cpu().detach()
                R_r = z_rotation_matrix(R_r).cpu().detach()

                wrist_l = wrist_l.view(1, -1, 3)
                wrist_r = wrist_r.view(1, -1, 3)

                pred_ges_l_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handl_base], dim=1)
                pred_ges_r_format = torch.cat([torch.tensor([[[0, 0, 0]]]), pred_handr_base], dim=1)

                pred_ges_l = pred_ges_l_format / hscale_l
                pred_ges_r = pred_ges_r_format / hscale_r

                # 座標変換: 手形状 → スケール → 回転 → 平行移動
                pred_ges_l = pred_ges_l.unsqueeze(1) @ R_l.transpose(2, 3) + wrist_l.unsqueeze(2)
                pred_ges_r = pred_ges_r.unsqueeze(1) @ R_r.transpose(2, 3) + wrist_r.unsqueeze(2)

                left_hand = pred_ges_l[0][0].detach().cpu().numpy()   # (23, 3)
                right_hand = pred_ges_r[0][0].detach().cpu().numpy()  # (23, 3)

                results.append((left_hand, right_hand))

        print(f"[GraspGenerator] {num_samples} 個の把持姿勢を生成しました。")
        return results

    def get_segmentation(self, pointcloud: np.ndarray):
        """
        パーツセグメンテーション結果のみを返す (デバッグ用)

        Args:
            pointcloud: (N, 3) numpy array

        Returns:
            point_set:   (2048, 3) 正規化済み点群
            pred_choice: (2048,) ラベル配列 (0/1/2)
        """
        if not self._loaded:
            raise RuntimeError("モデルがロードされていません。load_models() を先に呼んでください。")

        point_set = normalize_pointcloud(pointcloud.copy())
        n = len(point_set)
        choice = np.random.choice(n, 2048, replace=(n < 2048))
        point_set = point_set[choice, :].astype(np.float32)
        point_tensor = torch.from_numpy(point_set)

        with torch.no_grad():
            pt_input = point_tensor.view(1, point_tensor.size(0), point_tensor.size(1))
            point = pt_input.transpose(2, 1)
            pred, _, _, all_feat = self.pointnet(point)
            pred_choice = pred.data.max(2)[1].cpu().numpy()[0]  # (2048,)

        return point_set, pred_choice

    @staticmethod
    def _check_model_path(path: str, name: str):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{name} モデルが見つかりません: {path}\n"
                f"Shape2Gesture_GenerationModel/save_model/ から配置してください。"
            )
