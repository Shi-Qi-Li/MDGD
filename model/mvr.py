from typing import Dict

import torch
import torch.nn as nn
import numpy as np

import graph_ops

from utils import integrate_trans

from .builder import MODEL, build_model

    
@MODEL
class MVRLIT(nn.Module):
    def __init__(self, overlap_cfg: Dict, at_cfg: Dict, k: int = 10, graph_mode: str = "topk") -> None:
        super().__init__()

        self.overlap_predictor = build_model(overlap_cfg)
        for param in self.overlap_predictor.parameters():
            param.requires_grad = False
        self.at = build_model(at_cfg)
        self.k = k
        self.graph_mode = graph_mode

    def get_sub_params(self, name: str):
        if name == "overlap":
            return self.overlap_predictor.parameters()
        elif name == "at":
            return self.at.parameters()
        else:
            raise NotImplementedError
        
    def create_adjacent(self, overlap_pred: torch.Tensor) -> torch.Tensor:
        if self.graph_mode == "topk":    
            k = int(min(self.k, overlap_pred.shape[-1] / 2.0))
            indices = torch.topk(overlap_pred, k=k, dim=-1, largest=True)[1]
            adj_mat = torch.zeros_like(overlap_pred, device=overlap_pred.device)
            adj_mat.scatter_(-1, indices, 1)
            adj_mat = adj_mat
        elif self.graph_mode == "threshold":
            adj_mat = (overlap_pred > 0.2)
        elif self.graph_mode == "all":
            adj_mat = torch.ones_like(overlap_pred, device=overlap_pred.device)
        else:
            raise NotImplementedError

        return adj_mat
    
    @torch.no_grad()
    def get_absolute(self, rel_rotation: torch.Tensor, rel_translation: torch.Tensor, overlap_mat: torch.Tensor, ir_mat: torch.Tensor, dim: int = 4) -> torch.Tensor:
        device = rel_rotation.device
        camera_num = overlap_mat.shape[-1]
        
        weight_mat = overlap_mat.clone().detach() * ir_mat[..., 0].squeeze(dim=0).clone().detach()
        
        weight = weight_mat.cpu().numpy()

        tree = graph_ops.kruskal(camera_num, weight.astype(np.float64))
        tree = tree.reshape(camera_num, camera_num)

        root = graph_ops.get_tree_centroid(camera_num, tree)

        rel_rotation = rel_rotation.clone().detach().cpu().numpy()
        
        if dim == 3:
            rel_mat = rel_rotation
        elif dim == 4:
            rel_rotation = rel_rotation.reshape(-1, 3, 3)
            rel_translation = rel_translation.clone().detach().cpu().numpy()
            rel_translation = rel_translation.reshape(-1, 3)
            rel_mat = integrate_trans(rel_rotation, np.expand_dims(rel_translation, axis=-1))
        else:
            raise ValueError("dim must be 3 or 4")
        
        rel_mat = rel_mat.reshape(camera_num, camera_num, dim, dim)
        rel_mat = rel_mat.transpose(0, 2, 1, 3)
        rel_mat = rel_mat.reshape(camera_num * dim, camera_num * dim)

        abs_mat = graph_ops.get_pose_from_tree(camera_num, int(root), tree, rel_mat.astype(np.float64))
        abs_mat = abs_mat.reshape(1, camera_num, dim, dim)
        abs_mat = abs_mat.astype(np.float32)
        abs_mat = torch.from_numpy(abs_mat).to(device)

        abs_weight_mat = graph_ops.get_absolute_weight_from_tree(camera_num, int(root), tree, weight.astype(np.float64))
        abs_weight_mat = abs_weight_mat.reshape(camera_num, 2)
        abs_weight_mat = abs_weight_mat.astype(np.float32)
        abs_weight_mat = torch.from_numpy(abs_weight_mat).to(device)

        abs_rotation_mat = abs_mat[..., :3, :3].reshape(1, camera_num, 9)
        abs_translation_mat = abs_mat[..., :3, 3] if dim == 4 else None

        return abs_rotation_mat, abs_translation_mat, abs_weight_mat

    def translation_refine(self, abs_rotation_pred: torch.Tensor, abs_translation_pred: torch.Tensor, rel_translation_pred: torch.Tensor, weight_mat: torch.Tensor) -> torch.Tensor:
        camera_num = weight_mat.shape[0]
        device = weight_mat.device
        abs_rotation_pred = abs_rotation_pred.detach()
        abs_translation_inv = torch.matmul(-abs_rotation_pred.transpose(-1, -2), abs_translation_pred)
        
        topk_indices = torch.nonzero(weight_mat[:, :, 0])
        value = weight_mat[topk_indices[:, 0], topk_indices[:, 1], :].repeat(1, 3).flatten()

        P = torch.diag_embed(value)
        
        rel_translation_pred = rel_translation_pred.clone().detach().unsqueeze(dim=-1)
        L = torch.matmul(abs_rotation_pred[topk_indices[:, 0]].transpose(-1, -2), rel_translation_pred[topk_indices[:, 0], topk_indices[:, 1]]).reshape(-1, 1)
        L += (abs_translation_inv[topk_indices[:, 0]] - abs_translation_inv[topk_indices[:, 1]]).reshape(-1, 1)
        count_indices = torch.arange(topk_indices.shape[0], dtype=torch.long, device=device)
        B = torch.zeros((count_indices.shape[0], 3, camera_num, 3), device=device, dtype=torch.float32)

        B[count_indices, :, topk_indices[:, 0], :] = abs_rotation_pred[topk_indices[:, 0]].transpose(-1, -2)
        B[count_indices, :, topk_indices[:, 1], :] = -abs_rotation_pred[topk_indices[:, 1]].transpose(-1, -2)
        B = B.reshape(count_indices.shape[0] * 3, camera_num * 3)

        abs_translation_pred += torch.linalg.lstsq(B.T @ P @ B, B.T @ P @ L).solution.reshape(camera_num, 3, 1)

        return abs_translation_pred

    def forward(self, data: Dict[str, torch.Tensor]):
        overlap_out = self.overlap_predictor(data)

        data["adj_mat"] = self.create_adjacent(overlap_out["overlap_pred"])

        data["abs_rotation_obs"], data["abs_translation_obs"], data["abs_weight_mat"] = self.get_absolute(data["rel_rotation_obs"], data["rel_translation_obs"], overlap_out["overlap_pred"], data["ir_mat"], 4)
        
        data["overlap_pred"] = overlap_out["overlap_pred"].unsqueeze(dim=0)

        at_out = self.at(data)
        abs_translation_pred_coarse = [x.clone() for x in at_out["abs_translation_pred"]]
        
        iteration = len(at_out["abs_rotation_pred"])
        abs_translation_pred = []
        for i in range(iteration):
            abs_translation_pred.append(self.translation_refine(at_out["abs_rotation_pred"][i], at_out["abs_translation_pred"][i], at_out["rel_translation_pred"][i], at_out["weight_mat"][i]))


        if self.training:
            return {
                "overlap_pred": overlap_out["overlap_pred"],
                "abs_rotation_pred": at_out["abs_rotation_pred"],
                "abs_translation_pred": abs_translation_pred,
                "abs_translation_pred_coarse": abs_translation_pred_coarse,
                "rel_rotation_pred": at_out["rel_rotation_pred"],
                "rel_translation_pred": at_out["rel_translation_pred"],
                "weight_mat": at_out["weight_mat"],
                "adj_mat": data["adj_mat"]
            }
        else:
            return {
                "overlap_pred": overlap_out["overlap_pred"],
                "abs_rotation_pred": at_out["abs_rotation_pred"][-1],
                "abs_translation_pred": abs_translation_pred[-1],
                "abs_translation_pred_coarse": abs_translation_pred_coarse[-1],
                "rel_rotation_pred": at_out["rel_rotation_pred"][-1],
                "rel_translation_pred": at_out["rel_translation_pred"][-1],
                "weight_mat": at_out["weight_mat"][-1],
                "adj_mat": data["adj_mat"]
            }

    def create_input(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "keypoints": data_batch["keypoints_all"],
            "descriptors": data_batch["descriptors"],
            "rel_rotation_obs": data_batch["rel_mat"][..., :3, :3].reshape(data_batch["rel_mat"].shape[:-2] + (9,)).contiguous(),
            "rel_translation_obs": data_batch["rel_mat"][..., :3, 3].contiguous(),
            "ir_mat": data_batch["ir_mat"],
            "match_mat": data_batch["match_mat"]
        }
    
    def create_ground_truth(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "overlap_gt": data_batch["gt_overlap"].squeeze_(dim=0),
            "rel_rotation_gt": data_batch["rel_gt"].reshape(data_batch["gt_overlap"].shape[-1], 4, data_batch["gt_overlap"].shape[-1], 4)[:, :3, :, :3].reshape(1, data_batch["gt_overlap"].shape[-1] * 3, data_batch["gt_overlap"].shape[-1] * 3).contiguous(),
            "rel_translation_gt": data_batch["rel_gt"].reshape(data_batch["gt_overlap"].shape[-1], 4, data_batch["gt_overlap"].shape[-1], 4)[:, :3, :, 3].reshape(1, data_batch["gt_overlap"].shape[-1], 3, data_batch["gt_overlap"].shape[-1]).contiguous(),
            "abs_rotation_gt": data_batch["abs_gt"][..., :3, :3].squeeze(dim=0)
        }