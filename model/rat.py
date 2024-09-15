from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ConfidenceAttention
from .builder import MODEL

def batch_r6d2mat(r6d: torch.Tensor):
    assert r6d.shape[-1] == 6, "Invalid 6D representation!"

    component1, component2 = r6d[:, 0:3], r6d[:, 3:6]

    component1 = F.normalize(component1, p=2, dim=-1)

    component3 = torch.linalg.cross(component1, component2)
    component3 = F.normalize(component3, p=2, dim=-1)

    component2 = torch.linalg.cross(component3, component1)

    matrix = torch.cat([component1.unsqueeze(dim=-1), component2.unsqueeze(dim=-1), component3.unsqueeze(dim=-1)], dim=-1)
    
    return matrix

class GRU(nn.Module):
    def __init__(self, input_dim: int = 32, hidden_dim: int = 32) -> None:
        super(GRU, self).__init__()

        self.mlpz = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.mlpr = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.mlpq = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        xh = torch.cat([x, h], dim=-1)
        
        z = torch.sigmoid(self.mlpz(xh))
        r = torch.sigmoid(self.mlpr(xh))
        q = torch.tanh(self.mlpq(torch.cat([x, r * h], dim=-1)))

        h = (1 - z) * h + z * q

        return h

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, factor: int = 4):
        super(MLP, self).__init__()

        self.pre_layer = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * factor),
            nn.LayerNorm(hidden_dim * factor),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * factor, hidden_dim)
        )
        self.post_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        x = self.pre_layer(x)
        x = F.relu(self.layers(x) + x)
        x = self.post_layer(x)
        return x

@MODEL
class MAIT(nn.Module):
    def __init__(self, rotation_dim: int, translation_dim: int, confidence_dim: int, repeat_num: int, abs_num: int, rel_num: int):
        super(MAIT, self).__init__()

        self.rotation_dim = rotation_dim
        self.translation_dim = translation_dim
        self.confidence_dim = confidence_dim
        self.repeat_num = repeat_num
        self.abs_num = abs_num
        self.rel_num = rel_num

        self.rel_rotation_proj = MLP(9, 32, rotation_dim)
        self.abs_rotation_proj = MLP(9, 32, rotation_dim)
        self.rel_translation_proj = MLP(3, 32, translation_dim)
        self.abs_translation_proj = MLP(3, 32, translation_dim)

        self.rel_confidence_proj = MLP(3, 32, confidence_dim)
        self.abs_confidence_proj = MLP(2, 32, confidence_dim)

        self.rel_rotation_gru = GRU(rotation_dim + confidence_dim, rotation_dim + confidence_dim)
        self.abs_rotation_gru = GRU(rotation_dim + confidence_dim, rotation_dim + confidence_dim)
        self.rel_translation_gru = GRU(translation_dim + confidence_dim, translation_dim + confidence_dim)
        self.abs_translation_gru = GRU(translation_dim + confidence_dim, translation_dim + confidence_dim)
        
        self.rel_rotation_creator = MLP(rotation_dim + rotation_dim, 64, rotation_dim)
        self.abs_rotation_creator = MLP(rotation_dim + rotation_dim, 64, rotation_dim)
        self.rel_translation_creator = MLP(translation_dim + translation_dim + rotation_dim, 128, translation_dim)
        self.abs_translation_creator = MLP(translation_dim + rotation_dim + translation_dim, 128, translation_dim)

        self.rel_confidence_creator = MLP(confidence_dim + confidence_dim, 64, confidence_dim)

        self.rotation_attention = ConfidenceAttention(rotation_dim, confidence_dim, 4, 0.1)
        self.translation_attention = ConfidenceAttention(translation_dim, confidence_dim, 4, 0.1)

        self.rel_rotation_norm = nn.LayerNorm(rotation_dim)
        self.abs_rotation_norm = nn.LayerNorm(rotation_dim)
        self.rel_translation_norm = nn.LayerNorm(translation_dim)
        self.abs_translation_norm = nn.LayerNorm(translation_dim)

        self.rel_confidence_norm = nn.LayerNorm(confidence_dim)
        self.abs_confidence_norm = nn.LayerNorm(confidence_dim)

        self.abs_rotation_head = MLP(rotation_dim, 32, 6)
        self.rel_rotation_head = MLP(rotation_dim, 32, 6)

        self.abs_translation_head = MLP(translation_dim, 32, 3)
        self.rel_translation_head = MLP(translation_dim, 32, 3)

        self.weight_head = MLP(confidence_dim, 64, 1)

    def forward(self, data: Dict[str, torch.Tensor]):
        rel_rotation_obs = data["rel_rotation_obs"]
        abs_rotation_obs = data["abs_rotation_obs"]
        rel_translation_obs = data["rel_translation_obs"]
        abs_translation_obs = data["abs_translation_obs"]
        camera_num = rel_rotation_obs.shape[-2]
        adj_mat = data["adj_mat"]
        mask = (adj_mat == 0).reshape(camera_num, camera_num)

        overlap_ratio = data["overlap_pred"]
        inlier_ratio = data["ir_mat"]
        abs_weight = data["abs_weight_mat"]
        
        rel_confidence_feature = self.rel_confidence_proj(torch.cat([overlap_ratio.unsqueeze(dim=-1), inlier_ratio], dim=-1))
        abs_confidence_feature = self.abs_confidence_proj(abs_weight)        

        rel_confidence_feature = rel_confidence_feature.reshape(camera_num, camera_num, -1)
        abs_confidence_feature = abs_confidence_feature.reshape(camera_num, 1, -1)

        rel_rotation_feature = self.rel_rotation_proj(rel_rotation_obs)
        abs_rotation_feature = self.abs_rotation_proj(abs_rotation_obs)
        rel_translation_feature = self.rel_translation_proj(rel_translation_obs)
        abs_translation_feature = self.abs_translation_proj(abs_translation_obs)

        rel_rotation_feature = rel_rotation_feature.reshape(camera_num, camera_num, -1)
        abs_rotation_feature = abs_rotation_feature.reshape(camera_num, 1, -1)
        rel_translation_feature = rel_translation_feature.reshape(camera_num, camera_num, -1)
        abs_translation_feature = abs_translation_feature.reshape(camera_num, 1, -1)

        abs_rotation_pred, abs_translation_pred, rel_rotation_pred, rel_translation_pred, weight_mat = [], [], [], [], []

        for it in range(self.repeat_num):
            
            for _ in range(self.abs_num):
                new_abs_rotation = self.abs_rotation_creator(torch.cat([rel_rotation_feature, abs_rotation_feature.repeat(1, camera_num, 1)], dim=-1))
                new_abs_rotation_feature, new_abs_confidence = self.rotation_attention(abs_rotation_feature, new_abs_rotation, new_abs_rotation, abs_confidence_feature, rel_confidence_feature, mask)
                abs_rotation_feature, abs_confidence_feature = torch.split(self.abs_rotation_gru(torch.cat([new_abs_rotation_feature, new_abs_confidence], dim=-1), torch.cat([abs_rotation_feature, abs_confidence_feature], dim=-1)), [self.rotation_dim, self.confidence_dim], dim=-1)
                abs_rotation_feature = self.abs_rotation_norm(abs_rotation_feature)
                abs_confidence_feature = self.abs_confidence_norm(abs_confidence_feature)

            for _ in range(self.rel_num):
                new_rel_rotation = self.rel_rotation_creator(torch.cat([abs_rotation_feature.transpose(0, 1).repeat(camera_num, 1, 1), abs_rotation_feature.repeat(1, camera_num, 1)], dim=-1))
                new_rel_confidence = self.rel_confidence_creator(torch.cat([abs_confidence_feature.transpose(0, 1).repeat(camera_num, 1, 1), abs_confidence_feature.repeat(1, camera_num, 1)], dim=-1))
                rel_rotation_feature, rel_confidence_feature = torch.split(self.rel_rotation_gru(torch.cat([new_rel_rotation, new_rel_confidence], dim=-1), torch.cat([rel_rotation_feature, rel_confidence_feature], dim=-1)), [self.rotation_dim, self.confidence_dim], dim=-1)
                rel_rotation_feature = self.rel_rotation_norm(rel_rotation_feature)
                rel_confidence_feature = self.rel_confidence_norm(rel_confidence_feature)

            for _ in range(self.abs_num):
                new_abs_translation = self.abs_translation_creator(torch.cat([abs_translation_feature.repeat(1, camera_num, 1), rel_rotation_feature, rel_translation_feature], dim=-1))
                new_abs_translation_feature, new_abs_confidence = self.translation_attention(abs_translation_feature, new_abs_translation, new_abs_translation, abs_confidence_feature, rel_confidence_feature, mask)
                abs_translation_feature, abs_confidence_feature = torch.split(self.abs_translation_gru(torch.cat([new_abs_translation_feature, new_abs_confidence], dim=-1), torch.cat([abs_translation_feature, abs_confidence_feature], dim=-1)), [self.translation_dim, self.confidence_dim], dim=-1)
                abs_translation_feature = self.abs_translation_norm(abs_translation_feature)
                abs_confidence_feature = self.abs_confidence_norm(abs_confidence_feature)

            for _ in range(self.rel_num):
                new_rel_translation = self.rel_translation_creator(torch.cat([abs_translation_feature.transpose(0, 1).repeat(camera_num, 1, 1), abs_translation_feature.repeat(1, camera_num, 1), rel_rotation_feature], dim=-1))
                new_rel_confidence = self.rel_confidence_creator(torch.cat([abs_confidence_feature.transpose(0, 1).repeat(camera_num, 1, 1), abs_confidence_feature.repeat(1, camera_num, 1)], dim=-1))
                rel_translation_feature, rel_confidence_feature = torch.split(self.rel_translation_gru(torch.cat([new_rel_translation, new_rel_confidence], dim=-1), torch.cat([rel_translation_feature, rel_confidence_feature], dim=-1)), [self.translation_dim, self.confidence_dim], dim=-1)
                rel_translation_feature = self.rel_translation_norm(rel_translation_feature)
                rel_confidence_feature = self.rel_confidence_norm(rel_confidence_feature)

            if self.training or it == self.repeat_num - 1:
                abs_rotation = self.abs_rotation_head(abs_rotation_feature)
                abs_rotation = batch_r6d2mat(abs_rotation.reshape(-1, 6))
                abs_translation = self.abs_translation_head(abs_translation_feature).transpose(-1, -2)

                rel_rotation = self.rel_rotation_head(rel_rotation_feature)
                rel_rotation = batch_r6d2mat(rel_rotation.reshape(-1, 6))
                rel_rotation = rel_rotation.reshape(camera_num, camera_num, 3, 3)
                rel_translation = self.rel_translation_head(rel_translation_feature)

                weight = self.weight_head(rel_confidence_feature)
                weight = F.sigmoid(weight) * adj_mat.unsqueeze(dim=-1) 

                abs_rotation_pred.append(abs_rotation)
                abs_translation_pred.append(abs_translation)
                rel_rotation_pred.append(rel_rotation)
                rel_translation_pred.append(rel_translation)
                weight_mat.append(weight)

        prediction = {
            "abs_rotation_pred": abs_rotation_pred,
            "abs_translation_pred": abs_translation_pred, # N * 3 * 1
            "rel_rotation_pred": rel_rotation_pred,
            "rel_translation_pred": rel_translation_pred, # N * N * 3
            "weight_mat": weight_mat
        }

        return prediction