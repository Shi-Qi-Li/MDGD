from typing import Dict
import torch
import torch.nn as nn

from .builder import LOSS

@LOSS
class RelativeRotationLoss(nn.Module):
    def __init__(self, rel_weight: float = 0.0):
        super().__init__()
        self.rel_weight = rel_weight
        self.criterion = nn.L1Loss()

    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        rel_rotation_gt = ground_truth["rel_rotation_gt"].squeeze(dim=0)

        abs_rotation_pred = predictions["abs_rotation_pred"]
        
        rel_rotation_pred = abs_rotation_pred.reshape(-1, 3) @ abs_rotation_pred.reshape(-1, 3).T

        loss = self.criterion(rel_rotation_pred, rel_rotation_gt)  

        if self.rel_weight > 0:
            rel_rotation_pred_direct = predictions["rel_rotation_pred"]
            rel_rotation_pred_direct = rel_rotation_pred_direct.transpose(1, 2).reshape(rel_rotation_gt.shape)

            loss += self.rel_weight * self.criterion(rel_rotation_pred_direct, rel_rotation_gt)

        return {
            "loss": loss
        }

@LOSS
class RelativeTranslationLoss(nn.Module):
    def __init__(self, rel_weight: float = 0.0, coarse_weight: float = 0.0):
        super().__init__()
        self.rel_weight = rel_weight
        self.coarse_weight = coarse_weight
        self.criterion = nn.L1Loss()

    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        rel_translation_gt = ground_truth["rel_translation_gt"].squeeze(dim=0)
        abs_rotation_pred = predictions["abs_rotation_pred"].detach()
        abs_translation_pred = predictions["abs_translation_pred"]

        camera_num = abs_rotation_pred.shape[0]

        rel_translation_pred = abs_translation_pred - (abs_rotation_pred.reshape(-1, 3) @ torch.bmm(abs_rotation_pred.transpose(-1, -2), abs_translation_pred).transpose(0, 1).reshape(3, -1)).reshape(camera_num, 3, camera_num)

        loss = self.criterion(rel_translation_pred, rel_translation_gt)

        if self.rel_weight > 0:
            rel_translation_pred_direct = predictions["rel_translation_pred"].transpose(-1, -2)

            loss += self.rel_weight * self.criterion(rel_translation_pred_direct, rel_translation_gt)

        if self.coarse_weight > 0:
            abs_translation_pred_coarse = predictions["abs_translation_pred_coarse"]
            rel_translation_pred_coarse = abs_translation_pred_coarse - (abs_rotation_pred.reshape(-1, 3) @ torch.bmm(abs_rotation_pred.transpose(-1, -2), abs_translation_pred_coarse).transpose(0, 1).reshape(3, -1)).reshape(camera_num, 3, camera_num)

            loss += self.coarse_weight * self.criterion(rel_translation_pred_coarse, rel_translation_gt)

        return {
            "loss": loss
        }

@LOSS
class IterativeRelativeRotationLoss(nn.Module):
    def __init__(self, rel_weight: float = 0.0, factor: float = 0.5):
        super().__init__()
        self.rel_weight = rel_weight
        self.factor = factor
        self.criterion = nn.L1Loss()
        self.rel_criterion = nn.L1Loss(reduction="none")

    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        rel_rotation_gt = ground_truth["rel_rotation_gt"].squeeze(dim=0)

        abs_rotation_pred = predictions["abs_rotation_pred"]
        adj_mat = predictions["adj_mat"].float()
        mask = torch.kron(adj_mat, torch.ones((3, 3), device=adj_mat.device))
        mask_num = torch.sum(mask)
        
        iterations = len(abs_rotation_pred)
        loss = 0
        for i in range(iterations):
            current_weight = self.factor ** (iterations - i - 1)

            rel_rotation_pred = abs_rotation_pred[i].reshape(-1, 3) @ abs_rotation_pred[i].reshape(-1, 3).T

            loss += current_weight * self.criterion(rel_rotation_pred, rel_rotation_gt) 
        
            if self.rel_weight > 0:
                rel_rotation_pred_direct = predictions["rel_rotation_pred"]
                rel_rotation_pred_direct = rel_rotation_pred_direct[i].transpose(1, 2).reshape(rel_rotation_gt.shape)

                rel_loss = self.rel_criterion(rel_rotation_pred_direct, rel_rotation_gt) * mask

                loss += current_weight * self.rel_weight * torch.sum(rel_loss) / mask_num

        return {
            "loss": loss
        }

@LOSS
class IterativeRelativeTranslationLoss(nn.Module):
    def __init__(self, rel_weight: float = 0.0, coarse_weight: float = 0.0, factor: float = 0.5):
        super().__init__()
        self.rel_weight = rel_weight
        self.coarse_weight = coarse_weight
        self.factor = factor
        self.criterion = nn.L1Loss()
        self.rel_criterion = nn.L1Loss(reduction="none")

    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        rel_translation_gt = ground_truth["rel_translation_gt"].squeeze(dim=0)
        abs_rotation_pred = [x.detach() for x in predictions["abs_rotation_pred"]]
        abs_translation_pred = predictions["abs_translation_pred"]
        adj_mat = predictions["adj_mat"].float()
        adj_num = torch.sum(adj_mat)

        camera_num = rel_translation_gt.shape[0]

        iterations = len(abs_rotation_pred)
        loss = 0
        for i in range(iterations):
            current_weight = self.factor ** (iterations - i - 1)

            rel_translation_pred = abs_translation_pred[i] - (abs_rotation_pred[i].reshape(-1, 3) @ torch.bmm(abs_rotation_pred[i].transpose(-1, -2), abs_translation_pred[i]).transpose(0, 1).reshape(3, -1)).reshape(camera_num, 3, camera_num)

            loss += current_weight * self.criterion(rel_translation_pred, rel_translation_gt)

            if self.rel_weight > 0:
                rel_translation_pred_direct = predictions["rel_translation_pred"][i].transpose(-1, -2)

                rel_loss = self.rel_criterion(rel_translation_pred_direct, rel_translation_gt) * adj_mat.unsqueeze(dim=1)

                loss += current_weight * self.rel_weight * torch.sum(rel_loss) / (adj_num * 3)

            if self.coarse_weight > 0:
                abs_translation_pred_coarse = predictions["abs_translation_pred_coarse"]
                rel_translation_pred_coarse = abs_translation_pred_coarse[i] - (abs_rotation_pred[i].reshape(-1, 3) @ torch.bmm(abs_rotation_pred[i].transpose(-1, -2), abs_translation_pred_coarse[i]).transpose(0, 1).reshape(3, -1)).reshape(camera_num, 3, camera_num)

                loss += current_weight * self.coarse_weight * self.criterion(rel_translation_pred_coarse, rel_translation_gt)

        return {
            "loss": loss
        }