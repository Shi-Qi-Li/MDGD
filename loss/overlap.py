from typing import Dict
import torch
import torch.nn as nn

from .builder import LOSS

@LOSS
class OverlapLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.SmoothL1Loss()

    def forward(self, predictions: Dict, ground_truth: Dict) -> Dict[str, torch.Tensor]:
        overlap_gt = ground_truth["overlap_gt"]
        overlap_pred = predictions["overlap_pred"]
        
        loss = self.criterion(overlap_pred, overlap_gt)

        return {
            "loss": loss
        }