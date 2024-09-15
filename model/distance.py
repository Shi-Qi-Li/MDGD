from typing import Dict

import torch
import torch.nn as nn

from .builder import MODEL

class InvertedResidualBlock(nn.Module):
    def __init__(self, dim: int, factor: int = 4):
        super(InvertedResidualBlock, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(dim, dim * factor),
            nn.LayerNorm(dim * factor),
            nn.ReLU(inplace=True),
            nn.Linear(dim * factor, dim),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return x + self.layers(x)

@MODEL
class MatchDistanceNet(nn.Module):
    def __init__(self, dim: int = 64, factor: int = 4, layer_num: int = 2):
        super(MatchDistanceNet, self).__init__()

        self.proj = nn.Linear(8, dim)

        self.blocks = nn.ModuleList()
        for _ in range(layer_num):
            self.blocks.append(InvertedResidualBlock(dim, factor))

        self.head = nn.Sequential(
            nn.Linear(dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pair_feature = data_dict["match_mat"]
        
        pair_feature = self.proj(pair_feature)
        for block in self.blocks:
            pair_feature = block(pair_feature)
        pair_feature = self.head(pair_feature).squeeze(dim=-1)

        mask = torch.ones_like(pair_feature[0]).fill_diagonal_(0)
        overlap = pair_feature.squeeze(dim=0) * mask

        prediction = {
            "overlap_pred": overlap
        }

        return prediction

    def create_input(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "match_mat": data_batch["match_mat"]
        }
    
    def create_ground_truth(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "overlap_gt": data_batch["gt_overlap"].squeeze_(dim=0)
        }