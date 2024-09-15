from typing import Dict

import torch
import numpy as np

def overlap_metrics(predictions: Dict, ground_truth: Dict, info: Dict) -> Dict:
    overlap_pred = predictions["overlap_pred"].detach()
    overlap_gt = ground_truth["overlap_gt"]

    if overlap_gt.sum() == 0:
        return {}

    k = int(min(info.get("k", 10), overlap_gt.shape[0]/2.0)) if info is not None else 10
    frames = overlap_gt.shape[0]
    truely_pre = 0
    truely_gt  = k * frames
    for f in range(frames):
        gt = overlap_gt[f]
        pred = overlap_pred[f]
        # gt from large to small
        arg_gt = torch.argsort(gt, descending=True)[0:k]
        # pre from large to small
        arg_pre = torch.argsort(pred, descending=True)[0:k]
        for i in arg_pre:
            if i in arg_gt:
                truely_pre += 1
    recall = np.array(truely_pre / truely_gt)
    
    metrics = {
        "topk_recall": recall
    }
    
    return metrics