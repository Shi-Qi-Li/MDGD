import os
import sys
import pickle
from tqdm import tqdm

sys.path.append(os.getcwd())

import torch

from dataset import build_dataset, build_dataloader
from utils import set_random_seed


def create_cache():
    torch.multiprocessing.set_start_method("spawn")
    total_epoch = 150

    set_random_seed(3407)

    train_set = build_dataset({
        "name": "Scene",
        "data_path": "data",
        "split": "train",
        "ird": 0.07,
        "point_sample": False,
        "frame_sample": False,
        "frame_limit": [8, 60],
        "point_limit": 5000,
        "processes": 8
    })

    train_loader = build_dataloader(
        train_set, 
        True, 
        {
            "num_workers": 0,
            "batch_size": 1
        }
    )

    save_path = "data/cache"
    os.makedirs(save_path, exist_ok=True)
    
    for epoch in range(total_epoch):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        loop.set_description(f'Epoch [{epoch}/{total_epoch}]')
        for idx, data_batch in loop:
            data = {}
            data["gt_overlap"] = data_batch["gt_overlap"].squeeze(0).detach().cpu().numpy()
            data["rel_mat"] = data_batch["rel_mat"].squeeze(0).detach().cpu().numpy()
            data["ir_mat"] = data_batch["ir_mat"].squeeze(0).detach().cpu().numpy()
            data["match_mat"] = data_batch["match_mat"].squeeze(0).detach().cpu().numpy()
            data["abs_gt"] = data_batch["abs_gt"].squeeze(0).detach().cpu().numpy()
            data["rel_gt"] = data_batch["rel_gt"].squeeze(0).detach().cpu().numpy()
            data["scene"] = data_batch["scene"][0]

            with open(os.path.join(save_path, f"{epoch}_{idx}.pkl"), "wb") as f:
                pickle.dump(data, f)

if __name__ == "__main__":
    create_cache()