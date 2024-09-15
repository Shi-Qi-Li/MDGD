from typing import Dict, Optional

import os
import torch
import yaml
import logging
import numpy as np
from easydict import EasyDict as edict

from loss import LossLog
from metric import MetricLog

def set_random_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def write_scalar_to_tensorboard(writer, results, epoch):
    for key, value in results.items():
        if value != None:
            writer.add_scalar(key, value, epoch)

def save_model(checkpoints_path, name, model_state):
    saved_path = os.path.join(checkpoints_path, "{}{}".format(name, ".pth")) 
    torch.save(model_state, saved_path)

def load_cfg_file(model_cfg_path):
    with open(model_cfg_path) as cfg_file:
        cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)
    return edict(cfg)

def make_dirs(experiment_stamp, mode="train"):
    work_dir = os.path.join("exp", experiment_stamp)
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    
    if mode == "train":
        summary_dir = os.path.join(work_dir, "summary")
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        checkpoints_dir = os.path.join(work_dir, "checkpoints")
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
def summary_results(mode: str, metrics: MetricLog, loss: Optional[LossLog] = None):
    results = dict()

    if metrics is not None:
        for metric_category in metrics.all_metric_categories:
            results.update({''.join(["metrics_", mode, "/", metric_category]): metrics.get_metric(metric_category)})

    if loss is not None:
        for loss_category in loss.all_loss_categories:
            results.update({''.join(["loss_", mode, "/", loss_category]): loss.get_loss(loss_category)})

    return results

def to_cuda(data_batch: Dict):
    for key, value in data_batch.items():
        if isinstance(value, torch.Tensor):
            data_batch[key] = value.cuda()
        if isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                data_batch[key] = [v.cuda() for v in value]
            elif isinstance(value[0], list) and isinstance(value[0][0], torch.Tensor):
                data_batch[key] = [[v.cuda() for v in v_list] for v_list in value]
            else:
                data_batch[key] = value            
    
def init_logger(experiment_dir: str) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(filename=os.path.join(os.path.join("exp", experiment_dir), "log.log"))
    streamhandler = logging.StreamHandler()
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

    return logger

def dict_to_log(dictionary: Dict, logger: logging.Logger):
    for k, v in dictionary.items():
        logger.info("{}: {}".format(k, v))

def create_evaluate_dict(data_batch: Dict[str, torch.Tensor], eval_cfg):
    if eval_cfg is not None:
        eval_cfg["gt_file_path"] = data_batch["gt_file_path"]
    return eval_cfg