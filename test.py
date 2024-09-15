import os
import time
import argparse
import torch
from tqdm import tqdm

from dataset import build_dataset, build_dataloader
from model import build_model
from metric import MetricLog, compute_metrics
from utils import set_random_seed, load_cfg_file, make_dirs, summary_results, to_cuda, dict_to_log, init_logger, create_evaluate_dict

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--config', required=True, help='the config file path')
    parser.add_argument('--ckpt', required=True, help='checkpoint path')
    
    args = parser.parse_args()
    return args

def test_step(test_loader, model, eval_cfg):
    model.eval()
    
    test_metrics = MetricLog()
    with torch.no_grad():
        loop = tqdm(enumerate(test_loader), total=len(test_loader))
        loop.set_description("Test")
        for idx, data_batch in loop:
            to_cuda(data_batch)
            intput_data = model.create_input(data_batch)
            predictions = model(intput_data)
            ground_truth = model.create_ground_truth(data_batch)

            info = create_evaluate_dict(data_batch, eval_cfg)

            minibatch_metrics = compute_metrics(predictions, ground_truth, info)
            test_metrics.add_metrics(minibatch_metrics)
            
    results = summary_results("test", test_metrics, None)
    return results

def main():
    args = config_params()
    cfg = load_cfg_file(args.config)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join(cfg.experiment_name, "test", timestamp)
    make_dirs(experiment_dir, "test")
    
    logger = init_logger(experiment_dir)
    dict_to_log(cfg, logger)

    set_random_seed(cfg.seed)

    test_set = build_dataset(cfg.dataset.test_set)
    test_loader = build_dataloader(test_set, False, cfg.dataloader.test_loader)

    model = build_model(cfg.model)
    
    model_weight = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(model_weight)

    if torch.cuda.is_available():
        model = model.cuda()
    
    test_results = test_step(test_loader, model, cfg.get("eval"))
    logger.info("Test result: {}".format(test_results))

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()