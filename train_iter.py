import os
import time
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import build_dataset, build_dataloader
from model import build_model
from loss import build_loss, LossLog
from optim import build_optimizer, build_lr_scheduler
from metric import MetricLog, compute_metrics
from utils import write_scalar_to_tensorboard, save_model, set_random_seed, load_cfg_file, make_dirs, summary_results, to_cuda, dict_to_log, init_logger, create_evaluate_dict

torch.autograd.set_detect_anomaly(True)

def config_params():
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument('--config', required=True, help='the config file path')
    parser.add_argument('--overlap', required=False, help='the overlap predictor ckpt path')
    
    args = parser.parse_args()
    return args

def train_step(train_loader, model, optimizer, loss_func, epoch, total_epoch, writer, sheduler):
    model.train()
    train_loss = LossLog()
    
    loop = tqdm(enumerate(train_loader), total=len(train_loader))
    loop.set_description(f'Epoch [{epoch}/{total_epoch}]')
    for idx, data_batch in loop:
        optimizer.zero_grad()
        to_cuda(data_batch)
        
        intput_data = model.create_input(data_batch)
        predictions = model(intput_data)
        ground_truth = model.create_ground_truth(data_batch)
        
        loss = loss_func(predictions, ground_truth)
        loss["loss"].backward()
        train_loss.add_loss(loss)
        optimizer.step()

        loop.set_postfix(loss = loss["loss"].item())

        if (idx + 1) % 48 == 0:
            sheduler.step()
        
    results = summary_results("train", None, train_loss)
    write_scalar_to_tensorboard(writer, results, epoch)
    return results

def val_step(val_loader, model, val_step, eval_cfg, writer):
    model.eval()
    
    val_metrics = MetricLog()
    with torch.no_grad():
        loop = tqdm(enumerate(val_loader), total=len(val_loader))
        loop.set_description(f'Val [{val_step}]')
        for idx, data_batch in loop:
            to_cuda(data_batch)
            intput_data = model.create_input(data_batch)
            predictions = model(intput_data)
            ground_truth = model.create_ground_truth(data_batch)

            info = create_evaluate_dict(data_batch, eval_cfg)

            minibatch_metrics = compute_metrics(predictions, ground_truth, info)
            val_metrics.add_metrics(minibatch_metrics)
            
    results = summary_results("val", val_metrics, None)
    write_scalar_to_tensorboard(writer, results, val_step)
    return results

def main():
    args = config_params()
    cfg = load_cfg_file(args.config)
    timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    experiment_dir = os.path.join(cfg.experiment_name, timestamp)
    make_dirs(experiment_dir)
    
    logger = init_logger(experiment_dir)
    dict_to_log(cfg, logger)

    set_random_seed(cfg.seed)

    train_set = build_dataset(cfg.dataset.train_set)
    val_set = build_dataset(cfg.dataset.val_set)

    train_loader = build_dataloader(train_set, False, cfg.dataloader.train_loader)
    val_loader = build_dataloader(val_set, False, cfg.dataloader.val_loader)
    
    model = build_model(cfg.model)
    loss_func = build_loss(cfg.loss)
    
    if args.overlap:
        overlap_weight = torch.load(args.overlap, map_location="cpu")
        model.overlap_predictor.load_state_dict(overlap_weight)

    if torch.cuda.is_available():
        model = model.cuda()
        loss_func = loss_func.cuda()
    
    optimizer = build_optimizer(model, cfg.optimizer)
    scheduler = build_lr_scheduler(optimizer, cfg.lr_scheduler) if "lr_scheduler" in cfg else None

    summary_path = os.path.join("exp", experiment_dir, "summary")
    writer = SummaryWriter(summary_path)
    
    checkpoints_path = os.path.join("exp", experiment_dir, "checkpoints")
   
    for epoch in range(cfg.epoch):
        
        model.train()
        train_loss = LossLog()
    
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        # loop.set_description(f'Epoch [{epoch}/{total_epoch}]')
        for idx, data_batch in loop:
            model.train()
            optimizer.zero_grad()
            to_cuda(data_batch)
            
            intput_data = model.create_input(data_batch)
            predictions = model(intput_data)
            ground_truth = model.create_ground_truth(data_batch)
            
            loss = loss_func(predictions, ground_truth)
            loss["loss"].backward()
            train_loss.add_loss(loss)
            optimizer.step()

            loop.set_postfix(loss = loss["loss"].item())
            
            if (idx + 1) % 48 == 0:
                scheduler.step()
                
                tot_epoch = (epoch * len(train_loader)) // 48 + (idx + 1) // 48
                write_scalar_to_tensorboard(writer, {"learning_rate": scheduler.get_last_lr()[-1]}, tot_epoch)

                results = summary_results("train", None, train_loss)
                write_scalar_to_tensorboard(writer, results, tot_epoch)
                logger.info("Train Epoch {}: {}".format(tot_epoch, train_loss.get_loss("loss")))
                train_loss = LossLog()

                if tot_epoch % cfg.interval == 0:
                    val_results = val_step(val_loader, model, tot_epoch, cfg.get("eval"), writer)
                    logger.info("Val Epoch {}: {}".format(tot_epoch, val_results))
                    save_model(checkpoints_path, "epoch_{}".format(str(tot_epoch)), model.state_dict())
        
    writer.close()

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
