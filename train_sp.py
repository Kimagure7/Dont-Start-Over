import torch
import argparse
import torch.nn as nn
import numpy as np
import random
from utils.datasets import MoviesAndTVDatasetBuilder,SpDatasetRegistry
from utils.utils import get_train_ds_config, get_eval_ds_config
import logging
from utils.log import setup_logger,LoggerRegistry
from utils.model import RecModelRP,ModelRegistry
from utils.runner import SpRunner
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import deepspeed
from deepspeed import get_accelerator

deepspeed.utils.logging.logger.setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(
        description='train soft prompt')

    parser.add_argument("--cfg-path", default='configs/llama3_sp_amazon.yaml',
                        help="path to configuration file.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local rank for distributed training")
    parser.add_argument("--options", nargs="+",
                        help="Override specific config settings in the format key=value.")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


def setup_seeds(seed):
    if seed is not None:
        logging.info(f"Setting fixed seed: {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        get_accelerator().manual_seed_all(seed)


def convert_value(value):
    # Attempt to convert the value to an int or float if applicable
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def main():
    # Parse arguments
    args = parse_args()
    config = OmegaConf.load(args.cfg_path)
    if args.options:
        for option in args.options:
            key, value = option.split("=")
            value = convert_value(value)
            OmegaConf.update(config, key, value)

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        torch.cuda.set_device(args.local_rank)
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()
    args.world_size = torch.distributed.get_world_size()

    ds_config = get_train_ds_config(
        offload=False, dtype='fp16', stage=0, enable_hybrid_engine=False)
    ds_config["train_micro_batch_size_per_gpu"] = config.run.per_device_train_batch_size
    ds_config['gradient_accumulation_steps'] = config.run.gradient_accumulation_steps
    ds_config['train_batch_size'] = config.run.per_device_train_batch_size * \
        torch.distributed.get_world_size() * config.run.gradient_accumulation_steps
    setup_seeds(config.run.seed)
    torch.distributed.barrier(device_ids=[args.local_rank])

    # setup logger and output directory
    model_name = config.model.path.split('/')[-1]
    output_dir = Path(config.run.output_dir) / config.dataset.name / \
        (model_name + '-' + datetime.now().strftime("%Y%m%d%H%M"))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(output_dir / 'train.log')

    logging.info(OmegaConf.to_yaml(config))
    import os
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
    ###############################################################################
    # Load data
    ###############################################################################

    logging.info(f"Loading data from {config.dataset.train} and {config.dataset.valid}")
    train_data, val_data, aux_info = SpDatasetRegistry.create(name = config.dataset.name,config=config)
    logging.info(f"train data size: {len(train_data)}, val data size: {len(val_data)}")
    
    ###############################################################################
    # Build the model AND MetricLogger
    ###############################################################################
    model = ModelRegistry.create(config.dataset.name, config, nuser = aux_info['nuser'], nitem = aux_info['nitem'])
    logging.info(f"Model: {model}")
    model.to(device)
    metric_logger = LoggerRegistry.get_logger(config.dataset.name, config)
    ###############################################################################
    # Training code
    ###############################################################################

    runner = SpRunner(config, train_data, val_data, val_data,
                      model, output_dir, metric_logger, ds_config=ds_config, args=args)
    runner.train()


if __name__ == '__main__':
    main()
