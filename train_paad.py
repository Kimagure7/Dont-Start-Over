import torch
import argparse
import torch.nn as nn
import numpy as np
import random
from utils.datasets import MoviesAndTVDatasetBuilder,SpDatasetRegistry 
from utils.utils import get_train_ds_config, get_eval_ds_config
import logging
from utils.log import setup_logger,LoggerRegistry
from utils.model import RecModelRP,AdapterModelRP,ModelRegistry
from utils.runner import AdRunner
from omegaconf import OmegaConf
from datetime import datetime
from pathlib import Path
import deepspeed
from deepspeed import get_accelerator

deepspeed.utils.logging.logger.setLevel(logging.WARNING)

def parse_args():
    parser = argparse.ArgumentParser(
        description='train soft prompt for rating prediction')

    parser.add_argument("--cfg-path", default='configs/ad_llama3_amazon_test.yaml',
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

def get_output_dir(config):
    now = datetime.now()
    output_dir = Path(config.run.output_dir) / "adapter" / \
        config.dataset.name / \
        now.strftime("%Y%m%d") / \
        now.strftime("%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

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
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
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
    output_dir = get_output_dir(config)
    setup_logger(output_dir / 'train.log')
    logging.info(OmegaConf.to_yaml(config))
    import os
    logging.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', None)}")
    #torch.distributed.barrier()
    ###############################################################################
    # Load data
    ###############################################################################

    logging.info(f"Loading data from {config.dataset.train} and {config.dataset.valid}")
    train_data, val_data, aux_info = SpDatasetRegistry.create_ad(name = config.dataset.name,config=config)
    logging.info(f"train data size: {len(train_data)}, val data size: {len(val_data)}")
    
    ###############################################################################
    # Build the model AND MetricLogger
    ###############################################################################
    
    model = ModelRegistry.create_test(config.dataset.name, config, nuser = aux_info['nuser'], nitem = aux_info['nitem'])
    logging.info(f"Model: {model}")
    model.to(device)
    metric_logger = LoggerRegistry.get_logger(config.dataset.name, config)

    ###############################################################################
    # Training code
    ###############################################################################

    runner = AdRunner(config, train_data, val_data, val_data,
                      model, output_dir,metric_logger, ds_config=ds_config, args=args, device=device)
    runner.train()


if __name__ == '__main__':
    main()
