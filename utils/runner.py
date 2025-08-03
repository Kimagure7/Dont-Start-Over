import torch
from utils.log import RatingLogger,YNLogger
from utils.model import RecModelRP, PromptMigrationModel, AdapterModelRP, AdapterModel
import torch.nn.functional as F
from pathlib import Path
from datetime import  timedelta
import time
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
import deepspeed
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from deepspeed.ops.adam import FusedAdam
from transformers import get_scheduler
import logging
from utils.utils import get_all_reduce_mean
from sklearn.cluster import KMeans
import random
import json
import matplotlib.pyplot as plt

def prepare_sample(samples, device=torch.device('cuda')):
    def _apply_to_sample(f, sample):
        if len(sample) == 0:
            return {}

        def _apply(x):
            if torch.is_tensor(x):
                return f(x)
            elif isinstance(x, dict):
                return {key: _apply(value) for key, value in x.items()}
            elif isinstance(x, list):
                return [_apply(x) for x in x]
            else:
                return x

        return _apply(sample)

    def _move_to_device(tensor, device):
        return tensor.to(device)

    if device is not None:
        samples = _apply_to_sample(
            lambda x: _move_to_device(x, device), samples)

    return samples


class BaseRunner:
    def __init__(self):
        self._optimizer = None
        self._scaler = None
        self._dataloaders = None
        self._lr_sched = None

    def _load_checkpoint(self, ckpt_path):
        raise NotImplementedError

    @property
    def resume_ckpt_path(self):
        return self.config.run.get("resume_ckpt_path", None)

    @property
    def max_epoch(self):
        return int(self.config.run.max_epoch)

    def _get_train_params(self):
        num_params = 0
        p_wd, p_non_wd = [], []
        for n, p in self._model.named_parameters():
            if not p.requires_grad:
                continue
            num_params += p.data.nelement()
            p_non_wd.append(p)
        logging.info("Number of parameters: {}".format(num_params))
        optim_params = [{"params": p_non_wd}]

        return optim_params

    @property
    def init_lr(self):
        return float(self.config.run.init_lr)

    @property
    def min_lr(self):
        return float(self.config.run.min_lr)


class SpRunner(BaseRunner):
    def __init__(self, config, train_data, val_data, test_data, model, output_dir, metric_logger, ds_config=None, args=None, device=torch.device('cuda')):
        super().__init__()
        self.config = config
        self.ds_config = ds_config
        self.args = args
        self.device = device
        self._model: RecModelRP = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.evaluate_only = config.run.evaluate
        self.start_epoch = 0

        self.output_dir = output_dir
        self.metric_logger: RatingLogger | YNLogger = metric_logger
        self.local_rank = args.local_rank
        self._init_dataloader()
        self._init_deepspeed()
        self.store_ckpt = True

    def _init_dataloader(self):
        if self.local_rank == -1:
            train_sampler = RandomSampler(self.train_data)
            eval_sampler = SequentialSampler(self.val_data)
            test_sampler = SequentialSampler(self.test_data)
        else:
            train_sampler = DistributedSampler(self.train_data)
            eval_sampler = DistributedSampler(self.val_data)
            test_sampler = DistributedSampler(self.test_data)
        self.train_loader = DataLoader(
            self.train_data, sampler=train_sampler, batch_size=self.config.run.per_device_train_batch_size)
        self.val_loader = DataLoader(
            self.val_data, sampler=eval_sampler, batch_size=self.config.run.per_device_eval_batch_size)
        self.test_loader = DataLoader(
            self.test_data, sampler=test_sampler, batch_size=self.config.run.per_device_eval_batch_size)

    def _init_deepspeed(self):
        # 初始化optimizer
        optimizer = FusedAdam(self._get_train_params(), lr=self.init_lr)
        lr_scheduler = get_scheduler(
            name=self.config.run.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.config.run.get("num_warmup_steps", 0),
            num_training_steps=self.max_epoch * len(self.train_loader),
        )
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self._model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=self.ds_config,
            args=self.args,
            dist_init_required=True)

    def _train_epoch(self, cur_epoch):
        # Turn on training mode which enables dropout.
        self.model.train()
        for i, samples in enumerate(tqdm(self.train_loader, desc=f"Epoch {cur_epoch} Training", disable=(self.args.local_rank != 0))):
            samples = prepare_sample(samples, device=self.device)
            loss_dict = self.model(samples)  # celoss and mseloss
            loss = self.metric_logger.train_batch_add(loss_dict)
            
            self.model.backward(loss)
            self.model.step()

        return self.metric_logger.train_epoch_log(cur_epoch)

    def _evaluate_epoch(self, cur_epoch, skip_reload=False, not_reduction=False):
        if not_reduction:
            return self._evaluate_epoch_not_reduction(cur_epoch, skip_reload)
        # Turn on evaluation mode which disables dropout.
        if cur_epoch == "best":
            if not skip_reload:
                self._load_best_checkpoint()
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
        self.model.eval()
        with torch.no_grad():
            for i, samples in enumerate(tqdm(data_loader, desc=f"Epoch {cur_epoch} Validation", disable=(self.args.local_rank != 0))):
                samples = prepare_sample(samples, device=self.device)
                loss_dict = self.model.generate_for_samples(samples)  # celoss, mseloss, logits
                self.metric_logger.test_batch_add(loss_dict)
        return self.metric_logger.test_epoch_log(cur_epoch)


    def _evaluate_epoch_not_reduction(self, cur_epoch, skip_reload):
        if cur_epoch == "best":
            if not skip_reload:
                self._load_best_checkpoint()
            data_loader = self.test_loader
        else:
            data_loader = self.val_loader
        self.model.eval()
        self.metric_logger.not_reduced_init(user_count= self.user_count)

        with torch.no_grad():
            for i, samples in enumerate(tqdm(data_loader, desc=f"Epoch {cur_epoch} Validation", disable=(self.args.local_rank != 0))):
                samples = prepare_sample(samples, device=self.device)
                loss_dict = self.model.generate_for_samples(samples,reduction='none')
                
                # Get user IDs and losses for this batch
                user_ids = samples['UserID']
                self.metric_logger.test_batch_add_not_reduced(user_ids, loss_dict,self.args.local_rank)
        output_path = os.path.join(self.output_dir, f"user_losses.json")
        self.metric_logger.test_epoch_log_not_reduced(output_path, self.args.local_rank)
        return { 
            "ce_loss": 0,
            "mse_loss": 0,
            "accuracy": 0,
            "agg_loss": 0
        }
        
    @property
    def user_count(self):
        return self._model.user_embedding.weight.shape[0] - self.config.dataset.get("ratio",0)
        
    def train(self):
        start_time = time.time()
        best_loss = 100000
        best_epoch = 0
        not_change = 0  # 用于early stop

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            print(self.resume_ckpt_path)
            self._load_checkpoint(self.resume_ckpt_path)
        try:
            if not self.evaluate_only:  # with training
                for cur_epoch in range(self.start_epoch, self.max_epoch):
                    # training phase
                    
                    logging.info("Start training")
                    self._train_epoch(cur_epoch)

                    # evaluation phase
                    loss = self._evaluate_epoch(cur_epoch)
                    self.metric_logger.save(self.output_dir)
                    # 记录best model
                    if loss < best_loss:
                        best_loss = loss
                        best_epoch = cur_epoch
                        not_change = 0
                        if self.store_ckpt:
                            self._save_checkpoint(cur_epoch, is_best=True)
                    elif loss < best_loss + 0.02 and self.store_ckpt: 
                        self._save_checkpoint(cur_epoch) 
                        
                    not_change += 1
                    if not_change > 4 and cur_epoch > self.config.run.min_epoch:
                        logging.info(
                            "Early stop. The results has not changed up to 4 epochs.")
                        break
                    
                    if self.test_data is not self.val_data:
                        self._evaluate_epoch(cur_epoch="best", skip_reload=True)
                        
            if self.evaluate_only:
                # 暂时不改了
                logging.info("Evaluating after training or eval only.")
                status_dict = self._evaluate_epoch(
                    cur_epoch="eval", skip_reload=self.evaluate_only, not_reduction=False)
                
        finally:
            total_time = time.time() - start_time
            total_time_str = str(timedelta(seconds=int(total_time)))
            logging.info("Training time {}".format(total_time_str))
            
            if self.args.local_rank == 0 and not self.evaluate_only:
                self.metric_logger.save_and_draw(self.output_dir)
            torch.distributed.barrier(device_ids=[self.args.local_rank])
                
        
    def _save_checkpoint(self, cur_epoch, is_best=False):
        """
        Save the checkpoint at the current epoch.
        """
        save_to_model = os.path.join(
            self.output_dir,
            "checkpoint_model_{}.pth".format("best" if is_best else cur_epoch),
        )
        if self.args.local_rank <= 0:
            self._model.save_checkpoint(save_to_model)

        
    def _load_best_checkpoint(self):
        """
        Load the best checkpoint.
        """
        ckpt_path = os.path.join(self.output_dir, "checkpoint_model_best.pth")
        best_model = self._model._reload_best_model(
            model=self.model, ckpt_path=ckpt_path)
        
        
        return best_model
    
    def _resume_checkpoint(self):
        pass

class AdRunner(SpRunner):
    # 仅重写获取可训练参数，防止出错
    def __init__(self, config, train_data, val_data, test_data, model:AdapterModelRP, output_dir, metric_logger, ds_config=None, args=None, device=torch.device('cuda')):
        BaseRunner.__init__(self)
        self.config = config
        self.ds_config = ds_config
        self.args = args
        self.device = device
        self._model: AdapterModelRP = model.to(device)
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.evaluate_only = config.run.evaluate
        self.start_epoch = 0

        self.output_dir = output_dir
        self.metric_logger: RatingLogger | YNLogger = metric_logger
        self.local_rank = args.local_rank
        self._init_dataloader()
        self._init_deepspeed()
        self.store_ckpt = False
        
    def _train_epoch(self, cur_epoch):
        self.model.train()
        contra_list = []
        for i, samples in enumerate(tqdm(self.train_loader, desc=f"Epoch {cur_epoch} Training", disable=(self.args.local_rank != 0))):
            samples = prepare_sample(samples, device=self.device)
            loss_dict = self.model(samples)  # celoss and mseloss
            loss = self.metric_logger.train_batch_add(loss_dict)          
            self.model.backward(loss)
            self.model.step()
            
        return self.metric_logger.train_epoch_log(cur_epoch)
        