import logging
from pathlib import Path
import torch.distributed as dist
from utils.utils import get_all_reduce_mean
import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from utils.metrics import uAUC_me
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def setup_logger(filename=None):
    logger = logging.getLogger()
    if is_main_process():
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        logger.propagate = False
        
from abc import ABC, abstractmethod
from typing import Dict, Type

class BaseMetricLogger(ABC):
    """
    Abstract base class for metric logging during training and testing.
    This class defines the interface that any metric logger must implement. The purpose
    is to provide a standardized way to log and process metrics (both at the epoch and batch level)
    during training and testing of a model.
    Methods:
        train_epoch_log(epoch: int):
            Log or process metrics for a training epoch.
        test_epoch_log(epoch: int):
            Log or process metrics for a testing/validation epoch.
        train_batch_add(loss_dict: Dict[str, float]):
            Add loss metrics for a training batch. The loss_dict contains key-value pairs
            representing different loss components.
        test_batch_add(loss_dict: Dict[str, float]):
            Add loss metrics for a testing batch. The loss_dict contains key-value pairs
            representing different loss components.
        save_and_draw(output_dir):
            Save the logged data and generate plots or visualizations. This method is
            expected to be overridden by subclasses if such functionality is needed.
        test_batch_add_not_reduced(user_ids, loss_dict: Dict[str, float]):
            Record the batch metrics without reduction (raw data) for the test phase.
            This method typically involves tracking metrics for each user individually.
        not_reduced_init(user_count: int):
            Initialize internal structures to store non-reduced (raw) metrics for a 
            given number of users.
        test_epoch_log_not_reduced(output_path, local_rank):
            Finalize and log the non-reduced metrics at the end of a testing epoch.
            This might include aggregating data and writing out detailed logs to the specified output.
    """
    
    @abstractmethod
    def train_epoch_log(self, epoch: int):
        pass
    
    @abstractmethod
    def test_epoch_log(self, epoch: int):
        pass
    
    @abstractmethod
    def train_batch_add(self, loss_dict: Dict[str, float]):
        pass
    
    @abstractmethod
    def test_batch_add(self, loss_dict: Dict[str, float]):
        pass
    
    def save_and_draw(self,output_dir):
        raise NotImplementedError("save_and_draw method not implemented")
    
    @abstractmethod
    def test_batch_add_not_reduced(self, user_ids, loss_dict: Dict[str, float]):
        pass
    
    @abstractmethod
    def not_reduced_init(self, user_count: int):
        pass
    
    @abstractmethod
    def test_epoch_log_not_reduced(self, output_path,local_rank):
        pass
        
class LoggerRegistry:
    _registry = {} 
    
    @classmethod
    def register(cls, name: str):
        def decorator(logger_class: Type[BaseMetricLogger]):
            cls._registry[name] = logger_class
            return logger_class
        return decorator
    
    @classmethod
    def get_logger(cls, name: str, *args, **kwargs) -> BaseMetricLogger:
        dict = {
            "MoviesAndTV":"RatingLogger",
            "Yelp":"RatingLogger",
            "MIND":"YNLogger",
        }
        logger_name = dict.get(name, name)
        if logger_name not in cls._registry:
            raise ValueError(f"No logger registered for '{logger_name}'")
        return cls._registry[logger_name](*args, **kwargs)

@LoggerRegistry.register("RatingLogger")
class RatingLogger(BaseMetricLogger):
    """Rating Logger"""
    def __init__(self,config):
        self.celoss_list = []
        self.mseloss_list = []
        self.rmse_list = []
        self.mae_list = []
        self.acc_list = []
        self.norm_lambda = config.run.get("norm_lambda", 0.2)
        self.celoss_full = []
        self.mseloss_full = []

        
    def train_batch_add(self,loss_dict:dict):
        celoss = loss_dict['ce_loss']
        mseloss = loss_dict['mse_loss']
        with torch.no_grad():
            celoss_reduced = get_all_reduce_mean(celoss)
            mseloss_reduced = get_all_reduce_mean(mseloss)

            self.celoss_list.append(celoss_reduced.item())
            self.mseloss_list.append(mseloss_reduced.item())
            self.celoss_full.append(celoss_reduced.item())
            self.mseloss_full.append(mseloss_reduced.item())
        loss = celoss * self.norm_lambda + mseloss * (1 - self.norm_lambda)
        return loss
        
    def train_epoch_log(self,cur_epoch:int=0,auto_clean=True):
        celoss = np.mean(self.celoss_list)
        mseloss = np.mean(self.mseloss_list)
        agg_loss = celoss * self.norm_lambda + mseloss * (1 - self.norm_lambda)
        logging.info(
            "Epoch {} Training: ce_loss {:.4f}, mse_loss {:.4f}, agg_loss {:.4f}".format(
                cur_epoch, celoss, mseloss, agg_loss
            )
        )
        if auto_clean:
            self.reset_history()
        return agg_loss
        
    def reset_history(self):
        self.celoss_list = []
        self.mseloss_list = []
        self.rmse_list = []
        self.mae_list = []
        self.acc_list = []
        
    def test_batch_add(self,loss_dict:dict):
        with torch.no_grad():
            celoss, mseloss, rmse, mae, acc = loss_dict['ce_loss'], loss_dict['mse_loss'], loss_dict['rmse'], loss_dict['mae'], loss_dict['acc']
            celoss = get_all_reduce_mean(celoss)
            mseloss = get_all_reduce_mean(mseloss)
            rmse = get_all_reduce_mean(rmse)
            mae = get_all_reduce_mean(mae)
            acc = get_all_reduce_mean(acc)
            self.celoss_list.append(celoss.item())
            self.mseloss_list.append(mseloss.item())
            self.rmse_list.append(rmse.item())
            self.mae_list.append(mae.item())
            self.acc_list.append(acc.item())
            
    def test_epoch_log(self,cur_epoch:int=0,auto_clean=True):
        celoss = np.mean(self.celoss_list)
        mseloss = np.mean(self.mseloss_list)
        rmse = np.mean(self.rmse_list)
        mae = np.mean(self.mae_list)
        acc = np.mean(self.acc_list)
        agg_loss = celoss * self.norm_lambda + mseloss * (1 - self.norm_lambda)

        logging.info(
            "Epoch {} Testing: ce_loss {:.4f}, mse_loss {:.4f}, agg_loss {:.4f}, rmse {:.4f}, mae {:.4f}, acc {:.4f}".format(
                cur_epoch, celoss, mseloss, agg_loss, rmse, mae, acc
            )
        )
        if auto_clean:
            self.reset_history()
        return agg_loss
    
    def save_and_draw(self, output_dir):  
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        window_size = 50
        # Plot CE loss
        smoothed_ce_loss = moving_average(self.celoss_full, window_size)
        smoothed_steps = range(window_size, len(self.celoss_full) + 1)
        steps = range(1, len(self.celoss_full) + 1)
        ax1.plot(steps, self.celoss_full, 'b-', alpha=0.3, label='Original CE Loss')
        ax1.plot(smoothed_steps, smoothed_ce_loss, 'r-', linewidth=2, label='Smoothed CE Loss')
        ax1.set_title('Cross Entropy Loss History')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        mse_loss_cut = self.mseloss_full[50:] # drop first 50 steps for smoothing
        steps = range(1, len(mse_loss_cut) + 1)
        smoothed_mse_loss = moving_average(mse_loss_cut, window_size)
        smoothed_steps = range(window_size, len(mse_loss_cut) + 1)
        ax2.plot(steps, mse_loss_cut, 'b-', alpha=0.3, label='Original MSE Loss')
        ax2.plot(smoothed_steps, smoothed_mse_loss, 'r-', linewidth=2, label='Smoothed MSE Loss')
        ax2.set_title('MSE Loss History')
        ax2.set_xlabel('Steps') 
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        # Adjust spacing between subplots
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'loss_history.png'))
        plt.close()
    
    def save(self,output_dir):
        # Save loss history to JSON
        loss_history = {
            'ce_loss': self.celoss_full,
            'mse_loss': self.mseloss_full
        }
        output_path = os.path.join(output_dir, 'loss_history.json')
        with open(output_path, 'w') as f:
            json.dump(loss_history, f)
        logging.info(f"Saved loss history to {output_path}")
    
    def not_reduced_init(self,user_count):
        self.user_ce_losses = {i: [] for i in range(user_count)}
        self.user_mse_losses = {i: [] for i in range(user_count)}
        self.user_mae_losses = {i: [] for i in range(user_count)}
        
    def test_batch_add_not_reduced(self,user_ids,loss_dict:dict,local_rank):
        ce_losses = loss_dict['ce_loss'] 
        mse_losses = loss_dict['mse_loss']
        mae_losses = loss_dict['mae']

        # Gather losses from all processes
        gathered_user_ids = [torch.zeros_like(user_ids) for _ in range(torch.distributed.get_world_size())]
        gathered_ce_losses = [torch.zeros_like(ce_losses) for _ in range(torch.distributed.get_world_size())]
        gathered_mse_losses = [torch.zeros_like(mse_losses) for _ in range(torch.distributed.get_world_size())]
        gathered_mae_losses = [torch.zeros_like(mae_losses) for _ in range(torch.distributed.get_world_size())]
        
        torch.distributed.all_gather(gathered_user_ids, user_ids)
        torch.distributed.all_gather(gathered_ce_losses, ce_losses) 
        torch.distributed.all_gather(gathered_mse_losses, mse_losses)
        torch.distributed.all_gather(gathered_mae_losses, mae_losses)

        # Store losses by user on rank 0
        if local_rank == 0:
            for proc_user_ids, proc_ce_losses, proc_mse_losses, proc_mae_losses in zip(gathered_user_ids, gathered_ce_losses, gathered_mse_losses, gathered_mae_losses):
                for user_id, ce_loss, mse_loss, mae_loss in zip(proc_user_ids, proc_ce_losses, proc_mse_losses, proc_mae_losses):
                    self.user_ce_losses[user_id.item()].append(ce_loss.item())
                    self.user_mse_losses[user_id.item()].append(mse_loss.item())
                    self.user_mae_losses[user_id.item()].append(mae_loss.item())
                    
    def test_epoch_log_not_reduced(self, output_path,local_rank):
        total_data = 0
        if local_rank == 0:
            for user_id in self.user_ce_losses.keys():
                total_data += len(self.user_ce_losses[user_id])
            logging.info(f"Total data points: {total_data}")
            # Calculate average losses per user
            avg_user_losses = {}
            for user_id in self.user_ce_losses.keys():
                if len(self.user_ce_losses[user_id]) > 0:  # Only include users who have data
                    avg_user_losses[str(user_id)] = {
                        'ce_loss': self.user_ce_losses[user_id],
                        'mse_loss': self.user_mse_losses[user_id],
                        'mae_loss': self.user_mae_losses[user_id]
                    }
            
            # Save to JSON file
            output_path = os.path.join(self.output_dir, 'user_losses.json')
            with open(output_path, 'w') as f:
                json.dump(avg_user_losses, f, indent=2)
            
            logging.info(f"Saved per-user losses to {output_path}")
        

@LoggerRegistry.register("YNLogger")
class YNLogger(BaseMetricLogger):
    """YN Logger"""
    def __init__(self,config):
        self.loss_list = []
        self.loss_list_full = []
        self.results_logits = []
        self.labels = []
        self.users = []
        
    def train_batch_add(self,loss_dict:dict):
        loss = loss_dict['loss']
        with torch.no_grad():
            loss_reduced = get_all_reduce_mean(loss)
            self.loss_list.append(loss_reduced.item())
            self.loss_list_full.append(loss_reduced.item())
        return loss
    
    def train_epoch_log(self,cur_epoch:int=0,auto_clean=True):
        loss = np.mean(self.loss_list)
        logging.info(
            "Epoch {} Training: loss {:.4f}".format(
                cur_epoch, loss
            )
        )
        if auto_clean:
            self.reset_history()
        return loss
    
    def reset_history(self):
        self.loss_list = []
        self.results_logits = []
        self.labels = []
        self.users = []
        
    def test_batch_add(self, loss_dict:dict):
        with torch.no_grad():
            loss = loss_dict['loss']
            loss = get_all_reduce_mean(loss)
            self.loss_list.append(loss.item())
            
            # Process the batch data immediately
            if is_dist_avail_and_initialized():
                # Gather and process in smaller chunks
                batch_logits = loss_dict['logits'].detach()
                batch_labels = loss_dict['Label'].detach()
                batch_users = loss_dict['UserID'].detach()
                
                # Store as CPU numpy arrays to save GPU memory
                self.results_logits.append(batch_logits.cpu().numpy())
                self.labels.append(batch_labels.cpu().numpy())
                self.users.append(batch_users.cpu().numpy())
            else:
                # For single process, just store CPU numpy arrays
                self.results_logits.append(loss_dict['logits'].detach().cpu().numpy())
                self.labels.append(loss_dict['Label'].detach().cpu().numpy())
                self.users.append(loss_dict['UserID'].detach().cpu().numpy())
        return loss
        
    def test_epoch_log(self, cur_epoch:int=0, auto_clean=True):
        loss = np.mean(self.loss_list)
        logging.info(f"Epoch {cur_epoch} Testing: loss {loss:.4f}")
        
        # Concatenate numpy arrays instead of tensors
        all_logits = np.concatenate(self.results_logits)
        all_labels = np.concatenate(self.labels)
        all_users = np.concatenate(self.users)
        
        if is_dist_avail_and_initialized():
            # Gather data sizes from all processes
            local_size = torch.tensor([len(all_logits)], dtype=torch.long, device='cuda')
            size_list = [torch.ones_like(local_size) for _ in range(get_world_size())]
            dist.all_gather(size_list, local_size)
            
            # Convert numpy arrays to tensors for gathering
            gathered_logits = []
            gathered_labels = []
            gathered_users = []
            
            # Gather from all processes
            if is_main_process():
                for i, size in enumerate(size_list):
                    if i == get_rank():
                        gathered_logits.append(all_logits)
                        gathered_labels.append(all_labels)
                        gathered_users.append(all_users)
                    else:
                        # Receive data from other processes
                        logits = torch.zeros(size.item(), dtype=torch.float32, device='cuda')
                        labels = torch.zeros(size.item(), dtype=torch.float32, device='cuda')
                        users = torch.zeros(size.item(), dtype=torch.float32, device='cuda')
                        
                        dist.recv(logits, src=i)
                        dist.recv(labels, src=i)
                        dist.recv(users, src=i)
                        
                        gathered_logits.append(logits.cpu().numpy())
                        gathered_labels.append(labels.cpu().numpy())
                        gathered_users.append(users.cpu().numpy())
            else:
                # Send data to main process
                logits_tensor = torch.tensor(all_logits, dtype=torch.float32, device='cuda')
                labels_tensor = torch.tensor(all_labels, dtype=torch.float32, device='cuda')
                users_tensor = torch.tensor(all_users, dtype=torch.float32, device='cuda')
                
                dist.send(logits_tensor, dst=0)
                dist.send(labels_tensor, dst=0)
                dist.send(users_tensor, dst=0)
            
            # Compute metrics on main process
            if is_main_process():
                all_logits = np.concatenate(gathered_logits)
                all_labels = np.concatenate(gathered_labels)
                all_users = np.concatenate(gathered_users)
                
                auc = roc_auc_score(all_labels, all_logits)
                uauc, _, _ = uAUC_me(all_users, all_logits, all_labels)
                logging.info(f"Epoch {cur_epoch} Testing: auc {auc:.4f}, uauc {uauc:.4f}")
        else:
            # Single process mode
            auc = roc_auc_score(all_labels, all_logits)
            uauc, _, _ = uAUC_me(all_users, all_logits, all_labels)
            logging.info(f"Epoch {cur_epoch} Testing: auc {auc:.4f}, uauc {uauc:.4f}")
        
        if auto_clean:
            self.reset_history()
        return loss
        
    def save_and_draw(self, output_dir):
        
        # Create a single figure
        plt.figure(figsize=(10, 6))
        
        def moving_average(data, window_size):
            return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
        
        window_size = min(50, len(self.loss_list_full) // 4 + 1)  # Ensure window size is appropriate
        
        # Plot loss
        if len(self.loss_list_full) > window_size:
            smoothed_loss = moving_average(self.loss_list_full, window_size)
            smoothed_steps = range(window_size, len(self.loss_list_full) + 1)
            steps = range(1, len(self.loss_list_full) + 1)
            plt.plot(steps, self.loss_list_full, 'b-', alpha=0.3, label='Original Loss')
            plt.plot(smoothed_steps, smoothed_loss, 'r-', linewidth=2, label='Smoothed Loss')
        else:
            steps = range(1, len(self.loss_list_full) + 1)
            plt.plot(steps, self.loss_list_full, 'b-', label='Loss')
        
        plt.title('Loss History')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'loss_history.png'))
        plt.close()
        
    def save(self,output_dir):
        # Save loss history to JSON
        loss_history = {
            'loss': self.loss_list_full,
        }
        output_path = os.path.join(output_dir, 'loss_history.json')
        with open(output_path, 'w') as f:
            json.dump(loss_history, f)
        logging.info(f"Saved loss history to {output_path}")
        
    def not_reduced_init(self,user_count):
        self.user_ce_losses = {i: [] for i in range(user_count)}
        
    def test_batch_add_not_reduced(self,user_ids,loss_dict:dict,local_rank):
        ce_losses = loss_dict['loss'] 

        # Gather losses from all processes
        gathered_user_ids = [torch.zeros_like(user_ids) for _ in range(torch.distributed.get_world_size())]
        gathered_ce_losses = [torch.zeros_like(ce_losses) for _ in range(torch.distributed.get_world_size())]
        
        torch.distributed.all_gather(gathered_user_ids, user_ids)
        torch.distributed.all_gather(gathered_ce_losses, ce_losses) 

        # Store losses by user on rank 0
        if local_rank == 0:
            for proc_user_ids, proc_ce_losses in zip(gathered_user_ids, gathered_ce_losses):
                for user_id, ce_loss in zip(proc_user_ids, proc_ce_losses):
                    self.user_ce_losses[user_id.item()].append(ce_loss.item())
                    
    def test_epoch_log_not_reduced(self, output_path,local_rank):
        total_data = 0
        if local_rank == 0:
            for user_id in self.user_ce_losses.keys():
                total_data += len(self.user_ce_losses[user_id])
            logging.info(f"Total data points: {total_data}")
            # Calculate average losses per user
            avg_user_losses = {}
            for user_id in self.user_ce_losses.keys():
                if len(self.user_ce_losses[user_id]) > 0:  # Only include users who have data
                    avg_user_losses[str(user_id)] = {
                        'ce_loss': self.user_ce_losses[user_id],
                    }
            
            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(avg_user_losses, f, indent=2)
            
            logging.info(f"Saved per-user losses to {output_path}")