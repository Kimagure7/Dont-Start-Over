import os
import torch
import random
import pickle
from torch.utils.data import Dataset
import logging
from abc import ABC, abstractmethod
from typing import Dict, Type
import pandas as pd
import numpy as np
import ast
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
from typing import List, Dict
from collections import Counter
from utils.user_select import *

class DatasetBuilder(ABC):
    @abstractmethod
    def build(self, config: dict):
        pass
    
    @abstractmethod
    def build_datasets_ad(self, config: dict):
        pass
        
    @property
    def user_num(self):
        return len(self.user_dict)
    
    def get_train_data(self, config, dataset):
        """
        Retrieves the training subset of the dataset based on the training indices.

        This method computes the training indices using the provided configuration
        and then filters the dataset to include only rows corresponding to these indices.

        Parameters:
            config: An object or dictionary containing configuration parameters needed 
                    to determine the training indices.
            dataset (pandas.DataFrame): The complete dataset which must include a 'UserID' column.
            
        Returns:
            pandas.DataFrame: A DataFrame containing only the training data rows.
        """
        train_indices = self.get_train_indices(config,dataset)
        train_data_filtered = dataset[dataset['UserID'].isin(train_indices)]
        return train_data_filtered
    
    def get_train_indices(self, config, dataset):
        mode = config.dataset.get('mode', 0) 
        if mode == 0:
            train_indices = range(config.dataset.get('train_ratio', 2000))
        elif mode == 1:
            train_indices = select_users_with_stratified_variance_sampling(
                dataset=dataset,
                num_users_to_select=config.dataset.get('train_ratio', 2000),
                num_groups=8,
                weight_type="normal"
            )
        elif mode == 2:
            train_indices = select_users_with_clustering_and_variance_sampling(
                dataset=dataset,
                embedding_path=config.model.soft_prompt_path,
                num_users_to_select=config.dataset.get('train_ratio', 2000),
                num_clusters=15,
                num_variance_groups=5,
                min_users_per_cluster=50,
                random_state=42,
                weight_type="uniform"
            )
        elif mode == 3:
            train_indices = select_users_with_clustering_and_variance_sampling(
                dataset=dataset,
                embedding_path=config.model.soft_prompt_path,
                num_users_to_select=config.dataset.get('train_ratio', 2000),
                num_clusters=15,
                num_variance_groups=5,
                min_users_per_cluster=50,
                random_state=config.run.get('kmeans_seed', config.run.get('seed',42)), 
                weight_type="normal"
            )
        elif mode == 4:
            train_indices = ON_based_cluster(
                embedding_path=config.model.get('ffn_matrix', None),
                num_users_to_select=config.dataset.get('train_ratio', 2000),
                num_clusters=15,
            )
        elif mode == 5:
            train_indices = ON_based_cluster_and_loss_sampling(
                embedding_path=config.model.get('ffn_matrix', None),
                loss_file_path=config.model.get('loss_file', None),
                num_users_to_select=config.dataset.get('train_ratio', 2000),
            )
        elif mode == 6:
            train_indices = ON_based_cluster_and_variance_sampling(
                dataset=dataset,
                embedding_path=config.model.get('ffn_matrix', None),
                num_users_to_select=config.dataset.get('train_ratio', 2000),
            )
        return train_indices

class SpDatasetRegistry:
    _builders: Dict[str, Type[DatasetBuilder]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(builder_cls: Type[DatasetBuilder]):
            if name in cls._builders:
                raise ValueError(f"Dataset '{name}' is already registered!")
            cls._builders[name] = builder_cls
            return builder_cls
        return wrapper

    @classmethod
    def get_builder(cls, name: str) -> Type[DatasetBuilder]:
        if name not in cls._builders:
            raise ValueError(f"Dataset '{name}' is not registered!")
        return cls._builders[name]

    @classmethod
    def create(cls, name: str, config):
        builder_cls = cls.get_builder(name)
        return builder_cls().build(config)
    
    @classmethod
    def create_ad(cls, name: str, config):
        builder_cls = cls.get_builder(name)
        return builder_cls().build_datasets_ad(config)
    
    @classmethod
    def create_tf(cls, name: str, config):
        builder_cls = cls.get_builder(name)
        return builder_cls().build_datasets_tf(config)

class EntityDictionary:
    def __init__(self):
        self.idx2entity = []
        self.entity2idx = {}

    def add_entity(self, e):
        if e not in self.entity2idx:
            self.entity2idx[e] = len(self.idx2entity)
            self.idx2entity.append(e)

    def __len__(self):
        return len(self.idx2entity)
    
    def save(self, path):
        data = {
            'idx2entity': self.idx2entity,
            'entity2idx': self.entity2idx
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dictionary file not found at {path}")
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.idx2entity = data['idx2entity']
        self.entity2idx = data['entity2idx']
        return self

class MoviesAndTVDataset(Dataset):
    def __init__(self, data):
        super(MoviesAndTVDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'UserID': row['UserID'], 
            # 'ItemID':row['item'],
            'Rating': row['rating'], 
            'ItemTitle': row['ItemTitle']
        }
    
class MindDataset(Dataset):
    def __init__(self,data:pd.DataFrame):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            'UserID': row['UserID'],
            'ItemTitle': row['News_Title'],
            'Label': row['Label'],
            'History_Interact_Title': row['History_Interact_Title']
        }

class YelpDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        return {
            'UserID': row['UserID'],
            # 'ItemID': row['ItemID'],
            'Rating': row['rating'],
            'ItemTitle': row['business_description']
        }
@SpDatasetRegistry.register("MoviesAndTV")
class MoviesAndTVDatasetBuilder(DatasetBuilder):
    def __init__(self):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()

    def build(self,config):
        train_data = pd.read_csv(config.dataset.train, sep='\t')
        valid_data = pd.read_csv(config.dataset.valid, sep='\t')
        
        if config.dataset.get('user_dict',None):
            self.user_dict.load(config.dataset.user_dict)
        else:
            all_user = train_data['UserID'].unique()
            all_user_shuffled = np.random.permutation(all_user)
            for user in all_user_shuffled:
                self.user_dict.add_entity(user)
                self.user_dict.save("user_dict.pickle")
        
        train_data['UserID'] = train_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        valid_data['UserID'] = valid_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])

        train_dataset = MoviesAndTVDataset(train_data)
        valid_dataset = MoviesAndTVDataset(valid_data)
        return train_dataset, valid_dataset, {"nuser": len(self.user_dict), "nitem": len(self.item_dict)}
    
    def build_datasets_ad(self,config):
        
        train_data = pd.read_csv(config.dataset.train, sep='\t')
        valid_data = pd.read_csv(config.dataset.valid, sep='\t')
        
        # 加载dict
        if config.dataset.get('user_dict',None):
            self.user_dict.load(config.dataset.user_dict)
        else:
            all_user = train_data['UserID'].unique()
            all_user_shuffled = np.random.permutation(all_user)
            for user in all_user_shuffled:
                self.user_dict.add_entity(user)
            self.user_dict.save("user_dict.pickle")
        
        train_data['UserID'] = train_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        valid_data['UserID'] = valid_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        
        train_data_filtered = self.get_train_data(config, train_data)
        
        return MoviesAndTVDataset(train_data_filtered), MoviesAndTVDataset(valid_data), {"nuser": len(self.user_dict), "nitem": len(self.item_dict)}

@SpDatasetRegistry.register("MIND")
class MINDDatasetBuilder(DatasetBuilder):
    def __init__(self):
        self.user_dict = EntityDictionary()

    def build(self,config):
        train_data = pd.read_csv(config.dataset.train, sep='\t')
        valid_data = pd.read_csv(config.dataset.valid, sep='\t')
        
        if config.dataset.get("user_dict",None):
            self.user_dict.load(config.dataset.user_dict)
        else:
            all_user = train_data['UserID'].unique()
            all_user_shuffled = np.random.permutation(all_user)
            for user in all_user_shuffled:
                self.user_dict.add_entity(user)

        train_data['UserID'] = train_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        valid_data['UserID'] = valid_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        

        train_data["History_Interact_Title"] = train_data["History_Interact_Title"].apply(lambda x: ast.literal_eval(x))
        valid_data["History_Interact_Title"] = valid_data["History_Interact_Title"].apply(lambda x: ast.literal_eval(x))
        

        train_dataset = MindDataset(train_data)
        valid_dataset = MindDataset(valid_data)
        return train_dataset, valid_dataset, {"nuser": len(self.user_dict), "nitem": 0}
    
    def build_datasets_ad(self,config):
        
        train_data = pd.read_csv(config.dataset.train, sep='\t')
        valid_data = pd.read_csv(config.dataset.valid, sep='\t')
        
        if config.dataset.get('user_dict',None):
            self.user_dict.load(config.dataset.user_dict)
        else:
            all_user = train_data['UserID'].unique()
            all_user_shuffled = np.random.permutation(all_user)
            for user in all_user_shuffled:
                self.user_dict.add_entity(user)
        
        train_data['UserID'] = train_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        valid_data['UserID'] = valid_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        
        train_data["History_Interact_Title"] = train_data["History_Interact_Title"].apply(lambda x: ast.literal_eval(x))
        valid_data["History_Interact_Title"] = valid_data["History_Interact_Title"].apply(lambda x: ast.literal_eval(x))
        train_data['rating'] = train_data['Label']
        
        train_data_filtered = self.get_train_data(config, train_data)
        
        return MindDataset(train_data_filtered), MindDataset(valid_data), {"nuser": len(self.user_dict), "nitem": 0}
    
        
@SpDatasetRegistry.register("Yelp")
class YelpDatasetBuilder(DatasetBuilder):
    def __init__(self):
        self.user_dict = EntityDictionary()
        self.item_dict = EntityDictionary()

    def build(self,config):
        train_data = pd.read_csv(config.dataset.train, sep='\t')
        valid_data = pd.read_csv(config.dataset.valid, sep='\t')
        
        if config.dataset.get('user_dict',None):
            self.user_dict.load(config.dataset.user_dict)
        else:
            all_user = train_data['UserID'].unique()
            all_user_shuffled = np.random.permutation(all_user)
            for user in all_user_shuffled:
                self.user_dict.add_entity(user)
            self.user_dict.save("user_dict.pickle")
        
        train_data['UserID'] = train_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        valid_data['UserID'] = valid_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])

        train_dataset = YelpDataset(train_data)
        valid_dataset = YelpDataset(valid_data)
        return train_dataset, valid_dataset, {"nuser": len(self.user_dict), "nitem": len(self.item_dict)}
    
    def build_datasets_ad(self,config):
        
        train_data = pd.read_csv(config.dataset.train, sep='\t')
        valid_data = pd.read_csv(config.dataset.valid, sep='\t')
        
        if config.dataset.get('user_dict',None):
            self.user_dict.load(config.dataset.user_dict)
        else:
            all_user = train_data['UserID'].unique()
            all_user_shuffled = np.random.permutation(all_user)
            for user in all_user_shuffled:
                self.user_dict.add_entity(user)
            self.user_dict.save("user_dict.pickle")
        
        train_data['UserID'] = train_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        valid_data['UserID'] = valid_data['UserID'].apply(lambda x: self.user_dict.entity2idx[x])
        train_data_filtered = self.get_train_data(config, train_data)

        return YelpDataset(train_data_filtered), YelpDataset(valid_data), {"nuser": len(self.user_dict), "nitem": len(self.item_dict)}
    