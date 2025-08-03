from typing import List
import json
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from collections import Counter, defaultdict
import torch
import random
from tqdm import tqdm

def select_users_with_stratified_weighted_sampling(
    loss_file_path: str,
    num_users_to_select: int = 2000,
    num_groups: int = 5,
    weight_type: str = "linear"
) -> List[int]:
    with open(loss_file_path, 'r') as f:
        data = json.load(f)
        
    # 2. Calculate average MSE loss per user
    if 'mse_loss' in data['0'].keys():
        mse_loss_dict = {
            int(user): sum(user_data['mse_loss'])/len(user_data['mse_loss']) 
            for user, user_data in data.items()
        }
        users_df = pd.DataFrame.from_dict(mse_loss_dict, orient='index', columns=['loss']).sort_values('loss')
    else:
        ce_loss_dict = {
            int(user): sum(user_data['ce_loss'])/len(user_data['ce_loss']) 
            for user, user_data in data.items()
        }
        users_df = pd.DataFrame.from_dict(ce_loss_dict, orient='index', columns=['loss']).sort_values('loss')
        
    # Assign each user to a group (stratify by loss)
    users_df['group'] = pd.qcut(
        users_df['loss'], 
        q=num_groups,
        labels=False
    )
    
    # Define group weights based on the specified type
    if weight_type == "linear":
        weights = np.arange(1, num_groups + 1)
    elif weight_type == "exponential":
        weights = 2 ** np.arange(1, num_groups + 1)
    elif weight_type == "quadratic":
        weights = (np.arange(1, num_groups + 1)) ** 2
    elif weight_type == "normal":
        # Create normal distribution-like weights (peaked in the middle)
        x = np.linspace(-3, 3, num_groups)
        weights = np.exp(-x**2/2)
        weights = weights + 0.1
        # Add small constant to ensure no zero weights
    else:
        raise ValueError(f"Invalid weight_type: {weight_type}")
    
    # Normalize weights to get probabilities
    group_weights = weights / weights.sum()
    
    # Calculate how many users to sample from each group
    samples_per_group = (group_weights * num_users_to_select).astype(int)
    samples_per_group[-1] = num_users_to_select - samples_per_group[:-1].sum()  # Adjust for rounding
    
    # Sample users from each group
    selected_users = []
    for group_num in range(num_groups):
        group_users = users_df[users_df['group'] == group_num]
        n_samples = min(samples_per_group[group_num], len(group_users))
        selected = group_users.sample(n=n_samples, random_state=42)
        selected_users.extend(selected.index.tolist())
    
    logging.info(
        f"Selected {len(selected_users)} users for training using "
        f"{weight_type} weighted stratified sampling across {num_groups} groups by loss"
    )
    
    return selected_users

def select_users_with_stratified_variance_sampling(
    dataset: pd.DataFrame,
    num_users_to_select: int = 2000,
    num_groups: int = 8,
    weight_type: str = "linear"
) -> List[int]:
    # Calculate rating variance per user
    user_rating_variance = dataset.groupby('UserID')['rating'].var()
    
    # Convert to DataFrame and sort
    variance_df = pd.DataFrame.from_dict(
        user_rating_variance.to_dict(),
        orient='index',
        columns=['rating_variance']
    ).sort_values('rating_variance')
    
    # Assign each user to a group (stratify by variance)
    variance_df['group'] = pd.qcut(
        variance_df['rating_variance'],
        q=num_groups,
        labels=False,
        duplicates='drop'  # In case of duplicate variance values
    )
    
    # Adjust num_groups in case some were dropped due to duplicates
    actual_num_groups = variance_df['group'].nunique()
    if actual_num_groups < num_groups:
        logging.warning(f"Reduced number of groups from {num_groups} to {actual_num_groups} due to duplicate variance values")
        num_groups = actual_num_groups
    
    # Define group weights based on the specified type
    if weight_type == "linear":
        weights = np.arange(1, num_groups + 1)
    elif weight_type == "exponential":
        weights = 2 ** np.arange(1, num_groups + 1)
    elif weight_type == "quadratic":
        weights = (np.arange(1, num_groups + 1)) ** 2
    elif weight_type == "normal":
        # Create normal distribution-like weights (peaked in the middle)
        x = np.linspace(-3, 3, num_groups)  # Adjust range for desired peak width
        weights = np.exp(-x**2/2)  # Gaussian curve
        weights = weights + 0.1  # Add small constant to ensure no zero weights
    else:
        raise ValueError(f"Invalid weight_type: {weight_type}")
    
    # Normalize weights to get probabilities
    group_weights = weights / weights.sum()
    
    # Calculate how many users to sample from each group
    samples_per_group = (group_weights * num_users_to_select).astype(int)
    samples_per_group[-1] = num_users_to_select - samples_per_group[:-1].sum()  # Adjust for rounding
    
    # Sample users from each group
    selected_users = []
    for group_num in range(num_groups):
        group_users = variance_df[variance_df['group'] == group_num]
        n_samples = min(samples_per_group[group_num], len(group_users))
        selected = group_users.sample(n=n_samples, random_state=42)
        selected_users.extend(selected.index.tolist())
    
    logging.info(
        f"Selected {len(selected_users)} users for training using "
        f"{weight_type} weighted stratified sampling across {num_groups} groups "
        f"based on rating variance"
    )
    
    return selected_users

def select_users_with_embedding_and_loss(
    path: str,
    loss_file_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    num_loss_groups: int = 5,
    min_users_per_cluster: int = 50,
    random_state: int = 42,
    weight_type: str = "normal"
) -> List[int]:
    """
    Select users by clustering them based on embeddings and sampling based on loss values
    within each cluster using normal distribution.
    
    Args:
        path: Path to the file containing user embeddings and loss data
        num_users_to_select: Total number of users to select (default: 6000)
        num_clusters: Number of clusters for K-means on embeddings (default: 30)
        num_loss_groups: Number of loss groups within each cluster (default: 5)
        min_users_per_cluster: Minimum number of users to select from each cluster (default: 50)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        List of selected user IDs
    """
    # 1. Load user embeddings and loss data
    sp_path = torch.load(path, map_location='cpu', weights_only=True)
    user_embedding = sp_path['user_embedding']['weight'].numpy()
    
    # Load loss data
    with open(loss_file_path, 'r') as f:
        data = json.load(f)
    
    # 2. Calculate average MSE loss per user
    if 'mse_loss' in data['0'].keys():
        mse_loss_dict = {
            int(user): sum(user_data['mse_loss'])/len(user_data['mse_loss']) 
            for user, user_data in data.items()
        }
        users_df = pd.DataFrame.from_dict(mse_loss_dict, orient='index', columns=['loss'])
    else:
        ce_loss_dict = {
            int(user): sum(user_data['ce_loss'])/len(user_data['ce_loss']) 
            for user, user_data in data.items()
        }
        users_df = pd.DataFrame.from_dict(ce_loss_dict, orient='index', columns=['loss'])
    
    # 3. Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    users_df['cluster'] = kmeans.fit_predict(user_embedding)
    
    # 4. Calculate actual cluster sizes
    cluster_counts = Counter(users_df['cluster'])
    
    # 5. Initialize selection
    selected_users = []
    remaining_users = num_users_to_select
    
    # 6. First pass: select minimum users from each cluster
    for cluster_id in range(num_clusters):
        cluster_users = users_df[users_df['cluster'] == cluster_id]
        n_samples = min(min_users_per_cluster, len(cluster_users))
        
        if n_samples > 0:
            # Sort by loss to prepare for stratified sampling
            cluster_users = cluster_users.sort_values('loss')
            
            # Create loss groups within the cluster
            try:
                cluster_users['loss_group'] = pd.qcut(
                    cluster_users['loss'],
                    q=num_loss_groups,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                # If not enough unique values, assign all to same group
                cluster_users['loss_group'] = 0
            
            # Calculate actual number of loss groups
            actual_loss_groups = cluster_users['loss_group'].nunique()
            
            # Create normal distribution weights for loss groups
            if actual_loss_groups > 1:
                if weight_type == "normal":
                    x = np.linspace(-2, 2, actual_loss_groups)
                    weights = np.exp(-x**2/2)  # Gaussian curve
                    weights = weights + 0.1  # Add small constant
                    weights = weights / weights.sum()  # Normalize
                elif weight_type == "linear":
                    weights = np.arange(1, actual_loss_groups + 1)
                    weights = weights / weights.sum()
            else:
                weights = np.array([1.0])
            
            # Sample from each loss group with normal distribution weighting
            samples_per_group = (weights * n_samples).astype(int)
            samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()  # Adjust for rounding
            
            for group_num in range(actual_loss_groups):
                group_df = cluster_users[cluster_users['loss_group'] == group_num]
                n = min(samples_per_group[group_num], len(group_df))
                selected = group_df.sample(n=n, random_state=random_state)
                selected_users.extend(selected.index.tolist())
            
            remaining_users -= n_samples
    
    # 7. Second pass: distribute remaining users proportionally to cluster sizes
    if remaining_users > 0:
        total_users = sum(cluster_counts.values())
        for cluster_id in range(num_clusters):
            cluster_users = users_df[
                (users_df['cluster'] == cluster_id) & 
                (~users_df.index.isin(selected_users))
            ]
            
            if len(cluster_users) == 0:
                continue
                
            # Calculate proportional allocation
            cluster_quota = int(remaining_users * (cluster_counts[cluster_id] / total_users))
            n_samples = min(cluster_quota, len(cluster_users))
            
            if n_samples > 0:
                # Sort by loss for stratified sampling
                cluster_users = cluster_users.sort_values('loss')
                
                # Create loss groups
                try:
                    cluster_users['loss_group'] = pd.qcut(
                        cluster_users['loss'],
                        q=num_loss_groups,
                        labels=False,
                        duplicates='drop'
                    )
                except ValueError:
                    cluster_users['loss_group'] = 0
                
                # Sample with normal distribution weighting
                actual_loss_groups = cluster_users['loss_group'].nunique()
                if actual_loss_groups > 1:
                    x = np.linspace(-2, 2, actual_loss_groups)
                    weights = np.exp(-x**2/2)
                    weights = weights + 0.1
                    weights = weights / weights.sum()
                else:
                    weights = np.array([1.0])
                
                samples_per_group = (weights * n_samples).astype(int)
                samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
                
                for group_num in range(actual_loss_groups):
                    group_df = cluster_users[cluster_users['loss_group'] == group_num]
                    n = min(samples_per_group[group_num], len(group_df))
                    selected = group_df.sample(n=n, random_state=random_state)
                    selected_users.extend(selected.index.tolist())
                
                remaining_users -= n_samples
    
    # 8. Final adjustment if we didn't get exactly num_users_to_select due to rounding
    if len(selected_users) < num_users_to_select:
        remaining = num_users_to_select - len(selected_users)
        remaining_candidates = users_df[~users_df.index.isin(selected_users)]
        if len(remaining_candidates) > 0:
            additional = remaining_candidates.sample(n=remaining, random_state=random_state)
            selected_users.extend(additional.index.tolist())
    
    if len(selected_users) > num_users_to_select:
        selected_users = selected_users[:num_users_to_select]
    
    # 9. Log the distribution
    selected_clusters = users_df[
        users_df.index.isin(selected_users)
    ]['cluster'].value_counts().to_dict()
    
    logging.info(
        f"Selected {len(selected_users)} users from {num_clusters} embedding clusters "
        f"with loss-based normal sampling. Distribution: {selected_clusters}"
    )
    
    return selected_users

def select_users_with_clustering_and_variance_sampling(
    dataset: pd.DataFrame,
    embedding_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    num_variance_groups: int = 5,
    min_users_per_cluster: int = 50,
    random_state: int = 42,
    weight_type: str = "normal"
) -> List[int]:
    """
    Select users by first clustering them based on embeddings (loaded from path) and then 
    sampling based on rating variance within each cluster using normal distribution.
    
    Args:
        dataset: DataFrame containing user rating data with columns ['UserID', 'rating']
        embedding_path: Path to the file containing user embeddings
        num_users_to_select: Total number of users to select
        num_clusters: Number of clusters for K-means on embeddings
        num_variance_groups: Number of variance groups within each cluster
        min_users_per_cluster: Minimum number of users to select from each cluster
        random_state: Random seed for reproducibility
    
    Returns:
        List of selected user IDs
    """
    # 1. Load user embeddings from path
    sp_path = torch.load(embedding_path, map_location='cpu', weights_only=True)
    user_embeddings = sp_path['user_embedding']['weight'].numpy()
    
    # 2. Prepare user data - calculate rating variance per user
    user_variance = dataset.groupby('UserID')['rating'].var().reset_index()
    user_variance.columns = ['UserID', 'rating_variance']
    
    # 3. Perform K-means clustering on user embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(user_embeddings)
    user_variance['cluster'] = clusters
    
    # 4. Calculate actual cluster sizes
    cluster_counts = Counter(clusters)
    
    # 5. Initialize selection
    selected_users = []
    remaining_users = num_users_to_select
    
    # 6. First pass: select minimum users from each cluster
    for cluster_id in range(num_clusters):
        cluster_users = user_variance[user_variance['cluster'] == cluster_id]
        n_samples = min(min_users_per_cluster, len(cluster_users))
        
        if n_samples > 0:
            # Sort by variance to prepare for stratified sampling
            cluster_users = cluster_users.sort_values('rating_variance')
            
            # Create variance groups within the cluster
            try:
                cluster_users['variance_group'] = pd.qcut(
                    cluster_users['rating_variance'],
                    q=num_variance_groups,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                # If not enough unique values, assign all to same group
                cluster_users['variance_group'] = 0
            
            # Calculate actual number of variance groups
            actual_variance_groups = cluster_users['variance_group'].nunique()
            
            # Create normal distribution weights for variance groups
            weights = np.array([1.0])
            if actual_variance_groups > 1:
                if weight_type == "normal":
                    x = np.linspace(-2, 2, actual_variance_groups)
                    weights = np.exp(-x**2/2)  # Gaussian curve
                    weights = weights + 0.1  # Add small constant
                    weights = weights / weights.sum()  # Normalize
                elif weight_type == "linear":
                    weights = np.arange(1, actual_variance_groups + 1)
                    weights = weights / weights.sum()
                elif weight_type == 'uniform':
                    weights = np.ones(actual_variance_groups)
                    weights = weights / weights.sum()

            
            # Sample from each variance group with normal distribution weighting
            samples_per_group = (weights * n_samples).astype(int)
            samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
            
            for group_num in range(actual_variance_groups):
                group_df = cluster_users[cluster_users['variance_group'] == group_num]
                n = min(samples_per_group[group_num], len(group_df))
                selected = group_df.sample(n=n, random_state=random_state)
                selected_users.extend(selected['UserID'].tolist())
            
            remaining_users -= n_samples
    
    # 7. Second pass: distribute remaining users proportionally to cluster sizes
    if remaining_users > 0:
        total_users = sum(cluster_counts.values())
        for cluster_id in range(num_clusters):
            cluster_users = user_variance[
                (user_variance['cluster'] == cluster_id) & 
                (~user_variance['UserID'].isin(selected_users))
            ]
            
            if len(cluster_users) == 0:
                continue
                
            # Calculate proportional allocation
            cluster_quota = int(remaining_users * (cluster_counts[cluster_id] / total_users))
            n_samples = min(cluster_quota, len(cluster_users))
            
            if n_samples > 0:
                # Sort by variance for stratified sampling
                cluster_users = cluster_users.sort_values('rating_variance')
                
                # Create variance groups
                try:
                    cluster_users['variance_group'] = pd.qcut(
                        cluster_users['rating_variance'],
                        q=num_variance_groups,
                        labels=False,
                        duplicates='drop'
                    )
                except ValueError:
                    cluster_users['variance_group'] = 0
                
                # Sample with normal distribution weighting
                actual_variance_groups = cluster_users['variance_group'].nunique()
                if actual_variance_groups > 1:
                    x = np.linspace(-2, 2, actual_variance_groups)
                    weights = np.exp(-x**2/2)
                    weights = weights + 0.1
                    weights = weights / weights.sum()
                else:
                    weights = np.array([1.0])
                
                samples_per_group = (weights * n_samples).astype(int)
                samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
                
                for group_num in range(actual_variance_groups):
                    group_df = cluster_users[cluster_users['variance_group'] == group_num]
                    n = min(samples_per_group[group_num], len(group_df))
                    selected = group_df.sample(n=n, random_state=random_state)
                    selected_users.extend(selected['UserID'].tolist())
                
                remaining_users -= n_samples
    
    # 8. Final adjustment if we didn't get exactly num_users_to_select due to rounding
    if len(selected_users) < num_users_to_select:
        remaining = num_users_to_select - len(selected_users)
        remaining_candidates = user_variance[~user_variance['UserID'].isin(selected_users)]
        if len(remaining_candidates) > 0:
            additional = remaining_candidates.sample(n=remaining, random_state=random_state)
            selected_users.extend(additional['UserID'].tolist())
    
    if len(selected_users) > num_users_to_select:
        selected_users = selected_users[:num_users_to_select]
    
    # 9. Log the distribution
    selected_clusters = user_variance[
        user_variance['UserID'].isin(selected_users)
    ]['cluster'].value_counts().to_dict()
    
    logging.info(
        f"Selected {len(selected_users)} users from {num_clusters} embedding clusters "
        f"with variance-based sampling. Weight type {weight_type},  Distribution: {selected_clusters}"
    )
    
    return selected_users

def select_users_with_loss_based_embedding(
    dataset: pd.DataFrame,
    embedding_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    num_variance_groups: int = 5,
    min_users_per_cluster: int = 50,
    random_state: int = 42
) -> List[int]:
    # 对kmeans添加一个sample weight, 这个weight是根据loss和评分多样性计算的
    # loss说明这个建模的噪声大小，评分多样性惩罚信息量很低的用户（全是一种给分当然简单了）
    logging.info(f"using loss based embedding, ncluster = {num_clusters}，sample per cluster = auto, no clip or norm，*0.9 + 0.1")
    data = json.load(open('outputs/MoviesAndTV/Llama-3.2-1B-Instruct-202504192336/user_losses.json'))
    norm_lambda = 0.2
    user_dict = {}
    for user_id, losses in data.items():
        ce_losses = losses['ce_loss']
        mse_losses = losses['mse_loss']
        agg_loss = sum(norm_lambda * ce + (1 - norm_lambda) * mse for ce, mse in zip(ce_losses, mse_losses)) / len(ce_losses)
        mse_loss = sum(mse_losses) / len(mse_losses)
        ce_loss = sum(ce_losses) / len(ce_losses)
        user_dict[user_id] = {
            'agg_loss': agg_loss,
            'mse_loss': mse_loss,
            'ce_loss': ce_loss
        }
        
    user_df = pd.DataFrame.from_dict(user_dict, orient='index')
    user_df.reset_index(inplace=True)
    user_df.rename(columns={'index': 'user_id'}, inplace=True)
    
    import numpy as np
    def compute_weights_from_loss(loss, lower=0.01, upper=2.33,median = 0.5,beta=3):
        # Step 1: 截断极端值（防止离群值干扰）
        # loss = np.clip(loss, lower, upper)
        clipped_loss = loss
        # Step 2: 线性归一化到 [0, 1]
        loss = (clipped_loss - lower) / (upper - lower)
        
        # Step 3: 使用 Sigmoid 进行非线性压缩（提升中段区分度）
        weights = 1 - 1 / (1 + np.exp(-beta * (loss - median)))
        
        # 可选：再做一个缩放和平移，让最终输出在 [0.1, 0.9] 更紧凑
        # weights = weights * 0.9 + 0.1
        
        return weights
    # weights_from_loss = compute_weights_from_loss(user_df['agg_loss'].values,median=user_df['agg_loss'].median())
    weights_from_loss = compute_weights_from_loss(user_df['agg_loss'].values,lower=user_df['agg_loss'].quantile(0.1),upper=user_df['agg_loss'].quantile(0.98))
    
    from collections import defaultdict

    # 假设 train_data 已加载且 UserID 已转换为索引
    # train_data 包含列: UserID, ItemID, rating, timestamp, ItemTitle

    # Step 1: 统计每个用户的评分频率
    user_rating_counts = defaultdict(lambda: [0]*5)  # 每个用户有5个评分桶（0~4对应1~5分）

    for _, row in dataset.iterrows():
        user_id = row['UserID']
        rating = int(row['rating'])  # 确保是整数评分（如1、2、3、4、5）
        if 1 <= rating <= 5:
            user_rating_counts[user_id][rating - 1] += 1  # 映射到0-4索引

    # Step 2: 计算每个用户的评分概率分布及香农熵
    def compute_entropy(counts):
        counts = np.array(counts)
        total = counts.sum()
        if total == 0 or np.unique(counts).shape[0] == 1:
            return 0.0  # 防止除零或全0/单一分值的情况
        probs = counts / total
        entropy = -np.sum([p * np.log(p) for p in probs if p > 0])
        return entropy

    user_entropy = {
        user_id: compute_entropy(counts)
        for user_id, counts in user_rating_counts.items()
    }
        
    user_df['entropy'] = user_df['user_id'].astype(int).map(user_entropy)
    # 归一化处理
    min_entropy = user_df['entropy'].min()
    max_entropy = user_df['entropy'].max()
    # 归一化到 [0, 1]
    user_df['entropy'] = (user_df['entropy'] - min_entropy) / (max_entropy - min_entropy)
    
    weight = torch.load(embedding_path,map_location='cpu',weights_only=True)
    user_embedding = weight['user_embedding']['weight']
    
    alpha = 0.1  # 可调参数：alpha越大，越重视评分多样性
    final_weights = (
        alpha * user_df['entropy'].values +
        (1 - alpha) * weights_from_loss
    )   
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(user_embedding, sample_weight=final_weights)
    
    labels = kmeans.labels_
    # 统计每个簇的用户数量
    cluster_counts = np.bincount(labels)
    logging.info(f"香农熵权重 {alpha}, 每个簇的用户数量: {cluster_counts}")
    
    train_indices = []
    
    cluster_count = np.bincount(clusters)
    total_users_in_clusters = sum(cluster_count)
    cluster_proportions = cluster_count / total_users_in_clusters

    # Calculate how many samples we should take from each cluster
    samples_per_cluster = np.round(cluster_proportions * num_users_to_select).astype(int)
    samples_per_cluster = np.where(samples_per_cluster == 0, 1, samples_per_cluster) # 保证每个簇至少有一个样本
    for i in range(num_clusters):
        cluster_indices = np.where(clusters == i)[0]
        n_samples = min(samples_per_cluster[i], len(cluster_indices))
        if n_samples > 0:
            sampled = random.sample(cluster_indices.tolist(), k=int(n_samples))
            train_indices.extend(sampled)
    
    # # 获取每个样本到其簇中心的距离
    # distances = kmeans.transform(user_embedding)
    # # 对于每个样本，获取它到自己簇中心的距离
    # sample_distances = distances[np.arange(len(clusters)), clusters]
    # for cluster in range(num_clusters):
    #     cluster_indices = np.where(clusters == cluster)[0]
    #     if len(cluster_indices) > samples_per_cluster:
    #         # 获取该簇中所有样本的距离
    #         cluster_distances = sample_distances[cluster_indices]
    #         # 按距离排序并选择最近的samples_per_cluster个样本
    #         sorted_indices = np.argsort(cluster_distances)
    #         selected_indices = cluster_indices[sorted_indices[:samples_per_cluster]]
    #     else:
    #         selected_indices = cluster_indices
    #     train_indices.extend(selected_indices.tolist())
        
    logging.info(f"using {len(train_indices)} users for training")
    return train_indices
    
def select_users_with_loss_based_embedding_and_varience_sampling(
    dataset: pd.DataFrame,
    embedding_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    num_variance_groups: int = 5,
    min_users_per_cluster: int = 50,
    random_state: int = 42
) -> List[int]:
    # 对kmeans添加一个sample weight, 这个weight是根据loss和评分多样性计算的
    # loss说明这个建模的噪声大小，评分多样性惩罚信息量很低的用户（全是一种给分当然简单了）
    logging.info("using loss based embedding, sample per cluster = auto, no clip or norm，*0.9 + 0.1 + varience sample")
    data = json.load(open('outputs/MoviesAndTV/Llama-3.2-1B-Instruct-202504192336/user_losses.json'))
    norm_lambda = 0.2
    user_dict = {}
    for user_id, losses in data.items():
        ce_losses = losses['ce_loss']
        mse_losses = losses['mse_loss']
        agg_loss = sum(norm_lambda * ce + (1 - norm_lambda) * mse for ce, mse in zip(ce_losses, mse_losses)) / len(ce_losses)
        mse_loss = sum(mse_losses) / len(mse_losses)
        ce_loss = sum(ce_losses) / len(ce_losses)
        user_dict[user_id] = {
            'agg_loss': agg_loss,
            'mse_loss': mse_loss,
            'ce_loss': ce_loss
        }
        
    user_df = pd.DataFrame.from_dict(user_dict, orient='index')
    user_df.reset_index(inplace=True)
    user_df.rename(columns={'index': 'user_id'}, inplace=True)
    
    import numpy as np
    def compute_weights_from_loss(loss, lower=0.005, upper=2.94,median = 0.5,beta=3):
        # Step 1: 截断极端值（防止离群值干扰）
        # loss = np.clip(loss, lower, upper)
        # clipped_loss = loss
        # Step 2: 线性归一化到 [0, 1]
        # normalized = (clipped_loss - lower) / (upper - lower)
        
        # Step 3: 使用 Sigmoid 进行非线性压缩（提升中段区分度）
        weights = 1 - 1 / (1 + np.exp(-beta * (loss - median)))
        
        # 可选：再做一个缩放和平移，让最终输出在 [0.1, 0.9] 更紧凑
        weights = weights * 0.9 + 0.1
        
        return weights
    weights_from_loss = compute_weights_from_loss(user_df['agg_loss'].values,median=user_df['agg_loss'].median())
    
    from collections import defaultdict

    # 假设 train_data 已加载且 UserID 已转换为索引
    # train_data 包含列: UserID, ItemID, rating, timestamp, ItemTitle

    # Step 1: 统计每个用户的评分频率
    user_rating_counts = defaultdict(lambda: [0]*5)  # 每个用户有5个评分桶（0~4对应1~5分）

    for _, row in dataset.iterrows():
        user_id = row['UserID']
        rating = int(row['rating'])  # 确保是整数评分（如1、2、3、4、5）
        if 1 <= rating <= 5:
            user_rating_counts[user_id][rating - 1] += 1  # 映射到0-4索引

    # Step 2: 计算每个用户的评分概率分布及香农熵
    def compute_entropy(counts):
        counts = np.array(counts)
        total = counts.sum()
        if total == 0 or np.unique(counts).shape[0] == 1:
            return 0.0  # 防止除零或全0/单一分值的情况
        probs = counts / total
        entropy = -np.sum([p * np.log(p) for p in probs if p > 0])
        return entropy

    user_entropy = {
        user_id: compute_entropy(counts)
        for user_id, counts in user_rating_counts.items()
    }
        
    user_df['entropy'] = user_df['user_id'].astype(int).map(user_entropy)
    # 归一化处理
    min_entropy = user_df['entropy'].min()
    max_entropy = user_df['entropy'].max()
    # 归一化到 [0, 1]
    user_df['entropy'] = (user_df['entropy'] - min_entropy) / (max_entropy - min_entropy)
    
    weight = torch.load(embedding_path,map_location='cpu',weights_only=True)
    user_embedding = weight['user_embedding']['weight']
    
    alpha = 0.1  # 可调参数：alpha越大，越重视评分多样性
    final_weights = (
        alpha * user_df['entropy'].values +
        (1 - alpha) * weights_from_loss
    )   
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=15, random_state=random_state)
    clusters = kmeans.fit_predict(user_embedding, sample_weight=final_weights)
    
    labels = kmeans.labels_
    # 统计每个簇的用户数量
    cluster_counts = np.bincount(labels)
    logging.info(f"香农熵权重 {alpha}, 每个簇的用户数量: {cluster_counts}")
    
    
    cluster_count = np.bincount(clusters)
    total_users_in_clusters = sum(cluster_count)
    cluster_proportions = cluster_count / total_users_in_clusters

    user_variance = dataset.groupby('UserID')['rating'].var().reset_index()
    user_variance.columns = ['UserID', 'rating_variance']
    user_variance['cluster'] = clusters
    
    # 4. Calculate actual cluster sizes
    cluster_counts = Counter(clusters)
    
    # 5. Initialize selection
    selected_users = []
    remaining_users = num_users_to_select
    
    # 6. First pass: select minimum users from each cluster
    for cluster_id in range(num_clusters):
        cluster_users = user_variance[user_variance['cluster'] == cluster_id]
        n_samples = min(min_users_per_cluster, len(cluster_users))
        
        if n_samples > 0:
            # Sort by variance to prepare for stratified sampling
            cluster_users = cluster_users.sort_values('rating_variance')
            
            # Create variance groups within the cluster
            try:
                cluster_users['variance_group'] = pd.qcut(
                    cluster_users['rating_variance'],
                    q=num_variance_groups,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                # If not enough unique values, assign all to same group
                cluster_users['variance_group'] = 0
            
            # Calculate actual number of variance groups
            actual_variance_groups = cluster_users['variance_group'].nunique()
            
            # Create normal distribution weights for variance groups
            if actual_variance_groups > 1:
                x = np.linspace(-2, 2, actual_variance_groups)
                weights = np.exp(-x**2/2)  # Gaussian curve
                weights = weights + 0.1  # Add small constant
                weights = weights / weights.sum()  # Normalize
            else:
                weights = np.array([1.0])
            
            # Sample from each variance group with normal distribution weighting
            samples_per_group = (weights * n_samples).astype(int)
            samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
            
            for group_num in range(actual_variance_groups):
                group_df = cluster_users[cluster_users['variance_group'] == group_num]
                n = min(samples_per_group[group_num], len(group_df))
                selected = group_df.sample(n=n, random_state=random_state)
                selected_users.extend(selected['UserID'].tolist())
            
            remaining_users -= n_samples
    
    # 7. Second pass: distribute remaining users proportionally to cluster sizes
    if remaining_users > 0:
        total_users = sum(cluster_counts.values())
        for cluster_id in range(num_clusters):
            cluster_users = user_variance[
                (user_variance['cluster'] == cluster_id) & 
                (~user_variance['UserID'].isin(selected_users))
            ]
            
            if len(cluster_users) == 0:
                continue
                
            # Calculate proportional allocation
            cluster_quota = int(remaining_users * (cluster_counts[cluster_id] / total_users))
            n_samples = min(cluster_quota, len(cluster_users))
            
            if n_samples > 0:
                # Sort by variance for stratified sampling
                cluster_users = cluster_users.sort_values('rating_variance')
                
                # Create variance groups
                try:
                    cluster_users['variance_group'] = pd.qcut(
                        cluster_users['rating_variance'],
                        q=num_variance_groups,
                        labels=False,
                        duplicates='drop'
                    )
                except ValueError:
                    cluster_users['variance_group'] = 0
                
                # Sample with normal distribution weighting
                actual_variance_groups = cluster_users['variance_group'].nunique()
                if actual_variance_groups > 1:
                    x = np.linspace(-2, 2, actual_variance_groups)
                    weights = np.exp(-x**2/2)
                    weights = weights + 0.1
                    weights = weights / weights.sum()
                else:
                    weights = np.array([1.0])
                
                samples_per_group = (weights * n_samples).astype(int)
                samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
                
                for group_num in range(actual_variance_groups):
                    group_df = cluster_users[cluster_users['variance_group'] == group_num]
                    n = min(samples_per_group[group_num], len(group_df))
                    selected = group_df.sample(n=n, random_state=random_state)
                    selected_users.extend(selected['UserID'].tolist())
                
                remaining_users -= n_samples
    
    # 8. Final adjustment if we didn't get exactly num_users_to_select due to rounding
    if len(selected_users) < num_users_to_select:
        remaining = num_users_to_select - len(selected_users)
        remaining_candidates = user_variance[~user_variance['UserID'].isin(selected_users)]
        if len(remaining_candidates) > 0:
            additional = remaining_candidates.sample(n=remaining, random_state=random_state)
            selected_users.extend(additional['UserID'].tolist())
    
    if len(selected_users) > num_users_to_select:
        selected_users = selected_users[:num_users_to_select]
    
    # 9. Log the distribution
    selected_clusters = user_variance[
        user_variance['UserID'].isin(selected_users)
    ]['cluster'].value_counts().to_dict()
        
    logging.info(
        f"Selected {len(selected_users)} users from {num_clusters} embedding clusters "
        f"with variance-based sampling. Distribution: {selected_clusters}"
    )
    return selected_users

from sklearn.metrics.pairwise import pairwise_distances

def farthest_point_sampling(embeddings, num_samples):
    """
    Farthest Point Sampling (FPS) on a set of embeddings.

    Args:
        embeddings (np.ndarray): shape (n_points, dim)
        num_samples (int): number of samples to select

    Returns:
        list: indices of selected points (local to this cluster)
    """
    if num_samples >= len(embeddings):
        return list(range(len(embeddings)))

    # Convert to numpy array
    if not isinstance(embeddings, np.ndarray):
        embeddings = embeddings.detach().cpu().numpy()

    # Initialize with a random point
    indices = [np.random.randint(len(embeddings))]
    dists = pairwise_distances(embeddings, embeddings[indices], metric='euclidean').flatten()

    for _ in range(num_samples - 1):
        farthest_idx = np.argmax(dists)
        indices.append(farthest_idx)
        new_dists = pairwise_distances(embeddings, embeddings[[farthest_idx]], metric='euclidean').flatten()
        dists = np.minimum(dists, new_dists)

    return indices

def select_users_with_fps(
    dataset: pd.DataFrame,
    embedding_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    random_state: int = 42
) -> List[int]:
    """
    Select users using FPS (Farthest Point Sampling) based on user embeddings.
    """
    # Load user embeddings from path
    logging.info(f"using fps, ncluster = {num_clusters}，sample per cluster = auto")
    sp_path = torch.load(embedding_path, map_location='cpu', weights_only=True)
    user_embeddings = sp_path['user_embedding']['weight'].numpy()
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(user_embeddings)
    # 输出聚类结果，每个簇有多少个样本
    labels = kmeans.labels_
    cluster_counts = np.bincount(labels)
    logging.info(f"每个簇的用户数量: {cluster_counts}")
    
    # Assign users to clusters
    cluster_to_indices = defaultdict(list)
    for idx, cluster_id in enumerate(clusters):
        cluster_to_indices[cluster_id].append(idx)

    # Calculate number of samples per cluster (proportional allocation)
    total_users = len(user_embeddings)
    cluster_ids = sorted(cluster_to_indices.keys())
    cluster_sizes = np.array([len(cluster_to_indices[c]) for c in cluster_ids])
    sample_counts = np.round((cluster_sizes / total_users) * num_users_to_select).astype(int)

    # Adjust to ensure total is exactly num_users_to_select
    diff = num_users_to_select - sample_counts.sum()
    if diff > 0:
        # Add remaining to the largest cluster
        largest_idx = np.argmax(cluster_sizes)
        sample_counts[largest_idx] += diff
    elif diff < 0:
        # Subtract from the largest cluster
        largest_idx = np.argmax(cluster_sizes)
        sample_counts[largest_idx] += diff  # diff is negative
        
    selected_indices = []

    for cluster_id, num_samples in zip(cluster_ids, sample_counts):
        indices_in_cluster = cluster_to_indices[cluster_id]
        embeddings_in_cluster = user_embeddings[indices_in_cluster]

        fps_local_indices = farthest_point_sampling(embeddings_in_cluster, num_samples)
        global_indices = [indices_in_cluster[i] for i in fps_local_indices]
        selected_indices.extend(global_indices)

    logging.info(f"using {len(selected_indices)} users for training")
    return selected_indices

def select_users_dropout_user(
    embedding_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 200,
    random_state: int = 42,
    ctype: str = "kmeans",
):
    """
    kmeans聚类后，去除个位数样本的簇，对剩下的所有用户进行随机采样（相当于去除离群样本）
    """
    # 读取embedding
    weight = torch.load(embedding_path,map_location='cpu',weights_only=True)
    user_embedding = weight['user_embedding']['weight']
    
    # 进行kmeans聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(user_embedding)
    # 统计每个簇的用户数量
    cluster_counts = np.bincount(labels)
    logging.info(f"每个簇的用户数量: {cluster_counts}")
    # 选择大于10个用户的簇
    valid_clusters = np.where(cluster_counts > 0)[0]
    # 选择这些簇中的用户
    if ctype == "random":
        all_user = []
        for cluster_id in valid_clusters:
            cluster_indices = np.where(labels == cluster_id)[0]
            all_user.extend(cluster_indices.tolist())
        # 随机采样
        selected_indices = random.sample(all_user, k=num_users_to_select)
    elif ctype == "kmeans":
        # 对剩下的簇按簇大小按比例采样
        selected_indices = []
        total_users = sum(cluster_counts[valid_clusters])
        cluster_proportions = cluster_counts[valid_clusters] / total_users

        # Calculate how many samples to take from each valid cluster
        samples_per_cluster = np.round(cluster_proportions * num_users_to_select).astype(int)
        samples_per_cluster = np.where(samples_per_cluster == 0, 1, samples_per_cluster)  # Ensure at least one sample per cluster

        for cluster_id, n_samples in zip(valid_clusters, samples_per_cluster):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > n_samples:
                sampled = random.sample(cluster_indices.tolist(), k=int(n_samples))
                selected_indices.extend(sampled)
            else:
                selected_indices.extend(cluster_indices.tolist())
    logging.info(f"droping out outlier sample(defused)，then select sample with type = {ctype}")
    return selected_indices

def precompute_ffn_activations(model, prompt_embeddings,tokenizer, use_top_k_layers=3):
    """预计算所有prompt的FFN激活状态"""
    num_prompts = prompt_embeddings.shape[0]
    activation_list = []
    
    # 获取顶层layer的索引
    num_total_layers = len(model.model.layers)
    layer_indices = [-(i + 1) for i in range(use_top_k_layers)]  # 如[-1, -2, -3]
    
    # 注册hook
    captured_activations = {}
    hooks = []
    
    def _get_activation_hook(layer_name):
        def hook(model, input_tensor, output_tensor):
            captured_activations[layer_name] = output_tensor.detach().clone()
        return hook
    
    for layer_idx in layer_indices:
        layer_obj = model.model.layers[layer_idx]
        hook_handle = layer_obj.mlp.act_fn.register_forward_hook(
            _get_activation_hook(f"layer_{layer_idx}_mlp_act_fn_out")
        )
        hooks.append(hook_handle)
    
    # 预计算
    all_activations = []
    with torch.no_grad():
        for i in tqdm(range(num_prompts), desc="Precomputing FFN activations"):
            # 准备输入 (添加BOS)
            bos_token_id = torch.tensor([tokenizer.bos_token_id], dtype=torch.long, device="cuda:2")
            bos_embedding = model.get_input_embeddings()(bos_token_id)  # (1, hidden_dim)
            
            current_prompt = prompt_embeddings[i].unsqueeze(0)  # (1, prompt_length, hidden_dim)
            input_embeds = torch.cat([bos_embedding.unsqueeze(1), current_prompt], dim=1)
            
            # 前向传播
            model(inputs_embeds=input_embeds)
            
            # 收集激活
            layer_activations = []
            for layer_idx in layer_indices:
                key = f"layer_{layer_idx}_mlp_act_fn_out"
                act = captured_activations[key][0, -1, :]  # 取最后一个token位置
                binary_act = (act > 0).float()
                layer_activations.append(binary_act)
            
            # 拼接各层激活
            combined_act = torch.cat(layer_activations)
            all_activations.append(combined_act.cpu())  # 移到CPU节省显存
    
    # 移除hook
    for h in hooks:
        h.remove()
    
    return torch.stack(all_activations)  # (num_prompts, total_activation_dim)

def ON_based_cluster(
    embedding_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    random_state: int = 42,
):
    logging.info(f"using ON based cluster, ncluster = {num_clusters}，sample per cluster = auto")
    # embedding_1b = torch.load(embedding_path, map_location='cpu', weights_only=True)
    # user_embedding = embedding_1b['user_embedding']['weight']
    active_embedding = torch.load(embedding_path, map_location='cpu')
    active_embedding = active_embedding.to(torch.float32)
    
    norms = torch.norm(active_embedding, dim=1, keepdim=True)
    normalized_acts = active_embedding / (norms + 1e-8)
    
    # 进行kmeans聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    labels = kmeans.fit_predict(normalized_acts)
    # 统计每个簇的用户数量
    cluster_counts = np.bincount(labels)
    logging.info(f"每个簇的用户数量: {cluster_counts}")
    
    # 分层采样
    train_indices = []
    total_users_in_clusters = sum(cluster_counts)
    cluster_proportions = cluster_counts / total_users_in_clusters

    # Calculate how many samples we should take from each cluster
    samples_per_cluster = np.round(cluster_proportions * num_users_to_select).astype(int)
    samples_per_cluster = np.where(samples_per_cluster == 0, 1, samples_per_cluster) # 保证每个簇至少有一个样本
    logging.info(f"每个簇的采样数量: {samples_per_cluster}")
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        n_samples = min(samples_per_cluster[i], len(cluster_indices))
        if n_samples > 0:
            sampled = random.sample(cluster_indices.tolist(), k=int(n_samples))
            train_indices.extend(sampled)
    return train_indices

def ON_based_cluster_and_variance_sampling(
    dataset: pd.DataFrame,
    embedding_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    num_variance_groups: int = 5,
    min_users_per_cluster: int = 50,
    random_state: int = 42,
    weight_type: str = "normal"
) -> List[int]:

    # 1. Load user embeddings from path
    active_embedding = torch.load(embedding_path, map_location='cpu')
    active_embedding = active_embedding.to(torch.float32)
    
    norms = torch.norm(active_embedding, dim=1, keepdim=True)
    normalized_acts = active_embedding / (norms + 1e-8)
    
    # 2. Prepare user data - calculate rating variance per user
    user_variance = dataset.groupby('UserID')['rating'].var().reset_index()
    user_variance.columns = ['UserID', 'rating_variance']
    
    # 3. Perform K-means clustering on user embeddings
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(normalized_acts)
    user_variance['cluster'] = clusters
    
    # 4. Calculate actual cluster sizes
    cluster_counts = Counter(clusters)
    
    # 5. Initialize selection
    selected_users = []
    remaining_users = num_users_to_select
    
    # 6. First pass: select minimum users from each cluster
    for cluster_id in range(num_clusters):
        cluster_users = user_variance[user_variance['cluster'] == cluster_id]
        n_samples = min(min_users_per_cluster, len(cluster_users))
        
        if n_samples > 0:
            # Sort by variance to prepare for stratified sampling
            cluster_users = cluster_users.sort_values('rating_variance')
            
            # Create variance groups within the cluster
            try:
                cluster_users['variance_group'] = pd.qcut(
                    cluster_users['rating_variance'],
                    q=num_variance_groups,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                # If not enough unique values, assign all to same group
                cluster_users['variance_group'] = 0
            
            # Calculate actual number of variance groups
            actual_variance_groups = cluster_users['variance_group'].nunique()
            
            # Create normal distribution weights for variance groups
            weights = np.array([1.0])
            if actual_variance_groups > 1:
                if weight_type == "normal":
                    x = np.linspace(-2, 2, actual_variance_groups)
                    weights = np.exp(-x**2/2)  # Gaussian curve
                    weights = weights + 0.1  # Add small constant
                    weights = weights / weights.sum()  # Normalize
                elif weight_type == "linear":
                    weights = np.arange(1, actual_variance_groups + 1)
                    weights = weights / weights.sum()
                elif weight_type == 'uniform':
                    weights = np.ones(actual_variance_groups)
                    weights = weights / weights.sum()

            
            # Sample from each variance group with normal distribution weighting
            samples_per_group = (weights * n_samples).astype(int)
            samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
            
            for group_num in range(actual_variance_groups):
                group_df = cluster_users[cluster_users['variance_group'] == group_num]
                n = min(samples_per_group[group_num], len(group_df))
                selected = group_df.sample(n=n, random_state=random_state)
                selected_users.extend(selected['UserID'].tolist())
            
            remaining_users -= n_samples
    
    # 7. Second pass: distribute remaining users proportionally to cluster sizes
    if remaining_users > 0:
        total_users = sum(cluster_counts.values())
        for cluster_id in range(num_clusters):
            cluster_users = user_variance[
                (user_variance['cluster'] == cluster_id) & 
                (~user_variance['UserID'].isin(selected_users))
            ]
            
            if len(cluster_users) == 0:
                continue
                
            # Calculate proportional allocation
            cluster_quota = int(remaining_users * (cluster_counts[cluster_id] / total_users))
            n_samples = min(cluster_quota, len(cluster_users))
            
            if n_samples > 0:
                # Sort by variance for stratified sampling
                cluster_users = cluster_users.sort_values('rating_variance')
                
                # Create variance groups
                try:
                    cluster_users['variance_group'] = pd.qcut(
                        cluster_users['rating_variance'],
                        q=num_variance_groups,
                        labels=False,
                        duplicates='drop'
                    )
                except ValueError:
                    cluster_users['variance_group'] = 0
                
                # Sample with normal distribution weighting
                actual_variance_groups = cluster_users['variance_group'].nunique()
                if actual_variance_groups > 1:
                    x = np.linspace(-2, 2, actual_variance_groups)
                    weights = np.exp(-x**2/2)
                    weights = weights + 0.1
                    weights = weights / weights.sum()
                else:
                    weights = np.array([1.0])
                
                samples_per_group = (weights * n_samples).astype(int)
                samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
                
                for group_num in range(actual_variance_groups):
                    group_df = cluster_users[cluster_users['variance_group'] == group_num]
                    n = min(samples_per_group[group_num], len(group_df))
                    selected = group_df.sample(n=n, random_state=random_state)
                    selected_users.extend(selected['UserID'].tolist())
                
                remaining_users -= n_samples
    
    # 8. Final adjustment if we didn't get exactly num_users_to_select due to rounding
    if len(selected_users) < num_users_to_select:
        remaining = num_users_to_select - len(selected_users)
        remaining_candidates = user_variance[~user_variance['UserID'].isin(selected_users)]
        if len(remaining_candidates) > 0:
            additional = remaining_candidates.sample(n=remaining, random_state=random_state)
            selected_users.extend(additional['UserID'].tolist())
    
    if len(selected_users) > num_users_to_select:
        selected_users = selected_users[:num_users_to_select]
    
    # 9. Log the distribution
    selected_clusters = user_variance[
        user_variance['UserID'].isin(selected_users)
    ]['cluster'].value_counts().to_dict()
    
    logging.info(
        f"Selected {len(selected_users)} users from {num_clusters} ON based clusters "
        f"with variance-based sampling. Weight type {weight_type},  Distribution: {selected_clusters}"
    )
    
    return selected_users

def ON_based_cluster_and_loss_sampling(
    embedding_path: str,
    loss_file_path: str,
    num_users_to_select: int = 2000,
    num_clusters: int = 15,
    num_loss_groups: int = 5,
    min_users_per_cluster: int = 50,
    random_state: int = 42,
    weight_type: str = "normal"
) -> List[int]:
    """
    Select users by clustering them based on embeddings and sampling based on loss values
    within each cluster using normal distribution.
    
    Args:
        embedding_path: Path to the file containing user embeddings and loss data
        loss_file_path: Path to the JSON file containing user loss data
        num_users_to_select: Total number of users to select (default: 6000)
        num_clusters: Number of clusters for K-means on embeddings (default: 30)
        num_loss_groups: Number of loss groups within each cluster (default: 5)
        min_users_per_cluster: Minimum number of users to select from each cluster (default: 50)
        random_state: Random seed for reproducibility (default: 42)
    
    Returns:
        List of selected user IDs
    """
    # 1. Load user embeddings from path
    active_embedding = torch.load(embedding_path, map_location='cpu')
    active_embedding = active_embedding.to(torch.float32)
    
    norms = torch.norm(active_embedding, dim=1, keepdim=True)
    normalized_acts = active_embedding / (norms + 1e-8)
    
    # Load loss data
    with open(loss_file_path, 'r') as f:
        data = json.load(f)
    
    # 2. Calculate average MSE loss per user
    if 'mse_loss' in data['0'].keys():
        mse_loss_dict = {
            int(user): sum(user_data['mse_loss'])/len(user_data['mse_loss']) 
            for user, user_data in data.items()
        }
        users_df = pd.DataFrame.from_dict(mse_loss_dict, orient='index', columns=['loss'])
    else:
        ce_loss_dict = {
            int(user): sum(user_data['ce_loss'])/len(user_data['ce_loss']) 
            for user, user_data in data.items()
        }
        users_df = pd.DataFrame.from_dict(ce_loss_dict, orient='index', columns=['loss'])
    
    # 3. Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
    users_df['cluster'] = kmeans.fit_predict(normalized_acts)
    
    # 4. Calculate actual cluster sizes
    cluster_counts = Counter(users_df['cluster'])
    
    # 5. Initialize selection
    selected_users = []
    remaining_users = num_users_to_select
    
    # 6. First pass: select minimum users from each cluster
    for cluster_id in range(num_clusters):
        cluster_users = users_df[users_df['cluster'] == cluster_id]
        n_samples = min(min_users_per_cluster, len(cluster_users))
        
        if n_samples > 0:
            # Sort by loss to prepare for stratified sampling
            cluster_users = cluster_users.sort_values('loss')
            
            # Create loss groups within the cluster
            try:
                cluster_users['loss_group'] = pd.qcut(
                    cluster_users['loss'],
                    q=num_loss_groups,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError:
                # If not enough unique values, assign all to same group
                cluster_users['loss_group'] = 0
            
            # Calculate actual number of loss groups
            actual_loss_groups = cluster_users['loss_group'].nunique()
            
            # Create normal distribution weights for loss groups
            weights = np.array([1.0])
            if actual_loss_groups > 1:
                if weight_type == "normal":
                    x = np.linspace(-2, 2, actual_loss_groups)
                    weights = np.exp(-x**2/2)  # Gaussian curve
                    weights = weights + 0.1  # Add small constant
                    weights = weights / weights.sum()  # Normalize
                elif weight_type == "linear":
                    weights = np.arange(1, actual_loss_groups + 1)
                    weights = weights / weights.sum()
                elif weight_type == 'uniform':
                    weights = np.ones(actual_loss_groups)
                    weights = weights / weights.sum()
                
            
            # Sample from each loss group with normal distribution weighting
            samples_per_group = (weights * n_samples).astype(int)
            samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()  # Adjust for rounding
            
            for group_num in range(actual_loss_groups):
                group_df = cluster_users[cluster_users['loss_group'] == group_num]
                n = min(samples_per_group[group_num], len(group_df))
                selected = group_df.sample(n=n, random_state=random_state)
                selected_users.extend(selected.index.tolist())
            
            remaining_users -= n_samples
    
    # 7. Second pass: distribute remaining users proportionally to cluster sizes
    if remaining_users > 0:
        total_users = sum(cluster_counts.values())
        for cluster_id in range(num_clusters):
            cluster_users = users_df[
                (users_df['cluster'] == cluster_id) & 
                (~users_df.index.isin(selected_users))
            ]
            
            if len(cluster_users) == 0:
                continue
                
            # Calculate proportional allocation
            cluster_quota = int(remaining_users * (cluster_counts[cluster_id] / total_users))
            n_samples = min(cluster_quota, len(cluster_users))
            
            if n_samples > 0:
                # Sort by loss for stratified sampling
                cluster_users = cluster_users.sort_values('loss')
                
                # Create loss groups
                try:
                    cluster_users['loss_group'] = pd.qcut(
                        cluster_users['loss'],
                        q=num_loss_groups,
                        labels=False,
                        duplicates='drop'
                    )
                except ValueError:
                    cluster_users['loss_group'] = 0
                
                # Sample with normal distribution weighting
                actual_loss_groups = cluster_users['loss_group'].nunique()
                if actual_loss_groups > 1:
                    x = np.linspace(-2, 2, actual_loss_groups)
                    weights = np.exp(-x**2/2)
                    weights = weights + 0.1
                    weights = weights / weights.sum()
                else:
                    weights = np.array([1.0])
                
                samples_per_group = (weights * n_samples).astype(int)
                samples_per_group[-1] = n_samples - samples_per_group[:-1].sum()
                
                for group_num in range(actual_loss_groups):
                    group_df = cluster_users[cluster_users['loss_group'] == group_num]
                    n = min(samples_per_group[group_num], len(group_df))
                    selected = group_df.sample(n=n, random_state=random_state)
                    selected_users.extend(selected.index.tolist())
                
                remaining_users -= n_samples
    
    # 8. Final adjustment if we didn't get exactly num_users_to_select due to rounding
    if len(selected_users) < num_users_to_select:
        remaining = num_users_to_select - len(selected_users)
        remaining_candidates = users_df[~users_df.index.isin(selected_users)]
        if len(remaining_candidates) > 0:
            additional = remaining_candidates.sample(n=remaining, random_state=random_state)
            selected_users.extend(additional.index.tolist())
    
    if len(selected_users) > num_users_to_select:
        selected_users = selected_users[:num_users_to_select]
    
    # 9. Log the distribution
    selected_clusters = users_df[
        users_df.index.isin(selected_users)
    ]['cluster'].value_counts().to_dict()
    
    logging.info(
        f"Selected {len(selected_users)} users from {num_clusters} ON-based clusters "
        f"with loss-based normal sampling. Distribution: {selected_clusters}"
    )
    
    return selected_users
