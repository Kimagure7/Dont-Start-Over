# Baseline 复现指南

## 概览

Baseline 实验分为两组：

- **Group A**：主效果 baseline（RQ1，Table 2）——对比 PUMA 与全量重训练及退化情形
- **Group B**：用户选择策略 baseline（RQ2，Table 4）——对比不同用户采样策略对 adapter 训练的影响

所有实验均假设 Stage 1（source soft prompt 训练）已完成，最优检查点 `checkpoint_model_best.pth` 已保存。

---

## Group A — 主效果 Baseline（RQ1，Table 2）

### A1. Source Model Performance

直接在 source LLM 上评估 source soft prompt，不做任何迁移，无需训练。

```bash
deepspeed --num_gpus 4 train_sp.py \
    --cfg-path configs/llama3_1b_sp_amazon.yaml \
    --options run.evaluate=True
```

### A2. Random Initialization

在 target LLM 上运行 adapter 训练流程，但用户 prompt 随机初始化（不加载 source soft prompt）。将 `model.soft_prompt_path` 设为 `null`。

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options model.soft_prompt_path=null
```

### A3. Full Retraining

在 **target LLM** 上对所有用户从头训练 soft prompt，作为性能上界。

```bash
deepspeed --num_gpus 4 train_sp.py \
    --cfg-path configs/llama3_1b_sp_amazon.yaml \
    --options model.path=Llama-3.2-3B-Instruct
```

> 将 `Llama-3.2-3B-Instruct` 替换为 target LLM 的本地路径或 HuggingFace 模型名。

### A4. PUMA（全量用户）

使用全部用户训练 adapter（不做采样）。各数据集用户总数如下：

| 数据集 | 用户总数 |
|---|---|
| Amazon Movies & TV | 30,287 |
| MIND | 50,000 |
| Yelp | 32,850 |

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.train_ratio=30287 dataset.mode=0
```

---

## Group B — 用户选择策略 Baseline（RQ2，Table 4）

所有方法均使用 `train_ad.py`，仅 `dataset.mode` 和 `dataset.train_ratio` 不同。

**训练预算：**

| 数据集 | 标准预算 | 大预算（仅 B2） |
|---|---|---|
| Amazon Movies & TV | 2,000 | 6,000 |
| MIND | 1,500 | 6,000 |
| Yelp | 2,000 | 6,000 |

**mode 对照表：**

| Mode | 论文名称 | 备注 |
|---|---|---|
| `0` | Random | 用户已预先随机乱序，取前 N 个即为随机采样 |
| `1` | Variance Bucketing | 按用户评分方差分桶 + Gaussian 权重采样 |
| `2` | KMeans Stratified | prompt embedding 聚类 + uniform 方差分层采样 |
| `3` | KMeans + Variance Stratification（**PUMA**） | prompt embedding 聚类 + Gaussian 方差分层采样 |
| `4` | K-Means on FFN Activ. | 需要提前生成 `model.ffn_matrix` |
| `5` | K-Means on FFN + Loss | 需要 `model.ffn_matrix` 和 `model.loss_file` |
| `6` | K-Means on FFN + Variance | 需要 `model.ffn_matrix` |
| `7` | Loss Bucketing | 需要 `model.loss_file` |
| `8` | KMeans + Loss Stratification | 需要 `model.loss_file` |
| `9` | KMeans + FPS | 在每个 cluster 内用最远点采样保证多样性 |
| `10` | KMeans with PCA | 先将 prompt embedding PCA 降维至 256 维再聚类 |

### B1. Random

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=0 dataset.train_ratio=2000
```

### B2. Random（6k）

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=0 dataset.train_ratio=6000
```

### B3. Variance Bucketing

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=1 dataset.train_ratio=2000
```

### B4. Loss Bucketing

需要 Stage 1 产生的 `user_losses.json`，该文件在 Stage 1 输出目录下自动生成。

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=7 dataset.train_ratio=2000 \
        model.loss_file=outputs/MoviesAndTV/Llama-3.2-1B-Instruct-<timestamp>/user_losses.json
```

### B5. KMeans Stratified

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=2 dataset.train_ratio=2000
```

### B6. KMeans with PCA

先将 prompt embedding PCA 降维至 256 维，再做 KMeans 聚类采样。

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=10 dataset.train_ratio=2000
```

### B7. KMeans + FPS

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=9 dataset.train_ratio=2000
```

### B8. KMeans + Loss Stratification

需要 `user_losses.json`。

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=8 dataset.train_ratio=2000 \
        model.loss_file=outputs/MoviesAndTV/Llama-3.2-1B-Instruct-<timestamp>/user_losses.json
```

### B9. KMeans + Variance Stratification（PUMA）

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=3 dataset.train_ratio=2000
```

### B10 / B11 / B12 — 基于 FFN Activation 的方法

三种方法（mode 4、5、6）均需要预先计算 FFN activation matrix。

#### 第一步：生成 FFN activation matrix

```bash
python generate_ffn_matrix.py \
    --model_path Llama-3.2-1B-Instruct \
    --soft_prompt_path outputs/MoviesAndTV/Llama-3.2-1B-Instruct-<timestamp>/checkpoint_model_best.pth \
    --output_path datasets/MoviesAndTV/activation_matrix.pt \
    --top_k_layers 3 \
    --device cuda:0
```

**脚本执行逻辑：**
1. 加载 source LLM 和已训练的 source soft prompts
2. 对每个用户的 prompt 构造输入 `[BOS_embed, prompt_embed]`，做一次前向传播
3. 通过 forward hook 捕获最后 3 层 transformer 的 `act_fn`（SiLU gate）输出
4. 取最后一个 token 位置的激活值，二值化（`> 0` → 1，否则 0）
5. 三层拼接 → 每用户一个向量，形状为 `(num_users, 3 × intermediate_size)`
6. 保存为 int8 张量至 `activation_matrix.pt`

#### 第二步：运行对应 baseline

**B10. K-Means on FFN Activ.（mode 4）**

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=4 dataset.train_ratio=2000 \
        model.ffn_matrix=datasets/MoviesAndTV/activation_matrix.pt
```

**B11. K-Means on FFN + Loss Stratification（mode 5）**

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=5 dataset.train_ratio=2000 \
        model.ffn_matrix=datasets/MoviesAndTV/activation_matrix.pt \
        model.loss_file=outputs/MoviesAndTV/Llama-3.2-1B-Instruct-<timestamp>/user_losses.json
```

**B12. K-Means on FFN + Variance Stratification（mode 6）**

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml \
    --options dataset.mode=6 dataset.train_ratio=2000 \
        model.ffn_matrix=datasets/MoviesAndTV/activation_matrix.pt
```

---

## 注意事项

- 将 `configs/ad_llama3_amazon.yaml` 替换为 MIND 或 Yelp 对应的 config 文件。
- 将 `<timestamp>` 替换为实际的 Stage 1 输出目录时间戳后缀。
- Group B 所有方法的 `model.soft_prompt_path` 均需指向 source model 的检查点。
- `user_losses.json` 在 Stage 1 训练过程中自动生成，位于对应输出目录下。
