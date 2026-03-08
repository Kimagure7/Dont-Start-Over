# Don’t Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs

> The official implementation of the paper **"Don’t Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs"**.

## Overview

This repository provides the experimental code for a two-stage framework that enables **cross-model transfer of personalized user representations** (soft prompts) between Large Language Models (LLMs) for recommendation tasks.

The core idea is: when switching to a new LLM for recommendation, **you don't need to retrain user representations from scratch**. Instead, a lightweight adapter learns to migrate user soft prompts trained on a source LLM into the embedding space of a target LLM, significantly reducing training cost and preserving user knowledge.

### Key Features

- **Soft Prompt Training**: Trains personalized user embeddings (soft prompts) in an LLM's embedding space for rating prediction or yes/no recommendation tasks.
- **Cross-LLM Adapter**: A lightweight adapter model that migrates soft prompts from a source LLM to a target LLM without user interaction data.
- **Multi-dataset Support**: Amazon Movies & TV, MIND (news recommendation), and Yelp (restaurant reviews).
- **Flexible User Selection**: Multiple strategies for selecting training users, including random sampling, stratified variance sampling, and KMeans-based clustering.
- **Distributed Training**: Built on [DeepSpeed](https://github.com/microsoft/DeepSpeed) for efficient multi-GPU training.

---

## Environment Setup

```bash
conda create -n dso python=3.12
pip install torch "transformers==4.53.0" "deepspeed==0.17.1" omegaconf scikit-learn pandas numpy scipy tqdm matplotlib
```

---

## Project Structure

```
Dont-Start-Over/
├── train_sp.py                       # Stage 1: Train user soft prompts on a source LLM
├── train_ad.py                       # Stage 2: Train adapter for cross-LLM prompt migration
├── train_paad.py                     # Stage 2 (variant): Parallel-source adapter (RQ4)
├── configs/
│   ├── llama3_1b_sp_amazon.yaml      # Soft prompt config: Amazon
│   ├── llama3_1b_sp_mind.yaml        # Soft prompt config: MIND
│   ├── llama3_1b_sp_yelp.yaml        # Soft prompt config: Yelp
│   ├── ad_llama3_amazon.yaml         # Adapter config: Amazon
│   ├── ad_llama3_mind.yaml           # Adapter config: MIND
│   └── ad_llama3_yelp.yaml           # Adapter config: Yelp
├── build_dataset/
│   ├── build_dataset_amazon.py       # Dataset preprocessing for Amazon Movies & TV
│   ├── build_dataset_mind.py         # Dataset preprocessing for MIND
│   └── build_dataset_yelp.py         # Dataset preprocessing for Yelp
└── utils/
    ├── datasets.py                   # Dataset builders and loaders
    ├── model.py                      # Model definitions (RecModelRP, AdapterModelRP, …)
    ├── runner.py                     # Training and evaluation runners
    ├── metrics.py                    # Evaluation metrics (RMSE, MAE, uAUC, HSIC)
    ├── user_select.py                # User selection strategies for adapter training
    ├── log.py                        # Logging utilities
    └── utils.py                      # DeepSpeed configuration helpers
```

---

## Datasets

### 1. Download Raw Data

#### Amazon Movies & TV

Download from [Amazon Review Data 2023](https://amazon-reviews-2023.github.io/):

- Interactions: `Movies_and_TV.jsonl.gz`
- Metadata: `meta_Movies_and_TV.jsonl.gz`

Place both files under `data/amazon_mt_2023/`:
```
data/amazon_mt_2023/Movies_and_TV.jsonl.gz
data/amazon_mt_2023/meta_Movies_and_TV.jsonl.gz
```

#### MIND (Microsoft News)

Download [MIND-Large](https://msnews.github.io/) (`MINDlarge_train.zip` and `MINDlarge_VALID.zip`) and extract:
```
data/mind/train/behaviors.tsv
data/mind/train/news.tsv
data/mind/valid/behaviors.tsv
data/mind/valid/news.tsv
```

#### Yelp

Download the [Yelp Open Dataset](https://www.yelp.com/dataset) (`yelp_dataset.tar`) and extract the two JSON files:
```
data/yelp/yelp_academic_dataset_business.json
data/yelp/yelp_academic_dataset_review.json
```

### 2. Build Processed Datasets

Run the following scripts **from the project root**. Each script reads from `data/` and writes to `datasets/`:

```bash
python build_dataset/build_dataset_amazon.py
python build_dataset/build_dataset_mind.py
python build_dataset/build_dataset_yelp.py
```

After preprocessing, the directory layout should look like:

```
datasets/
├── MoviesAndTV/
│   ├── train_data_30k.csv
│   ├── valid_data_30k.csv
│   └── user_dict_30k.pickle
├── MIND/
│   ├── train_sampled_50k.tsv
│   ├── valid_sampled_50k.tsv
│   └── user_dict_mind.pickle
└── Yelp/
    ├── train_reviews_32k.csv
    ├── valid_reviews_32k.csv
    └── user_dict.pickle
```

These paths are already set in the corresponding config files under `configs/`.



## Models

The framework supports the following LLMs (local paths or HuggingFace model IDs):

- `Llama-3.2-1B-Instruct`
- `Llama-3.2-3B-Instruct`
- `Phi-3-mini-4k-instruct`
- `Qwen2.5-3B-Instruct`
- `stablelm-2-1_6b-chat`
- `gemma` series

Set the `model.path` field in the config file to the local directory or HuggingFace model name.

---

## Training

### Stage 1 — Train User Soft Prompts (Source LLM)

Train personalized user embeddings on a source LLM:

```bash
deepspeed --num_gpus 4 train_sp.py \
    --cfg-path configs/llama3_1b_sp_amazon.yaml
```

Checkpoints are saved to `outputs/<DatasetName>/<ModelName>-<Timestamp>/`.

The best checkpoint (`checkpoint_model_best.pth`) is used as input to Stage 2.

**Key config options (`configs/llama3_1b_sp_amazon.yaml`):**

| Field | Description |
|---|---|
| `model.path` | Path/name of the source LLM |
| `model.prompt_path` | Path to the prompt template file |
| `dataset.name` | Dataset name (`MoviesAndTV`, `MIND`, `Yelp`) |
| `run.max_epoch` | Number of training epochs |
| `run.init_lr` | Initial learning rate |
| `run.norm_lambda` | Weight balancing CE loss and MSE loss (λ) |
| `run.evaluate` | Set to `True` for evaluation-only mode |

---

### Stage 2 — Train Cross-LLM Adapter

Migrate soft prompts from the source LLM to a target LLM using a lightweight adapter:

```bash
deepspeed --num_gpus 4 train_ad.py \
    --cfg-path configs/ad_llama3_amazon.yaml
```

**Key config options (`configs/ad_llama3_amazon.yaml`):**

| Field | Description |
|---|---|
| `model.path` | Path/name of the **target** LLM |
| `model.soft_prompt_path` | Path to the best checkpoint from Stage 1 (source soft prompts) |
| `dataset.train_ratio` | Number of users used for adapter training |
| `dataset.mode` | User selection strategy (see below) |
| `run.norm_lambda` | Weight balancing CE loss and MSE loss (λ) |

---

### Stage 2 (Variant) — Parallel source Adapter Training

```bash
deepspeed --num_gpus 4 train_paad.py \
    --cfg-path configs/ad_llama3_amazon.yaml
```

---

### Overriding Config via Command Line

Any config field can be overridden at runtime using `--options`:

```bash
deepspeed train_sp.py --cfg-path configs/llama3_1b_sp_amazon.yaml \
    --options run.init_lr=1e-3 run.max_epoch=20
```

---

## User Selection Strategies

Controlled by `dataset.mode` in the adapter config:

| Mode | Strategy |
|---|---|
| `0` | Random selection (sequential, first N users) |
| `1` | Stratified variance sampling by rating variance |
| `2` | KMeans clustering + variance sampling (uniform weights) |
| `3` | KMeans clustering + variance sampling (Gaussian weights) |
| `4` | ON-based clustering from FFN activation matrix |
| `5` | ON-based clustering + loss-aware sampling |
| `6` | ON-based clustering + variance sampling |

---

## Evaluation Metrics

| Metric | Task |
|---|---|
| RMSE | Rating prediction |
| MAE | Rating prediction |
| Accuracy | Rating prediction |
| uAUC | Click prediction (MIND) |
| HSIC | Statistical independence test for representation analysis |

Evaluation is performed on the validation set after each epoch. The best model is saved based on the primary metric.

---

## Configuration Reference

### Soft Prompt Config (`configs/llama3_1b_sp_*.yaml`)

```yaml
model:
  path: Llama-3.2-3B-Instruct     # LLM path or HuggingFace model ID
  prompt_path: prompts/amazon_title.txt
  max_txt_len: 256
  use_item_embedding: False        # True: learnable item embedding; False: use item title text

dataset:
  name: MoviesAndTV
  train: datasets/MoviesAndTV/train_data_30k.csv
  valid: datasets/MoviesAndTV/valid_data_30k.csv
  user_dict: datasets/MoviesAndTV/user_dict_30k.pickle

run:
  seed: 42
  evaluate: False
  output_dir: outputs
  init_lr: 5e-4
  per_device_train_batch_size: 42
  max_epoch: 15
  norm_lambda: 0.2
  lr_scheduler: cosine
  gradient_accumulation_steps: 1
```

### Adapter Config (`configs/ad_llama3_*.yaml`)

```yaml
model:
  path: Phi-3-mini-4k-instruct                        # Target LLM
  soft_prompt_path: outputs/MoviesAndTV/.../checkpoint_model_best.pth  # Source soft prompts
  dropout: 0.1
  freeze_predictor: false

dataset:
  train_ratio: 6000   # Number of users for adapter training
  mode: 0             # User selection strategy

run:
  init_lr: 1e-4
  max_epoch: 4
  norm_lambda: 0.2
```

---

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{zhao2026dontstartovercosteffective,
      title={Don't Start Over: A Cost-Effective Framework for Migrating Personalized Prompts Between LLMs}, 
      author={Ziyi Zhao and Chongming Gao and Yang Zhang and Haoyan Liu and Weinan Gan and Huifeng Guo and Yong Liu and Fuli Feng},
      year={2026},
      eprint={2601.12034},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.12034}, 
}
```

---

## License

This project is released for research purposes. Please refer to the licenses of the respective LLMs and datasets you use.

