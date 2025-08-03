import torch.nn as nn
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM
import numpy as np
import torch.nn.functional as F
import contextlib
import logging
from utils.datasets import SoftPromptDataset
from abc import ABC, abstractmethod
from typing import Dict, Type

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.float16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(dtype=dtype, device_type='cuda')
        else:
            return contextlib.nullcontext()

    @classmethod
    def from_config(cls, config):
        pass

    def show_n_params(self, return_str=True):
        tot = 0
        for p in self.parameters():
            w = 1
            for x in p.shape:
                w *= x
            tot += w
        if return_str:
            if tot >= 1e6:
                return "{:.1f}M".format(tot / 1e6)
            else:
                return "{:.1f}K".format(tot / 1e3)
        else:
            return tot

    def _reload_best_model(self, model: nn.Module, ckpt_path):
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(
                ckpt_path, map_location="cpu", weights_only=True)
            # Load user and item embeddings
            model.user_embedding.load_state_dict(checkpoint['user_embedding'])
            try:
                model.item_embedding.load_state_dict(
                    checkpoint['item_embedding'])
            except KeyError:
                pass
            # Load rating predictor if it exists
            if model.rating_predictor is not None and 'rating_predictor' in checkpoint:
                model.rating_predictor.load_state_dict(
                    checkpoint['rating_predictor'])
            logging.info(
                "Loaded best checkpoint from checkpoint: %s", ckpt_path)
        else:
            logging.warning(
                "Best model checkpoint path is invalid: %s", ckpt_path)
        return model

class ModelRegistry:
    _models: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str):
        def wrapper(model_cls: Type[BaseModel]):
            if name in cls._models:
                raise ValueError(f"Model '{name}' is already registered!")
            cls._models[name] = model_cls
            return model_cls
        return wrapper

    @classmethod
    def get_model(cls, name: str) -> Type[BaseModel]:
        dict = {
            "MoviesAndTV": "RecModelRP",
            "MIND": "RecModelYN",
            "Yelp": "RecModelRP",
        }
        model_name = dict.get(name, name)
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' is not registered!")
        return cls._models[model_name]

    @classmethod
    def create(cls, name: str, config: dict, **kwargs):
        model_cls = cls.get_model(name)
        
        if hasattr(model_cls, 'from_config'):
            return model_cls.from_config(config, **kwargs)
        
        return model_cls(config=config, **kwargs) 
    
    @classmethod
    def get_model_ad(cls, name: str) -> Type[BaseModel]:
        dict = {
            "MoviesAndTV": "AdapterModelRP",
            "MIND": "AdapterModelYN",
            "Yelp": "AdapterModelRP",
        }
        model_name = dict.get(name, name)
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' is not registered!")
        return cls._models[model_name]
    
    @classmethod
    def get_model_test(cls, name: str) -> Type[BaseModel]:
        dict = {
            "MoviesAndTV": "AdapterModelRP_PA1",
        }
        model_name = dict.get(name, name)
        if model_name not in cls._models:
            raise ValueError(f"Model '{model_name}' is not registered!")
        return cls._models[model_name]
    
    @classmethod
    def create_ad(cls, name: str, config: dict, **kwargs):
        model_cls = cls.get_model_ad(name)
        
        if hasattr(model_cls, 'from_config'):
            return model_cls.from_config(config, **kwargs)
        
        return model_cls(config=config, **kwargs)
    
    @classmethod
    def create_test(cls, name: str, config: dict, **kwargs):
        """支持自由kwargs传递到具体Model类"""
        model_cls = cls.get_model_test(name)
    
        if hasattr(model_cls, 'from_config'):
            return model_cls.from_config(config, **kwargs)
        
        return model_cls(config=config, **kwargs) 
    

class RatingPredictor(nn.Module):
    def __init__(self):
        super(RatingPredictor, self).__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 12)
        self.fc3 = nn.Linear(12, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LLMModel(BaseModel):
    def __init__(
        self,
        llm_model="",
        prompt_path="",
        nuser=0,
        nitem=0,
        max_txt_len=128,
        use_item_embedding=False,
    ):
        super().__init__()
        logging.info("runing RecModel")

        self.max_txt_len = max_txt_len
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        if 'Qwen' in llm_model and self.llm_tokenizer.bos_token is None:
            self.llm_tokenizer.bos_token = "<|im_start|>"
        elif self.llm_tokenizer.bos_token is None:
            logging.warning("No bos token in tokenizer")
            
        if "gemma" in llm_model.lower():
            self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16,attn_implementation='eager')
        else:
            self.llm_model: LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.float16)
        for name, param in self.llm_model.named_parameters():
            param.requires_grad = False
            
        def need_to_add_unk(tokenizer):
            if tokenizer.unk_token is None:
                return True
            unk_token = tokenizer.unk_token
            duplicate_tokens = []
            for token_name, token_value in tokenizer.special_tokens_map.items():
                if token_value == unk_token and token_name != 'unk_token':
                    duplicate_tokens.append(token_name)
            if len(duplicate_tokens) > 0:
                logging.info(f"unk_token: {unk_token}, Special tokens that duplicate unk_token: {duplicate_tokens}")
                return True
            
        if need_to_add_unk(self.llm_tokenizer):
            self.llm_tokenizer.add_special_tokens(
                {'unk_token': '<unk>'})  # work for most models
            # IMPACTANT, otherwise will raise a cuda error when embedding
            self.llm_model.resize_token_embeddings(
                len(self.llm_tokenizer), mean_resizing=False)
        logging.info('Loading LLM Done')

        embedding_size = self.llm_model.config.hidden_size
        self.use_item_embedding = use_item_embedding
        self._init_soft_prompt(nuser, nitem, embedding_size)
        logging.info("Soft prompt set up already, user embedding size: %d, item embedding size: %d, embedding size: %d" % (
            nuser, nitem, embedding_size))

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            self.prompt = raw_prompts[0]
            logging.info("Prompt loaded from %s: %s" %
                         (prompt_path, self.prompt))
        else:
            self.prompt = "<UserID> <ItemID>"
            logging.warning("No prompt given")
        self.prompt_log_once = False

    def _init_soft_prompt(self, nuser, nitem, embedding_szie, init_range=0.1):
        self.user_embedding = nn.Embedding(nuser, embedding_szie)
        self.user_embedding.weight.data.uniform_(-init_range, init_range)
        if self.use_item_embedding:
            self.item_embedding = nn.Embedding(nitem, embedding_szie)
            self.item_embedding.weight.data.uniform_(-init_range, init_range)
            
    def prompt_based_encode(self, prompt, samples):
        samples_encode = self.encode_recdata(samples)
        sample_embeds, atts_sample = self.recprompt_wrap(
            samples_encode, samples, prompt)
        return sample_embeds, atts_sample

    def encode_recdata(self, samples):
        """
        Encodes recommendation data samples into user and (optionally) item embeddings.

        This method takes a dictionary of sample data, extracts the user and item IDs,
        and converts them into corresponding embeddings using the model's embedding layers.
        If the model is operating on a non-CPU device, the embeddings are automatically
        converted to half precision.

        Parameters:
            samples (dict): A dictionary containing at least the following keys:
                - "UserID": Tensor with shape (batch_size, ...) representing user identifiers.
                - "ItemID": Tensor with shape (batch_size, ...) representing item identifiers
                            (required only if 'use_item_embedding' is True).

        Returns:
            dict: A dictionary with the following keys:
                - "User_emb": Tensor of shape (batch_size, -1, hidden_size), the encoded user embeddings.
                - "Item_emb": Tensor of shape (batch_size, -1, hidden_size) if item embeddings are used,
                              otherwise None.
        """
        # with self.maybe_autocast():
        batch_size = samples['UserID'].shape[0]
        hidden_size = self.llm_model.config.hidden_size
        user_embeds = self.user_embedding(samples['UserID']).reshape(
            # (batch_size, 1, embed_size) 相当于unsqueeze(1)
            batch_size, -1, hidden_size)

        if self.use_item_embedding:
            item_embeds = self.item_embedding(samples['ItemID']).reshape(
                batch_size, -1, hidden_size)  # (batch_size, 1, embed_size)
        else:
            item_embeds = None

        if self.device != torch.device("cpu"):
            user_embeds = user_embeds.half()
            if self.use_item_embedding:
                item_embeds = item_embeds.half()
        sample_embeds_llm = {
            "User_emb": user_embeds,
            "Item_emb": item_embeds
        }
        return sample_embeds_llm
    
    def recprompt_wrap(self, samples, ori_samples, prompt):
        """
        Wrap the input prompt by inserting tokens and replacing placeholders with corresponding embeddings.
        This method prepares the prompt for the language model by:
        - Prepending the beginning-of-sentence token if required.
        - Replacing placeholder strings (e.g., "<UserID>", "<ItemID>", "<ItemTitle>", "<ItemTitleList>") with a designated unknown token.
        - For each sample in the batch, inserting the correct item title and, if available, a list of historical interaction titles.
        - Tokenizing the modified prompts to generate input IDs and corresponding attention masks.
        - Retrieving the embedding vectors for these tokenized inputs and substituting the embedding vectors at placeholder positions with appropriate user and item embeddings extracted from the provided samples.
        - Optionally, logging an example prompt during the first call.
        Parameters:
            samples (dict): Dictionary that should contain:
                - 'User_emb' (torch.Tensor): User embedding tensor.
                - 'Item_emb' (torch.Tensor): Item embedding tensor.
            ori_samples (dict): Dictionary containing original sample data including:
                - 'UserID' (torch.Tensor): Tensor of user IDs to determine batch size.
                - 'ItemTitle' (list or tensor): Titles corresponding to each item.
                - 'History_Interact_Title' (optional, list): Titles of historical interactions.
            prompt (str): The initial prompt template that may include placeholders such as
                          "<UserID>", "<ItemID>", "<ItemTitle>", and optionally "<ItemTitleList>".
        Returns:
            tuple: A tuple containing:
                - prompt_embeds (torch.Tensor): The tensor containing the prompt embeddings after replacing placeholder tokens.
                - attention_mask (torch.Tensor): The attention mask corresponding to the tokenized prompt.
        """
        
        prompt_ori = prompt
        batch_size = ori_samples['UserID'].shape[0]
        bos_token = self.llm_tokenizer.bos_token
        # <unk> will be replaced by the soft prompt
        unk_token = self.llm_tokenizer.unk_token
        if not "stablelm" in self.llm_name.lower(): # for this model, its bos == eos
            prompt = bos_token + prompt
        prompt = prompt.replace("<UserID>", unk_token)
        prompt = prompt.replace("<ItemID>", unk_token) 
        
        prompt_list = []
        for k in range(batch_size):
            prompt_ = prompt + "" 
            if not isinstance(ori_samples['ItemTitle'][k],str):
                print(f"title float {ori_samples['ItemTitle'][k]}")
                print(f"ori_samples['History_Interact_Title']: {ori_samples[k]['History_Interact_Title']}")
            prompt_ = prompt_.replace("<ItemTitle>", str(ori_samples['ItemTitle'][k]))
            if "History_Interact_Title" in ori_samples.keys():
                ItemTitleList = ", ".join([i[k] for i in ori_samples['History_Interact_Title']])
                prompt_ = prompt_.replace("<ItemTitleList>", ItemTitleList)
            prompt_list.append(prompt_)
            
        if not self.prompt_log_once:
            self.prompt_log_once = True
            logging.info("Prompt example: %s" % prompt_list[0])
            
        self.llm_tokenizer.padding_side = "left" 
        prompts_tokens = self.llm_tokenizer(
            prompt_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(ori_samples['UserID'].device)
        unk_token_id = self.llm_tokenizer.unk_token_id
        replaced_idx = torch.nonzero(prompts_tokens.input_ids == unk_token_id)
        prompt_embeds = self.llm_model.model.embed_tokens(
            prompts_tokens.input_ids)  # (batch_size, txt_len, embed_size)

        if "<UserID>" in prompt_ori and "<ItemID>" in prompt_ori:
            prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]] = torch.cat(
                [samples['User_emb'], samples['Item_emb']], dim=-2).reshape(-1, samples['User_emb'].shape[-1])
        elif "<UserID>" in prompt_ori and "<ItemTitle>" in prompt_ori:
            prompt_embeds[replaced_idx[:, 0], replaced_idx[:, 1]
                          ] = samples['User_emb'].reshape(-1, samples['User_emb'].shape[-1])
        else:
            pass
        return prompt_embeds, prompts_tokens.attention_mask
    
    def save_checkpoint(self, filename):
        """
        Save the model's trainable parameters to a checkpoint file.

        :param filename: Path to the checkpoint file.
        """
        checkpoint = {
            'user_embedding': self.user_embedding.state_dict()
        }
        if self.use_item_embedding:
            checkpoint['item_embedding'] = self.item_embedding.state_dict()
        if hasattr(self, 'rating_predictor'):
            checkpoint['rating_predictor'] = self.rating_predictor.state_dict()

        torch.save(checkpoint, filename)
        logging.info("Model Checkpoint saved to %s" % filename)

    def load_embedding(self, embedding: torch.Tensor, embedding_type: str = 'user'):
        if embedding_type == 'user':
            self.user_embedding.weight.data.copy_(embedding)
        elif embedding_type == 'item':
            self.item_embedding.weight.data.copy_(embedding)
        elif embedding_type == 'both' or embedding_type == 'user+item':
            self.user_embedding.weight.data.copy_(
                embedding[:self.user_embedding.num_embeddings])
            self.item_embedding.weight.data.copy_(
                embedding[self.user_embedding.num_embeddings:])
        else:
            raise ValueError("embedding_type should be 'user' or 'item'")

@ModelRegistry.register("RecModelRP") 
class RecModelRP(LLMModel):
    # rating prediction model
    def __init__(
        self,
        llm_model="",
        prompt_path="",
        nuser=0,
        nitem=0,
        max_txt_len=128,
        use_item_embedding=False, 
    ):
        self.llm_name = llm_model
        super().__init__(llm_model=llm_model,prompt_path=prompt_path,nuser=nuser,nitem=nitem,max_txt_len=max_txt_len,use_item_embedding=use_item_embedding)

        self.rating_predictor = RatingPredictor()
        logging.info("Rating predictor set up already")
        rating_ans_text = ["1", "2", "3", "4", "5"]
        self.rating_ans_token = self.llm_tokenizer(
            rating_ans_text, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len, add_special_tokens=False).to(self.device).input_ids
        
        # convert to 1D tensor if necessary
        if self.rating_ans_token.shape[1] == 1:
            self.rating_ans_token = self.rating_ans_token.squeeze(1)
        elif self.rating_ans_token.shape[1] == 2 and "Phi-3" in self.llm_name:
            # in phi3, 12345 will be tokenized into 2 tokens, we use the second token
            self.rating_ans_token = self.rating_ans_token[:, 1]
            logging.info("Phi-3 model detected, using second token for rating prediction")
        else:
            logging.error("Rating token shape error: %s" %
                          self.rating_ans_token.shape)


    def forward(self, samples):
        sample_embeds, atts_samples = self.prompt_based_encode(
            self.prompt, samples)
        self.llm_tokenizer.padding_side = "right"
        device = samples['UserID'].device  # cpu should be
        ans_text = samples['Rating'].reshape(-1).tolist()
        ans_text = [str(x) for x in ans_text]

        to_regress_tokens = self.llm_tokenizer(
            ans_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )  # in rating prediction, nothing happened

        empty_targets = torch.ones(
            [atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(-100)
        # (batch_size, txt_len), all tokens except the last one (the target) are set to -100.
    
        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llm_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_samples, to_regress_tokens.attention_mask], dim=1)

        # with self.maybe_autocast():
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        # -t_posi is used to extract the prediction scores for the last token, shape: (batch_size, 5)
        logits = outputs.logits[:, -t_posi, :][:, self.rating_ans_token]

        ce_loss = F.cross_entropy(
            logits, samples['Rating'].to(torch.int64) - 1)

        if self.rating_predictor is None:
            return {"ce_loss": ce_loss, "mse_loss": 0}

        predict_rating = self.rating_predictor(logits)
        mse_loss = F.mse_loss(
            predict_rating, samples['Rating'].to(torch.float16).reshape(-1, 1))
        return {"ce_loss": ce_loss, "mse_loss": mse_loss}

    @classmethod
    def from_config(cls, config, nuser, nitem):
        model = cls(
            llm_model=config.model.path,
            prompt_path=config.model.prompt_path,
            nuser=nuser,
            nitem=nitem,
            max_txt_len=config.model.max_txt_len,
            use_item_embedding=config.model.use_item_embedding
        )
        if config.model.get('ckpt', None):
            checkpoint_path = config.model.ckpt
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True)
                model.user_embedding.load_state_dict(checkpoint['user_embedding'])
                if model.use_item_embedding:
                    model.item_embedding.load_state_dict(checkpoint['item_embedding'])
                model.rating_predictor.load_state_dict(checkpoint['rating_predictor'])
                logging.info(
                    "Loaded parameters from checkpoint: %s", checkpoint_path)
            else:
                logging.warning(
                    "Checkpoint path is invalid: %s", checkpoint_path)
        return model


    def generate_for_samples(self, samples, return_all=False, reduction='mean'):
        """
        Generates predictions for the given samples and computes the loss.
        Args:
            samples (dict): A dictionary containing the input samples. It must include 'UserID' and 'Rating' tensors.
            return_all (bool, optional): If True, returns the full model outputs. Otherwise, returns only the logits. Defaults to False.
        Returns:
            dict: A dictionary containing the following keys:
                - "ce_loss" (torch.Tensor): The cross-entropy loss between the predicted and actual ratings.
                - "mse_loss" (torch.Tensor): The mean squared error loss between the predicted and actual ratings, used as a regularization term.
                - "predict_rating" (torch.Tensor, optional): The predicted ratings if a rating predictor is defined.
                - "outputs" (transformers.modeling_outputs.Seq2SeqLMOutput, optional): The full model outputs if return_all is True.
                - "logits" (torch.Tensor, optional): The logits for the predicted ratings if return_all is False.
        """
        sample_embeds, atts_samples = self.prompt_based_encode(
            self.prompt, samples)
        self.llm_tokenizer.padding_side = "right"
        device = samples['UserID'].device  

        ans_text = samples['Rating'].reshape(-1).tolist()
        ans_text = [str(x) for x in ans_text]

        to_regress_tokens = self.llm_tokenizer(
            ans_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        ) 

        empty_targets = torch.ones(
            [atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(-100)
        
        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llm_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_samples, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        logits = outputs.logits[:, -t_posi, :][:, self.rating_ans_token]

        ce_loss = F.cross_entropy(
            logits, samples['Rating'].to(torch.int64) - 1,reduction=reduction)
        return_dict = {"ce_loss": ce_loss}

        predict_rating = self.rating_predictor(logits)
        return_dict["mse_loss"] = F.mse_loss(
            predict_rating, samples['Rating'].to(torch.float16).reshape(-1, 1),reduction=reduction)
        return_dict["predict_rating_float"] = predict_rating
        
        return_dict["predict_rating_int"] = torch.round(
            predict_rating).clamp(1, 5).int()
        return_dict["rmse"] = torch.sqrt(F.mse_loss(
            predict_rating, samples['Rating'].to(torch.float16).reshape(-1, 1),reduction=reduction))
        return_dict["mae"] = F.l1_loss(
            predict_rating, samples['Rating'].to(torch.float16).reshape(-1, 1),reduction=reduction)
        return_dict["acc"] = (return_dict["predict_rating_int"] == samples['Rating'].to(torch.int64)).float().mean()

        if return_all:
            return_dict['outputs'] = outputs
        return_dict['logits'] = logits
        return return_dict

@ModelRegistry.register("RecModelYN")
class RecModelYN(LLMModel):
    # yes or no prediction model
    def __init__(
        self,
        llm_model="",
        prompt_path="",
        nuser=0,
        nitem=0,
        max_txt_len=256,
        use_item_embedding=False, 
    ):
        self.llm_name = llm_model
        super().__init__(llm_model=llm_model,prompt_path=prompt_path,nuser=nuser,nitem=nitem,max_txt_len=max_txt_len,use_item_embedding=use_item_embedding)
        result_ans_text = ["Yes", "No"]
        self.result_ans_token = self.llm_tokenizer(
            result_ans_text, return_tensors="pt", padding="longest", truncation=True, max_length=self.max_txt_len, add_special_tokens=False).to(self.device).input_ids
        if self.result_ans_token.shape[1] == 1:
            self.result_ans_token = self.result_ans_token.squeeze(1)
        else:
            logging.error("Rating token shape error: %s" %
                          self.result_ans_token.shape)
            
    def forward(self, samples):
        sample_embeds, atts_samples = self.prompt_based_encode(
            self.prompt, samples)
        self.llm_tokenizer.padding_side = "right"
        device = samples['UserID'].device  # cpu should be
        
        ans_text = samples['Label'].reshape(-1).tolist()
        # Convert 0 to "No" and 1 to "Yes"
        ans_text = ["No" if Label == 0 else "Yes" for Label in ans_text]

        to_regress_tokens = self.llm_tokenizer(
            ans_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        )

        empty_targets = torch.ones(
            [atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(-100)

        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llm_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_samples, to_regress_tokens.attention_mask], dim=1)

        # with self.maybe_autocast():
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        pos_ans_id = self.result_ans_token[0]
        
        logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        loss = F.binary_cross_entropy_with_logits(logits,samples['Label'].to(torch.float32).reshape(-1))
        return {"loss": loss}
    
    def generate_for_samples(self, samples, return_all=False, reduction='mean'):
        """
        Generates predictions for the given samples and computes the loss.
        """
        sample_embeds, atts_samples = self.prompt_based_encode(
            self.prompt, samples)
        self.llm_tokenizer.padding_side = "right"
        device = samples['UserID'].device  # cpu should be
        
        ans_text = samples['Label'].reshape(-1).tolist()
        # Convert 0 to "No" and 1 to "Yes"
        ans_text = ["No" if Label == 0 else "Yes" for Label in ans_text]

        to_regress_tokens = self.llm_tokenizer(
            ans_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(device)
        t_posi = to_regress_tokens.input_ids.shape[-1] + 1

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llm_tokenizer.pad_token_id, -100
        ) 

        empty_targets = torch.ones(
            [atts_samples.shape[0], atts_samples.shape[1]], dtype=torch.long).to(device).fill_(-100)
        targets = torch.cat([empty_targets, targets], dim=1)

        to_regress_embeds = self.llm_model.model.embed_tokens(
            to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([sample_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat(
            [atts_samples, to_regress_tokens.attention_mask], dim=1)

        # with self.maybe_autocast():
        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        pos_ans_id = self.result_ans_token[0]
        logits = outputs.logits[:, -t_posi, :][:, pos_ans_id]
        loss = F.binary_cross_entropy_with_logits(logits,samples['Label'].to(torch.float32).reshape(-1),reduction=reduction)
        return_dict = {"loss": loss, "logits": logits, "Label": samples['Label'], "UserID": samples['UserID']}
        if return_all:
            return_dict['outputs'] = outputs
        return return_dict
    
    @classmethod
    def from_config(cls, config, nuser, nitem):
        model = cls(
            llm_model=config.model.path,
            prompt_path=config.model.prompt_path,
            nuser=nuser,
            nitem=nitem,
            max_txt_len=config.model.max_txt_len,
            use_item_embedding=config.model.use_item_embedding
        )
        if config.model.get('ckpt', None):
            checkpoint_path = config.model.ckpt
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=True)
                model.user_embedding.load_state_dict(checkpoint['user_embedding'])
                if model.use_item_embedding:
                    model.item_embedding.load_state_dict(checkpoint['item_embedding'])
                logging.info(
                    "Loaded parameters from checkpoint: %s", checkpoint_path)
            else:
                logging.warning(
                    "Checkpoint path is invalid: %s", checkpoint_path)
        return model

class AdapterModel(BaseModel):
    def __init__(self,input_dim,output_dim,dropout=0.1):
        # AdapterMigrationModel 中的adapter部分
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, input_dim),
        )
        self.residual = nn.Identity()
        self.proj_up = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.proj_2 = nn.Sequential(
            nn.Linear(output_dim, 4096),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim),
        )
        self.proj_out = nn.Sequential(
            nn.Linear(output_dim, 4096),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim),
        )
        self.norm3 = nn.LayerNorm(output_dim)
        self.norm4 = nn.LayerNorm(output_dim)
        
    def forward(self,x):
        x1 = self.proj(self.norm1(x))
        x1 = self.residual(x) + x1
        x1 = self.proj_up(self.norm2(x1))
        
        x2 = self.proj_2(self.norm3(x1))
        x2 = self.residual(x1) + x2
        x2 = self.proj_out(self.norm4(x2))
        
        return x2
    
    def save_checkpoint(self, filename):
        # 保存模型的可训练参数到一个检查点文件
        torch.save(self.state_dict(), filename)
        logging.info("PromptMigrationModel Checkpoint saved to %s" % filename)

    @classmethod
    def build_model(cls, input_dim, output_dim, config):
        model = cls(
            input_dim=input_dim,
            output_dim=output_dim,
        )
        if config.model.ckpt is not None:
            checkpoint_path = config.model.ckpt
            if os.path.isfile(checkpoint_path):
                model.load_state_dict(torch.load(
                    checkpoint_path, map_location="cpu"))
                logging.info(
                    "Loaded parameters from checkpoint: %s", checkpoint_path)
            else:
                logging.warning(
                    "Checkpoint path is invalid: %s", checkpoint_path)
        return model
    
class AdapterModelSmall(AdapterModel):
    def __init__(self,input_dim,output_dim,dropout=0.1):
        # AdapterMigrationModel 中的adapter部分
        BaseModel.__init__(self)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, input_dim),
        )
        self.residual = nn.Identity()
        self.proj_up = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4096, output_dim),
        )
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        x1 = self.proj(self.norm1(x))
        x1 = self.residual(x) + x1
        x1 = self.proj_up(self.norm2(x1))
        return x1
    
@ModelRegistry.register("AdapterModelRP")
class AdapterModelRP(RecModelRP):
    def __init__(
        self,llm_model="",
        prompt_path="",
        nuser=0,
        nitem=0,
        max_txt_len=128,
        use_item_embedding=False, 
        adapter_input_dim=2048,
        adapter_dropout=0.1,
        freeze_rating_predictor=False,
        config=None
    ):
        super().__init__(llm_model,prompt_path,nuser,nitem,max_txt_len,use_item_embedding)
        self._init_soft_prompt(nuser,nitem,adapter_input_dim) 
        self.freeze_rating_predictor = freeze_rating_predictor
        self.adapter_input_dim = adapter_input_dim
        self.adapter = AdapterModel(adapter_input_dim,self.llm_model.config.hidden_size,adapter_dropout)
        if freeze_rating_predictor:
            for name,param in self.rating_predictor.named_parameters():
                param.requires_grad = False
        for name,param in self.user_embedding.named_parameters():
            param.requires_grad = False
        if self.use_item_embedding:
            for name,param in self.item_embedding.named_parameters():
                param.requires_grad = False
                
    @classmethod
    def from_config(cls,config,nuser,nitem):
        sp_path = config.model.get("soft_prompt_path",None)
        if sp_path is None or os.path.isfile(sp_path) is False:
            raise ValueError("Soft prompt path is not set or invalid: %s" % sp_path)

        checkpoint = torch.load(sp_path,map_location="cpu",weights_only=True)
        sp_dim = checkpoint['user_embedding']['weight'].shape[1]
        
        model = cls(
            llm_model=config.model.path,
            prompt_path=config.model.prompt_path,
            nuser=nuser,
            nitem=nitem,
            max_txt_len=config.model.max_txt_len,
            use_item_embedding=config.model.use_item_embedding,
            adapter_input_dim=sp_dim,
            adapter_dropout=config.model.dropout,
            freeze_rating_predictor=config.model.freeze_predictor,
            config=config
        )

        model.user_embedding.load_state_dict(checkpoint["user_embedding"])
        if model.use_item_embedding:
            model.item_embedding.load_state_dict(checkpoint["item_embedding"])
        if model.rating_predictor is not None and "rating_predictor" in checkpoint:
            model.rating_predictor.load_state_dict(checkpoint["rating_predictor"])
        logging.info("Loaded embedding from checkpoint: %s",sp_path)
        
        if config.model.get('ckpt',None):
            checkpoint_path = config.model.ckpt
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
                model.adapter.load_state_dict(checkpoint['adapter'])
                if model.rating_predictor is not None and 'rating_predictor' in checkpoint:
                    model.rating_predictor.load_state_dict(checkpoint['rating_predictor'])
                logging.info("Loaded adapter ckpt from checkpoint: %s",checkpoint_path)
            else:
                logging.warning("Checkpoint path is invalid: %s",checkpoint_path)
        return model
    
    def encode_recdata(self, samples):
        batch_size = samples['UserID'].shape[0]
        hidden_size = self.adapter_input_dim
        user_embeds = self.user_embedding(samples['UserID']).reshape(
            # (batch_size, 1, embed_size) 
            batch_size, -1, hidden_size)

        if self.use_item_embedding:
            item_embeds = self.item_embedding(samples['ItemID']).reshape(
                batch_size, -1, hidden_size)  # (batch_size, 1, embed_size)
        else:
            item_embeds = None

        if self.device != torch.device("cpu"):
            user_embeds = user_embeds.half()
            if self.use_item_embedding:
                item_embeds = item_embeds.half()
        sample_embeds_llm = {
            "User_emb": user_embeds,
            "Item_emb": item_embeds
        }
        sample_embeds_llm['User_emb'] = self.adapter(sample_embeds_llm['User_emb'])
        if self.use_item_embedding:
            sample_embeds_llm['Item_emb'] = self.adapter(sample_embeds_llm['Item_emb'])
        return sample_embeds_llm
            
    
    def save_checkpoint(self, filename):
        """
        Save the model's trainable parameters to a checkpoint file.

        :param filename: Path to the checkpoint file.
        """
        checkpoint = {
            'adapter': self.adapter.state_dict(),
            'rating_predictor': self.rating_predictor.state_dict(),
        }
        
        torch.save(checkpoint, filename)
        logging.info("Model Checkpoint saved to %s" % filename)
        
    def _reload_best_model(self, model, ckpt_path):
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.adapter.load_state_dict(checkpoint['adapter'])
            if 'rating_predictor' in checkpoint:
                model.rating_predictor.load_state_dict(checkpoint['rating_predictor'])
            logging.info("Loaded parameters from checkpoint: %s", ckpt_path)
        else:
            logging.warning("Checkpoint path is invalid: %s", ckpt_path)
        return model

@ModelRegistry.register("AdapterModelYN")
class AdapterModelYN(RecModelYN):
    def __init__(
        self,llm_model="",
        prompt_path="",
        nuser=0,
        nitem=0,
        max_txt_len=128,
        use_item_embedding=False,   
        adapter_input_dim=2048,
        adapter_dropout=0.1,
        config=None
    ):
        super().__init__(llm_model,prompt_path,nuser,nitem,max_txt_len,use_item_embedding)
        self._init_soft_prompt(nuser,nitem,adapter_input_dim)
        self.adapter_input_dim = adapter_input_dim
        self.adapter = AdapterModel(adapter_input_dim,self.llm_model.config.hidden_size,adapter_dropout)
        for name,param in self.user_embedding.named_parameters():
            param.requires_grad = False
        if self.use_item_embedding:
            for name,param in self.item_embedding.named_parameters():
                param.requires_grad = False
                
    def encode_recdata(self, samples):
        batch_size = samples['UserID'].shape[0]
        hidden_size = self.adapter_input_dim
        user_embeds = self.user_embedding(samples['UserID']).reshape(
            batch_size, -1, hidden_size)

        if self.use_item_embedding:
            item_embeds = self.item_embedding(samples['ItemID']).reshape(
                batch_size, -1, hidden_size)  # (batch_size, 1, embed_size)
        else:
            item_embeds = None

        if self.device != torch.device("cpu"):
            user_embeds = user_embeds.half()
            if self.use_item_embedding:
                item_embeds = item_embeds.half()
        sample_embeds_llm = {
            "User_emb": user_embeds,
            "Item_emb": item_embeds
        }
        sample_embeds_llm['User_emb'] = self.adapter(sample_embeds_llm['User_emb'])
        if self.use_item_embedding:
            sample_embeds_llm['Item_emb'] = self.adapter(sample_embeds_llm['Item_emb'])
        return sample_embeds_llm
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'adapter': self.adapter.state_dict(),
        }
        
        torch.save(checkpoint, filename)
        logging.info("Model Checkpoint saved to %s" % filename)     
    
    def _reload_best_model(self, model, ckpt_path):
        if os.path.isfile(ckpt_path):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model.adapter.load_state_dict(checkpoint['adapter'])
            logging.info("Loaded parameters from checkpoint: %s", ckpt_path)
        else:
            logging.warning("Checkpoint path is invalid: %s", ckpt_path)
        return model      
    
    @classmethod
    def from_config(cls,config,nuser,nitem):
        
        sp_path = config.model.get("soft_prompt_path",None)
        if sp_path is None or os.path.isfile(sp_path) is False:
            raise ValueError("Soft prompt path is not set or invalid: %s" % sp_path)

        checkpoint = torch.load(sp_path,map_location="cpu",weights_only=True)
        sp_dim = checkpoint['user_embedding']['weight'].shape[1]
        
        model = cls(
                llm_model=config.model.path,
                prompt_path=config.model.prompt_path,
                nuser=nuser,
                nitem=nitem,
                max_txt_len=config.model.max_txt_len,
                use_item_embedding=config.model.use_item_embedding,
                adapter_input_dim=sp_dim,
                adapter_dropout=config.model.dropout,
                config=config
            )
        
        model.user_embedding.load_state_dict(checkpoint["user_embedding"])
        if model.use_item_embedding:
            model.item_embedding.load_state_dict(checkpoint["item_embedding"])
        logging.info("Loaded embedding from checkpoint: %s",sp_path)
                
        if config.model.get('ckpt',None):
            checkpoint_path = config.model.ckpt
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path,map_location="cpu",weights_only=True)
                model.adapter.load_state_dict(checkpoint['adapter'])
                logging.info("Loaded adapter ckpt from checkpoint: %s",checkpoint_path)
            else:
                logging.warning("Checkpoint path is invalid: %s",checkpoint_path)
        return model
  
@ModelRegistry.register("AdapterModelRP_PA1")    
class AdapterModelRP_PA1(AdapterModelRP):
    @classmethod
    def from_config(cls, config, nuser, nitem):
        sp_path = config.model.get("soft_prompt_path", None)
        if sp_path is None or not os.path.isfile(sp_path):
            raise ValueError(f"Soft prompt path is invalid: {sp_path}")
        checkpoint = torch.load(sp_path, map_location="cpu", weights_only=True)
        sp_dim = checkpoint['user_embedding']['weight'].shape[1]
        
        sp_path_2 = config.model.get("soft_prompt_path_2", None)  
        if sp_path_2 is None or not os.path.isfile(sp_path_2):
            raise ValueError(f"Second soft prompt path is invalid: {sp_path_2}")
        checkpoint_2 = torch.load(sp_path_2, map_location="cpu", weights_only=True)  
        sp_dim_2 = checkpoint_2['user_embedding']['weight'].shape[1] 


        total_sp_dim = sp_dim + sp_dim_2
        
        model = cls(
            llm_model=config.model.path,
            prompt_path=config.model.prompt_path,
            nuser=nuser,
            nitem=nitem,
            max_txt_len=config.model.max_txt_len,
            use_item_embedding=config.model.use_item_embedding,
            adapter_input_dim=total_sp_dim,  
            adapter_dropout=config.model.dropout,
            freeze_rating_predictor=config.model.freeze_predictor,
            config=config
        )
        

        user_weight_1 = checkpoint["user_embedding"]["weight"]
        user_weight_2 = checkpoint_2["user_embedding"]["weight"]
        combined_user_weight = torch.cat([user_weight_1, user_weight_2], dim=1)
        model.user_embedding.weight.data = combined_user_weight


        if model.use_item_embedding:
            item_weight_1 = checkpoint["item_embedding"]["weight"]
            item_weight_2 = checkpoint_2["item_embedding"]["weight"]
            combined_item_weight = torch.cat([item_weight_1, item_weight_2], dim=1)
            model.item_embedding.weight.data = combined_item_weight
        
        if model.rating_predictor is not None and "rating_predictor" in checkpoint:
            model.rating_predictor.load_state_dict(checkpoint["rating_predictor"])
        
        logging.info("Loaded combined embeddings from: %s and %s", sp_path, sp_path_2)
        
        if config.model.get('ckpt', None):
            checkpoint_path = config.model.ckpt
            if os.path.isfile(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model.adapter.load_state_dict(ckpt['adapter'])
                if model.rating_predictor is not None and 'rating_predictor' in ckpt:
                    model.rating_predictor.load_state_dict(ckpt['rating_predictor'])
                logging.info("Loaded adapter ckpt from: %s", checkpoint_path)
            else:
                logging.warning("Checkpoint path is invalid: %s", checkpoint_path)
                
        return model