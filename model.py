import torch
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
import torch.nn as nn
from torch.nn import Module, Linear, Identity, Parameter, Embedding, RMSNorm, functional as F
from config import config
from datasets import IterableDataset
import xformers.ops as xops
import os
import pandas as pd
from itertools import zip_longest

# ========= BITLINEAR LAYER ============ #
def activation_quant(x, num_bits = 8):
    Qn = -2 ** (num_bits - 1)
    Qp = 2 ** (num_bits - 1) - 1
    scale = Qp / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x*scale).round().clamp_(Qn, Qp) / scale
    return y

def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w*scale).round().clamp_(-1, 1)
    return u, scale

class BitLinear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.rms_norm = RMSNorm(in_features)

    def forward(self, x):
        # Implement 1-bit forward pass here
        # This is a placeholder implementation
        w = self.weight
        x_norm = self.rms_norm(x)
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_new, scale = weight_quant(w)
        self.scale = scale
        w_quant = w + (w_new - w).detach()
        return F.linear(x_quant, w_quant) / scale
# =========================================== #

# switch model's nn.Linear to BitLinear
# model => bitnet_model
def prepare_for_1_58bit_training(model):
    for name, module in model.named_children():
        if isinstance(module, Linear):
            setattr(model, name, BitLinear(module.in_features, module.out_features))
        elif isinstance(module, RMSNorm):
            setattr(model, name, Identity())
        else:
            prepare_for_1_58bit_training(module)

    return model

# build a tokenizer from data (DataFrame, specifically)
def build_tkizer(df_path: str):
    if config.tkizer_save:
        if os.path.exists(config.tkizer_save_path):
            tkizer = AutoTokenizer.from_pretrained(config.tkizer_save_path, use_fast=True)
            print(f"Tokenizer loaded")
            config.pad_token_id = tkizer.pad_token_id
            return tkizer

    df = pd.read_csv(df_path)
    
     # Interleave source and target texts
    src_texts = df['src'].tolist()
    tgt_texts = df['tgt'].tolist()
    interleaved_texts = [text for pair in zip_longest(src_texts, tgt_texts) for text in pair if text is not None]


    # Train tokenizer from our datasets
    pretrained_tkizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    
    # Train new tokenizer with interleaved texts
    tkizer = pretrained_tkizer.train_new_from_iterator(interleaved_texts, vocab_size=config.vocab_size)

    # Ensure all necessary special tokens are present
    special_tokens = {
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
    }
    
    # Add special tokens
    tkizer.add_special_tokens(special_tokens)
    config.pad_token_id = tkizer.pad_token_id

    # Add extra_id tokens (sentinel tokens)
    num_extra_ids = 100  # T5 typically uses 100 sentinel tokens
    tkizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})

    # Save tokenizer
    if config.tkizer_save:
        if not os.path.exists(config.tkizer_save_path):
            os.makedirs(config.tkizer_save_path)
        tkizer.save_pretrained(config.tkizer_save_path)
        print(f"Tokenizer saved")

    return tkizer

# create new model => BitLinear Model
def create_model():
    model_config = T5Config(
        vocab_size=config.vocab_size,  # Use a single vocab_size
        d_model=config.d_model,
        d_kv=config.d_model // config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_position_embeddings=config.max_length,
        decoder_start_token_id=config.pad_token_id,
        # use_cache=False,
    )
    
    # Use T5ForConditionalGeneration instead of DualTokenizerT5
    model = T5ForConditionalGeneration(model_config)
    
    # The lm_head is already correctly sized in T5ForConditionalGeneration,
    # so we don't need to resize it manually

    # Initialize weights randomly
    def init_weights(m):
        if isinstance(m, Linear) or isinstance(m, BitLinear):
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, Embedding):
            torch.nn.init.xavier_uniform_(m.weight)

    model = prepare_for_1_58bit_training(model)
    model.apply(init_weights)
    model = enable_xformers(model)
    
    return model

def enable_xformers(model):
    for layer in model.encoder.block + model.decoder.block:
        layer.layer[0].SelfAttention.process_mask = xops.memory_efficient_attention
        if hasattr(layer.layer[-1], 'EncDecAttention'):
            layer.layer[-1].EncDecAttention.process_mask = xops.memory_efficient_attention
    return model