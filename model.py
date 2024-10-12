import torch
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
import torch.nn as nn
from torch.nn import Module, Linear, RMSNorm, Identity, Parameter, Embedding, functional as F
from config import config
from datasets import IterableDataset
import xformers.ops as xops
import os
import pandas as pd

# dataset = => Tuple(src_tkizer, tgt_tkizer : T5Tokenizer)
def build_tkizers(df_path: str):

    if config.tkizer_save:
        if os.path.exists(config.src_tkizer_save_path) and os.path.exists(config.tgt_tkizer_save_path):
            src_tkizer = AutoTokenizer.from_pretrained(config.src_tkizer_save_path, use_fast=True)
            tgt_tkizer = AutoTokenizer.from_pretrained(config.tgt_tkizer_save_path, use_fast=True)
            print(f"Tokenizers loaded")
            config.pad_token_id = src_tkizer.pad_token_id
            return src_tkizer, tgt_tkizer

    df = pd.read_csv(df_path)
    iterator = lambda x: df[x].tolist()

    # train tkizer from our datasets
    pretrained_tkizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    src_tkizer = pretrained_tkizer.train_new_from_iterator(iterator('src'), vocab_size=config.src_vocab_size)
    tgt_tkizer = pretrained_tkizer.train_new_from_iterator(iterator('tgt'), vocab_size=config.tgt_vocab_size)

    # Ensure all necessary special tokens are present
    special_tokens = {
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
    }

    # add special tokens
    src_tkizer.add_special_tokens(special_tokens)
    tgt_tkizer.add_special_tokens(special_tokens)
    config.pad_token_id = src_tkizer.pad_token_id

    # Add extra_id tokens (sentinel tokens)
    num_extra_ids = 100  # T5 typically uses 100 sentinel tokens
    src_tkizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})
    tgt_tkizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})


    # save tkizers
    if config.tkizer_save:
        if not os.path.exists(config.src_tkizer_save_path):
            os.makedirs(config.src_tkizer_save_path)
            src_tkizer.save_pretrained(config.src_tkizer_save_path)
        if not os.path.exists(config.tgt_tkizer_save_path):
            os.makedirs(config.tgt_tkizer_save_path)
            tgt_tkizer.save_pretrained(config.tgt_tkizer_save_path)
        print(f"Tokenizer saved")


    return src_tkizer, tgt_tkizer


# src/tgt tkizer child of t5
class DualTokenizerT5(T5ForConditionalGeneration):
    def __init__(self, t5config):
        super().__init__(t5config)
        self.shared = None
        self.encoder.embed_tokens = Embedding(config.src_vocab_size, config.d_model)
        self.decoder.embed_tokens = Embedding(config.tgt_vocab_size, config.d_model)
        # self.gradient_checkpointing_enable()
        self.enable_xformers()

    def enable_xformers(self):
        for layer in self.encoder.block + self.decoder.block:
            layer.layer[0].SelfAttention.process_mask = xops.memory_efficient_attention
            if hasattr(layer.layer[-1], 'EncDecAttention'):
                layer.layer[-1].EncDecAttention.process_mask = xops.memory_efficient_attention

# creates model
def create_model():
    model_config = T5Config(
        vocab_size=max(config.src_vocab_size, config.tgt_vocab_size),  # Set to max for compatibility
        d_model=config.d_model,
        d_kv=config.d_model // config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_position_embeddings=config.max_length,
        decoder_start_token_id=config.pad_token_id,
        # use_cache=False,
    )
    model = DualTokenizerT5(model_config)

    # Resize the output layer to match target vocabulary size
    model.lm_head = Linear(config.d_model, config.tgt_vocab_size, bias=False)

    # Initialize weights randomly
    def init_weights(m):
        if isinstance(m, Linear) or isinstance(m, BitLinear):
            torch.nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, Embedding):
            torch.nn.init.xavier_uniform_(m.weight)

    model = prepare_for_1_58bit_training(model)
    model.apply(init_weights)

    return model

def prepare_for_1_58bit_training(model):
    for name, module in model.named_children():
        if isinstance(module, Linear):
            setattr(model, name, BitLinear(module.in_features, module.out_features))
        # elif isinstance(module, nn.SwiGLU):
        #     setattr(model, name, BitLinear(module.in_features, module.out_features))
        elif isinstance(module, RMSNorm):
            # Remove RMSNorm layers
            setattr(model, name, Identity())
        else:
            # Recursively apply the function to submodules
            prepare_for_1_58bit_training(module)
    
    return model


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