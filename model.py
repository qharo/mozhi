import torch
from transformers import T5ForConditionalGeneration, T5Config, AutoTokenizer
import torch.nn as nn
from config import config
from datasets import IterableDataset
import xformers.ops as xops
import os

# dataset = => Tuple(src_tkizer, tgt_tkizer : T5Tokenizer)
def build_tkizers(dataset: IterableDataset):
    # load directly if saved
    if config.tkizer_save:
        if os.path.exists(config.src_tkizer_save_path) and os.path.exists(config.tgt_tkizer_save_path):
            src_tkizer = AutoTokenizer.from_pretrained(config.src_tkizer_save_path, use_fast=True)
            tgt_tkizer = AutoTokenizer.from_pretrained(config.tgt_tkizer_save_path, use_fast=True)
            print(f"Tokenizers loaded")
            return src_tkizer, tgt_tkizer

    # iterator over dataset
    src_iterator = (item['src'] for item in dataset)
    tgt_iterator = (item['tgt'] for item in dataset)

    # train tkizer from our datasets
    pretrained_tkizer = AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    src_tkizer = pretrained_tkizer.train_new_from_iterator(src_iterator, vocab_size=config.src_vocab_size)
    tgt_tkizer = pretrained_tkizer.train_new_from_iterator(tgt_iterator, vocab_size=config.tgt_vocab_size)

    # Ensure all necessary special tokens are present
    special_tokens = {
        'eos_token': '</s>',
        'unk_token': '<unk>',
        'pad_token': '<pad>',
    }

    # add special tokens
    src_tkizer.add_special_tokens(special_tokens)
    tgt_tkizer.add_special_tokens(special_tokens)
    # config.pad_token_id = src_tkizer.pad_token_id

    # Add extra_id tokens (sentinel tokens)
    num_extra_ids = 100  # T5 typically uses 100 sentinel tokens
    src_tkizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})
    tgt_tkizer.add_special_tokens({'additional_special_tokens': [f'<extra_id_{i}>' for i in range(num_extra_ids)]})

    print(f"Tokenizers built")

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
        self.encoder.embed_tokens = nn.Embedding(config.src_vocab_size, config.d_model)
        self.decoder.embed_tokens = nn.Embedding(config.tgt_vocab_size, config.d_model)
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
    model.lm_head = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)

    # Initialize weights randomly
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.xavier_uniform_(m.weight)

    model.apply(init_weights)

    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # if training for bitnet
    # if config.use_bit_linear:
    #     replace_linear_layers(model)

    return model


# def replace_linear_layers(model):
#     replacement_count = 0
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             parent_name = '.'.join(name.split('.')[:-1])
#             child_name = name.split('.')[-1]
#             parent = model.get_submodule(parent_name)
#             custom_linear = BitLinear(module.in_features, module.out_features, bias=module.bias is not None)
#             setattr(parent, child_name, custom_linear)
#             replacement_count += 1
#     print(f"Replaced {replacement_count} nn.Linear layers with CustomLinear")

   # for name, module in list(model.named_modules()):  # Create a list to avoid mutation during iteration
    #     if isinstance(module, nn.Linear):
    #         parent_name, child_name = name.rsplit('.', 1) if '.' in name else ('', name)
    #         parent = model if parent_name == '' else model.get_submodule(parent_name)
    #         setattr(parent, child_name, bnb.nn.Linear8bitLt(
    #             module.in_features, 
    #             module.out_features, 
    #             bias=module.bias is not None
    #         ))