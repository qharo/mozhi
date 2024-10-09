from transformers import AutoConfig, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Config
import torch.nn as nn
from config import config

# custom
class DualTokenizerT5(T5ForConditionalGeneration):
    def __init__(self, t5config):
        super().__init__(t5config)
        self.shared = None
        self.encoder.embed_tokens = nn.Embedding(config.src_vocab_size, config.d_model)
        self.decoder.embed_tokens = nn.Embedding(config.tgt_vocab_size, config.d_model)

# creates model
def create_model(pad_token_id):
    model_config = T5Config(
        vocab_size=max(config.src_vocab_size, config.tgt_vocab_size),  # Set to max for compatibility
        d_model=config.d_model,
        d_kv=config.d_model // config.num_heads,
        d_ff=config.d_ff,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        max_position_embeddings=config.max_length,
        decoder_start_token_id=pad_token_id
    ) 
    model = DualTokenizerT5(model_config)

    # Resize the output layer to match target vocabulary size
    model.lm_head = nn.Linear(config.d_model, config.tgt_vocab_size, bias=False)
    
    # if training for bitnet
    if config.use_bit_linear:
        replace_linear_layers(model)

    return model


def replace_linear_layers(model):
    replacement_count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            parent = model.get_submodule(parent_name)
            custom_linear = CustomLinear(module.in_features, module.out_features, bias=module.bias is not None)
            setattr(parent, child_name, custom_linear)
            replacement_count += 1
    print(f"Replaced {replacement_count} nn.Linear layers with CustomLinear")
