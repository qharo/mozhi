from transformers import AutoConfig, AutoModelForSeq2SeqLM
import torch.nn as nn
from bitlinear import BitLinear
from config import config

def create_model():
    model_config = AutoConfig.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_config(model_config)

    # if config.use_bit_linear:
    #     replace_linear_layers(model)

    return model

# def replace_linear_layers(model):
#     for name, module in model.named_modules():
#         if isinstance(module, nn.Linear):
#             parent_name = '.'.join(name.split('.')[:-1])
#             child_name = name.split('.')[-1]
#             parent = model.get_submodule(parent_name)
#             setattr(parent, child_name, BitLinear(module.in_features, module.out_features, bias=module.bias is not None))
