import os

from configs.template import get_config as default_config

def get_config():
    
    config = default_config()
    config.model_paths = [
        "lmsys/vicuna-7b-v1.3",
        # more models
    ]
    config.tokenizer_paths = [
        "lmsys/vicuna-7b-v1.3",
        # more tokenizers
    ]
    return config