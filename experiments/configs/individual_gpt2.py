import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_gpt2'

    # see `transformers.AutoTokenizer.from_pretrained`'s parameter `pretrained_model_name_or_path`
    config.tokenizer_paths=["openai-community/gpt2"]
    
    # see `transformers.PreTrainedModel.from_pretrained`'s parameter `pretrained_model_name_or_path`
    config.model_paths=["openai-community/gpt2"]

    # see conversation templates supported by FastChat 
    # https://github.com/lm-sys/FastChat/blob/main/fastchat/model/model_adapter.py#L2505
    config.conversation_templates=['gpt2-chatbot']

    return config