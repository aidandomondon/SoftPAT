import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.transfer = True
    config.logfile = ""

    config.progressive_goals = False
    config.stop_on_success = False
    config.tokenizer_paths = [
        "openai-community/gpt2",
    ]

    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "openai-community/gpt2"
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False}
    ]

    config.conversation_templates = ["gpt2-chatbot"]
    config.devices = ["cuda:0"]
    config.benign_file="../../data/benign/benign_gpt2.csv"
    config.logfile="../../data/gpt2_log.csv"

    return config
