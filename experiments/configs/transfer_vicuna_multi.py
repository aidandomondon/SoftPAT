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
        "drive/MyDrive/LLMs/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6",
        # "path/to/model/vicuna-13b-v1.3",
    ]
    # config.tokenizer_kwargs = [{"use_fast": False}, {"use_fast": False}]
    config.tokenizer_kwargs = [{"use_fast": False}]
    config.model_paths = [
        "drive/MyDrive/LLMs/models--lmsys--vicuna-7b-v1.3/snapshots/236eeeab96f0dc2e463f2bebb7bb49809279c6d6",
        # "/path/to/model/vicuna-7b-v1.3",
    ]
    config.model_kwargs = [
        {"low_cpu_mem_usage": True, "use_cache": False},
        # {"low_cpu_mem_usage": True, "use_cache": False}
    ]
    # config.conversation_templates = ["vicuna", "vicuna"]
    # config.devices = ["cuda:0", "cuda:1"]
    config.conversation_templates = ["vicuna"]
    config.devices = ["cuda:0"]
    config.benign_file="../../data/benign/benign_vicuna.csv"

    return config
