import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_vicuna'

    config.tokenizer_paths=["/mnt/workspace/workgroup/yiran/Vicuna-13b"]
    config.model_paths=["/mnt/workspace/workgroup/yiran/Vicuna-13b"]
    config.conversation_templates=['vicuna']

    return config