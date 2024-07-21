import os

os.sys.path.append("..")
from configs.template import get_config as default_config

def get_config():
    
    config = default_config()

    config.result_prefix = 'results/individual_vicuna'

    config.tokenizer_paths=["/mnt/workspace/workgroup/workgroup_intern/yiran/Vicuna-13b-v1.5"]
    config.model_paths=["/mnt/workspace/workgroup/workgroup_intern/yiran/Vicuna-13b-v1.5"]
    config.conversation_templates=['vicuna']

    return config