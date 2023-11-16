import yaml
from easydict import EasyDict as edict

def read_config(yml_config):
    try:
        with open(yml_config, 'r') as f:
            config = yaml.safe_load(f)
        return edict(config)
    except:
        print('config file cannot be read.') 