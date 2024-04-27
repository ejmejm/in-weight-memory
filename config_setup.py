import sys
import yaml
sys.path.append('minGPT/')

import torch

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import CfgNode as CN


def get_config(cmd_args=None):
  # Create the default config
  config = create_default_config()

  # Check if any of the args start with '--config='
  # If so, load the config from that file and remove
  # the arg from cmd_args
  for i, arg in enumerate(cmd_args):
    if arg.startswith('--config='):
      config_path = arg.split('=')[1]
      cmd_args.pop(i)

      print(f'Loading config from {config_path}')
      
      config_dict = load_yaml(config_path)

      config.merge_from_dict(config_dict)
      break
  
  
  if cmd_args is not None:
    config.merge_from_args(cmd_args)

  return config


def load_yaml(config_path):
  """Loads a dictionary of config params from a yaml file."""
  with open(config_path, 'r') as f:
    config_dict = yaml.load(f, Loader=yaml.FullLoader)
  return config_dict


def create_default_config():
  C = CN()

  # system
  C.system = CN()
  C.system.seed = 3407
  C.system.work_dir = './out/working_dir'
  C.system.debug = False # Currently unused
  C.system.wandb = False
  C.system.project = 'in-weight-memory'
  C.system.log_model_stats = False
  C.system.tag = None
  C.system.run_id = None
  C.system.dataset = 'tinystories' # 'bookcorpus'

  # model
  C.model = GPT.get_default_config()
  C.model.block_size = 64 # context length

  # trainer
  C.trainer = Trainer.get_default_config()
  C.trainer.learning_rate = 1e-4
  C.trainer.batch_size = 64
  C.trainer.grad_accumulation_steps = 1
  C.trainer.test_split = 0.1
  C.trainer.valid_split = 0.01
  C.trainer.epochs = 10
  C.trainer.max_samples = None
  C.trainer.log_freq = 20
  C.trainer.print_freq = 100
  C.trainer.eval_freq = 1000
  C.trainer.n_examples = 4
  C.trainer.device = 'cuda' \
    if torch.cuda.is_available() else 'cpu'
  C.trainer.fp16 = True

  return C
