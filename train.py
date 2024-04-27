import os
import random
import string
import sys
sys.path.append('minGPT/')

from torch.utils.tensorboard import SummaryWriter

from mingpt.bpe import BPETokenizer
from mingpt.model import GPT
from mingpt.utils import set_seed, setup_logging

from config_setup import get_config
from data_handling import *
from training import *


if __name__ == '__main__':

  ### Create the tokenizer ###

  tokenizer = BPETokenizer()
  tokenizer.pad_token_id = tokenizer.encoder.encoder['<|endoftext|>']


  ### Prepare the config ###

  config = get_config(sys.argv[1:])
  config.model.vocab_size = len(tokenizer.encoder.encoder)
  print(config)
  setup_logging(config)

  if config.system.wandb:
    import wandb
    tags = [config.system.tag] if config.system.tag else None
    wandb.init(project=config.system.project, sync_tensorboard=True,
               config=config.to_dict(), tags=tags)

  logger = None


  ### Set the seed ###

  set_seed(config.system.seed)


  ### Construct the model and optimizer ###

  model = GPT(config.model)
  model = model.to(config.trainer.device)
  optimizer = model.configure_optimizers(config.trainer)
  print(f'Running on device {config.trainer.device}')


  ### Prepare the data ###

  datasets = prepare_data(config)
  dataloaders = prepare_dataloaders(datasets, tokenizer, config)

  
  ### Train the model ###
  
  opt_model = torch.compile(model)
  opt_model, optimizer = train_loop(
    opt_model, optimizer, dataloaders, config, logger, tokenizer)