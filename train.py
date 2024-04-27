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


  ### Set up logging ###

  # Set log dir to a random new folder in $SLURM_TMPDIR if running on a cluster
  random.seed()
  rand_dir_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
  log_dir = os.environ.get('SLURM_TMPDIR') or './runs'
  log_dir = os.path.join(log_dir, rand_dir_name)
  print(f'Logging to {log_dir}')

  os.makedirs(log_dir, exist_ok=True)
  logger = SummaryWriter(log_dir=log_dir)

  logger.add_text('config', str(config))


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