import contextlib
import copy
import sys
sys.path.append('minGPT/')

import numpy as np
import torch
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from mingpt.model import GPT


EPSILON = 1e-7


# def eval_model(model, dataloader, train_config):
#   model.eval()
#   total_loss = 0
#   total_count = 0

#   with torch.no_grad():
      
#     # Calculate eval loss
#     for batch in dataloader:
#       batch = [t.to(train_config.device) for t in batch]
#       x, y, mask = batch

#       # Forward the model
#       _, loss = model(x, y, loss_mask=mask)
#       total_loss += loss.item() * x.shape[0]
#       total_count += x.shape[0]

#   return total_loss / total_count


def get_mem_weights(model):
  mem_weights = []
  for block in model.transformer.h:
    for layer in block.c_fc_mem.layers:
      mem_weights.append(layer.weight.data)
  return torch.stack(mem_weights)


def train_loop(
    model,
    optimizer,
    dataloaders,
    config,
    logger = None,
    tokenizer = None):
  model.train()

  train_config = config.trainer

  # Mixed precision context manager
  mp_ctx = autocast() if train_config.fp16 else contextlib.suppress()

  fp16 = train_config.fp16
  if fp16:
    print('Using mixed precision training.')
    scaler = GradScaler()

  progress_bar = tqdm(position=0, leave=True)
  loss_hist = []
  hist_size = max(train_config.print_freq, train_config.log_freq)
  batch_iter = 0
  sample_iter = 0
  eval_loss = np.nan
  print_loss = np.nan
  gradient_accumulation_steps = train_config.grad_accumulation_steps

  # Data queue is a list of input, output sequence pairs
  data_iter = iter(dataloaders['train'])
  curr_sequences = next(data_iter)
  model.reset_memory(len(curr_sequences['input_ids']))

  epoch = 0
  while epoch < train_config.epochs:
    # If the current sequence is empty, get the next batch of sequences
    if curr_sequences['input_ids'].shape[1] == 0:
      try:
        curr_sequences = next(data_iter)
        model.reset_memory(len(curr_sequences['input_ids']))
        sample_iter += curr_sequences['input_ids'].shape[0]
      except StopIteration:
        data_iter = iter(dataloaders['train'])
        epoch += 1
        continue
      
    # Create a training batch from a portion of the current sequences
    batch = {k: v[:, :config.model.block_size].to(train_config.device) for k, v in curr_sequences.items()}
    curr_sequences = {k: v[:, config.model.block_size:] for k, v in curr_sequences.items()}

    batch_iter += 1

    # TODO: Fix this error with called even when grad accumulation steps is not 1
    model.zero_grad(set_to_none=True)

    with mp_ctx:
      _, loss = model(batch['input_ids'], batch['labels'])

    if fp16:
      scaler.scale(loss).backward()
      scaler.unscale_(optimizer)
    else:
      loss.backward()

    # Collect and log gradients and parameters as distributions
    if config.system.wandb \
        and config.system.log_model_stats \
        and batch_iter % int(train_config.log_freq * 50) == 0:
      
      import wandb

      for name, param in model.named_parameters():
        wandb.log({f'grads/{name}': wandb.Histogram(param.grad.detach().cpu().reshape(-1))})
        wandb.log({f'params/{name}': wandb.Histogram(param.detach().cpu().reshape(-1))})


    # Clip gradients
    torch.nn.utils.clip_grad_norm_(
      model.parameters(), train_config.grad_norm_clip)
    
    # Apply weight update
    if batch_iter % gradient_accumulation_steps == 0:
      if fp16:
        scaler.step(optimizer)
        scaler.update()
      else:
        optimizer.step()

    # Keep track of vars to log
    loss_hist.append(loss.item())
    loss_hist = loss_hist[-hist_size:]

    # Log loss
    if logger is not None and batch_iter % train_config.log_freq == 0:
      log_loss = np.mean(loss_hist[-train_config.log_freq:])
      logger.add_scalar('loss', log_loss, batch_iter)
      logger.add_scalar('epoch', epoch, batch_iter)
      logger.add_scalar('samples', sample_iter, batch_iter)

    # Update progress bar
    if batch_iter % train_config.print_freq == 0:
      print_loss = np.mean(loss_hist[-train_config.print_freq:])
      progress_bar.set_description(
        f'Epoch {epoch} | Samples {sample_iter:.2e} | ' + \
        f'Eval Loss {eval_loss:.3f} | Loss {print_loss:.3f}')
    progress_bar.update(1)

    # # Evaluate the model & generate examples
    # if batch_iter % train_config.eval_freq == 0:
    #   eval_loss = eval_model(
    #     model, dataloaders['val'], train_config)

    #   if tokenizer is not None:
    #     log_examples(
    #       model, dataloaders['val'], tokenizer,
    #       logger, train_config, batch_iter)
      
    #   model.train()

    #   if logger is not None:
    #     logger.add_scalar('eval_loss', eval_loss, batch_iter)
    #   progress_bar.set_description(
    #     f'Epoch {epoch} | Samples {sample_iter:.2e} | FLOPs {total_flops:.2e} | ' + \
    #     f'Eval Loss {eval_loss:.3f} | Loss {print_loss:.3f}')
    
    # if train_config.total_flops is not None \
    #     and total_flops >= train_config.total_flops:
    #   break

  progress_bar.close()

  return model, optimizer


def get_model_params(model):
    """ Returns number of parameters in model. """
    n_params = 0
    for param in model.parameters():
        n_params += np.prod(param.shape)

    return n_params
