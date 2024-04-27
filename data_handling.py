import os
import requests
from typing import Any, Dict, List

from datasets import load_dataset
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


TINY_STORIES_LINK = 'https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt'


class TextDataset(Dataset):
  """BookCorpus dataset."""

  def __init__(self, hf_dataset, tokenizer):
    """
    Args:
      hf_dataset: HuggingFace dataset.
      tokenizer: Tokenizer.
    """
    self.hf_dataset = hf_dataset
    self.tokenizer = tokenizer

  def __len__(self):
    return len(self.hf_dataset)

  def __getitem__(self, idx):
    text = self.hf_dataset[idx]['text']
    tokens = self.tokenizer(text)
    input_ids, labels = tokens[0][:-1], tokens[0][1:]
    return {'input_ids': input_ids, 'labels': labels}

def download_datafile(link):
  filename = link.split('/')[-1]
  if os.path.exists(os.path.join('data/', filename)):
    return
  
  os.makedirs('data', exist_ok=True)
  r = requests.get(link, allow_redirects=True)
  open(os.path.join('data/', filename), 'wb').write(r.content)

  return os.path.exists(os.path.join('data/', filename))

def datafile_exists(link):
  filename = link.split('/')[-1]
  return os.path.exists(os.path.join('data/', filename))

def prepare_data(config):
  train_config = config.trainer

  if config.system.dataset == 'tinystories':

    if not datafile_exists(TINY_STORIES_LINK):
      if not download_datafile(TINY_STORIES_LINK):
        raise Exception('Could not download TinyStories dataset!')

    raw_dataset = load_dataset('text', data_files={'train': f'data/{TINY_STORIES_LINK.split("/")[-1]}'},
                               split='train', cache_dir='data', ignore_verifications=True)
  else:
    raw_dataset = load_dataset(config.system.dataset)
    raw_dataset = raw_dataset['train']

  # Use only a portion of the dataset if specified
  if train_config.max_samples is not None \
     and train_config.max_samples < len(raw_dataset):
    raw_dataset = raw_dataset.select(range(train_config.max_samples))

  # Split validation set from train set
  dataset = raw_dataset.train_test_split(
    test_size=train_config.valid_split, shuffle=True, seed=config.system.seed)
  dataset['val'] = dataset['test']
  del dataset['test']

  # Split test set from train set
  test_size = train_config.test_split / (1 - train_config.valid_split)
  dataset.update(dataset['train'].train_test_split(
    test_size=test_size, shuffle=True, seed=config.system.seed))
  
  return dataset


class SupervisedDataCollator:
  """Simple data collator for language modeling with padding."""
  def __init__(self, tokenizer):
    self.pad_token_id = tokenizer.pad_token_id

  def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    bsz = len(features)
    max_length = max(len(x['input_ids']) for x in features)

    input_ids = torch.full((bsz, max_length), self.pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_length), -1, dtype=torch.long)

    for i in range(bsz):
      input_ids[i, :len(features[i]['input_ids'])] = features[i]['input_ids']
      labels[i, :len(features[i]['labels'])] = features[i]['labels']

    return dict(
        input_ids = input_ids,
        labels = labels,
      )


def prepare_dataloaders(dataset, tokenizer, config):
  dataloaders = {}
  for split in ('train', 'test', 'val'):
    dataset_split = TextDataset(dataset[split], tokenizer)

    # setup the dataloader
    train_loader = DataLoader(
      dataset_split,
      # sampler=torch.utils.data.RandomSampler(
      #   dataset_split, replacement=True, num_samples=int(1e10)),
      shuffle = True if split == 'train' else False,
      pin_memory = True,
      batch_size = config.trainer.batch_size,
      num_workers = config.trainer.num_workers,
      collate_fn = SupervisedDataCollator(tokenizer),
    )

    dataloaders[split] = train_loader

  return dataloaders