### Experiment Set 2 ###
#
# The purpose of this file is to run experiments to 


import argparse
import copy
from typing import List

from datasets import load_dataset
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer
from tqdm import tqdm
import wandb

from model import MultiLayerRNN
from tokenization import CharacterTokenizer


# Repetitive phrases to remove from the TinyStories dataset
REMOVAL_PHRASES = [
    'Once upon a time, ',
    'One day, ',
]


def collate_batch(batch, tokenizer, max_length):
    """
    Tokenizes and pads batch of text to longest sequence with ignored padding labels
    
    Args:
        batch: List of examples from dataset
        tokenizer: GPT2 tokenizer instance
        max_length: Maximum sequence length
        
    Returns:
        Dictionary with padded input_ids, labels and attention mask tensors
    """
    texts = []
    for example in batch:
        text = example['text']
        for phrase in REMOVAL_PHRASES:
            text = text.replace(phrase, '')
        texts.append(tokenizer.bos_token + text + tokenizer.eos_token)
    
    # First tokenize without padding
    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length + 1,
        return_tensors=None  # Return list of token ids
    )

    input_ids = encoded['input_ids']
    labels = copy.deepcopy(input_ids)
    
    input_ids = [torch.tensor(x, dtype=torch.long) for x in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

    labels = [torch.tensor(x, dtype=torch.long) for x in labels]
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': input_ids[:, :-1],
        'labels': labels[:, 1:],
        'attention_mask': input_ids.ne(tokenizer.pad_token_id)[:, :-1]
    }


def prepare_dataloaders(tokenizer, args):
    # Load dataset
    dataset = load_dataset('roneneldan/TinyStories')
    dataset['train'] = dataset['train'].select(range(100000))

    # Split into train and validation
    val_size = min(1000, int(len(dataset['train']) * 0.1))
    train_size = len(dataset['train']) - val_size
    train_dataset, val_dataset = random_split(
        dataset['train'], 
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_batch(b, tokenizer, args.max_length)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_batch(b, tokenizer, args.max_length)
    )

    return train_loader, val_loader


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-layer RNN model on the TinyStories dataset")
    parser.add_argument('--d_model', type=int, default=512,
                        help='Hidden dimension size of the model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for truncation')
    parser.add_argument('--expansion_factor', type=float, default=1.0,
                        help='Factor to expand hidden dimension in the GRU layers')
    parser.add_argument('--repeat_sequence', action='store_true', default=False,
                        help='Change the task from next-token prediction to the copying task')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = CharacterTokenizer()

    train_loader, val_loader = prepare_dataloaders(tokenizer, args)

    # Initialize model
    model = MultiLayerRNN(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
        use_min_gru=False,
        n_heads=1,
    ).to(device)

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    
    # Print config
    print('\nConfig:')
    for k, v in vars(args).items():
        print(f'{k}: {v}')

    # Print model architecture
    print('\nModel architecture:')
    print(model)
    
    # Print and log numbers of parameters in the model
    print(f'Number of parameters in the model: {sum(p.numel() for p in model.parameters())}')
    if args.use_wandb:
        args.num_params = sum(p.numel() for p in model.parameters())

    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project='story-mingru-testing-v2', config=args)

    ### Start Training ###

    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        
        batch_losses = []
        batch_accuracies = []
        
        progress = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for idx, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['labels'].to(device)

            if args.repeat_sequence:
                input_ids = input_ids.repeat(1, 2)
                target_ids = target_ids.repeat(1, 2)
                target_ids[:, :len(target_ids) // 2] = -100 # Only predict second half of the sequence

            logits, _ = model(input_ids)


            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy ignoring padding tokens
            mask = (target_ids != -100)
            correct = (logits.argmax(dim=-1) == target_ids) * mask
            accuracy = correct.sum().float() / mask.sum()
            
            batch_losses.append(loss.item())
            batch_accuracies.append(accuracy.item())

            progress.set_postfix({'loss': loss.item(), 'accuracy': accuracy.item()})
            
            if args.use_wandb and idx % 20 == 0:
                wandb.log({
                    'train_loss': torch.tensor(batch_losses).mean().item(),
                    'train_accuracy': torch.tensor(batch_accuracies).mean().item(),
                    'epoch': epoch,
                    'train_step': idx + epoch * len(train_loader),
                })
                batch_losses = []
                batch_accuracies = []

        torch.cuda.empty_cache()
                
        train_loss = total_loss / len(train_loader)
        
        # Evaluate
        if epoch % args.eval_every == 0:
            model.eval()
            total_loss = 0
            total_accuracy = 0
            
            for batch in tqdm(val_loader, desc='Evaluating'):
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['labels'].to(device)
                
                
                if args.repeat_sequence:
                    input_ids = input_ids.repeat(1, 2)
                    target_ids = target_ids.repeat(1, 2)
                    target_ids[:, :len(target_ids) // 2] = -100 # Only predict second half of the sequence

                with torch.no_grad():
                    logits, _ = model(input_ids)
                
                
                # Calculate loss
                loss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    target_ids.reshape(-1),
                    ignore_index=-100,
                )
                
                # Calculate accuracy ignoring padding tokens
                mask = (target_ids != -100)
                correct = (logits.argmax(dim=-1) == target_ids) * mask
                accuracy = correct.sum().float() / mask.sum()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                
            val_loss = total_loss / len(val_loader)
            val_accuracy = total_accuracy / len(val_loader)
            
            print(f'\nEpoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {val_accuracy:.4f}')
            
            if args.use_wandb:
                wandb.log({
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                    'epoch': epoch
                })