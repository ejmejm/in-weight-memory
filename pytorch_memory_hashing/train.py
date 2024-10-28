import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from tqdm.auto import tqdm
import wandb
from pytorch_memory_hashing.model import StoryNetwork
import argparse
from typing import Optional
from transformers import GPT2Tokenizer


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    use_wandb: bool = False
) -> float:
    """
    Trains model for one epoch
    
    Args:
        model: The neural network
        loader: DataLoader for training data
        optimizer: The optimizer
        device: Device to train on
        epoch: Current epoch number
        use_wandb: Whether to log to wandb
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    progress = tqdm(loader, desc=f'Training Epoch {epoch}')
    for idx, batch in enumerate(progress):
        input_ids = batch['input_ids'][:, :-1].to(device)
        target_ids = batch['input_ids'][:, 1:].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target_ids.view(-1)
        )
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Update progress bar
        progress.set_postfix({'loss': loss.item()})
        
        # Log to wandb
        if use_wandb and idx % 100 == 0:
            wandb.log({
                'train_loss': loss.item(),
                'epoch': epoch
            })
            
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str
) -> float:
    """
    Evaluates model on validation set
    
    Args:
        model: The neural network
        loader: DataLoader for validation data
        device: Device to evaluate on
        
    Returns:
        Average loss on validation set
    """
    model.eval()
    total_loss = 0
    
    for batch in tqdm(loader, desc='Evaluating'):
        input_ids = batch['input_ids'][:, :-1].to(device)
        target_ids = batch['input_ids'][:, 1:].to(device)
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Calculate loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            target_ids.view(-1)
        )
        
        total_loss += loss.item()
        
    return total_loss / len(loader)


def collate_batch(batch, tokenizer):
    """
    Tokenizes and pads batch of text to longest sequence
    
    Args:
        batch: List of examples from dataset
        tokenizer: GPT2 tokenizer instance
        
    Returns:
        Dictionary with padded input_ids tensor
    """
    texts = [example['text'] for example in batch]
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,  # Set a reasonable max length
        return_tensors='pt'
    )
    return encoded


def main(args):
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project='story-mingru')
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset('roneneldan/TinyStories')
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset['train']))
    val_size = len(dataset['train']) - train_size
    train_dataset, val_dataset = random_split(
        dataset['train'], 
        [train_size, val_size]
    )
    
    # Create dataloaders with custom collate function
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_batch(b, tokenizer)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_batch(b, tokenizer)
    )
    
    # Initialize model with tokenizer vocab size
    model = StoryNetwork(
        vocab_size=len(tokenizer),
        d_model=args.d_model
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, 
            train_loader, 
            optimizer, 
            device, 
            epoch,
            args.use_wandb
        )
        
        # Evaluate
        if epoch % args.eval_every == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f'\nEpoch {epoch}:')
            print(f'Train Loss: {train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            
            if args.use_wandb:
                wandb.log({
                    'val_loss': val_loss,
                    'epoch': epoch
                })
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_size', type=int, default=50257)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--max_length', type=int, default=128)
    
    args = parser.parse_args()
    main(args)
