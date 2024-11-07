### Experiment Set 3 ###
#
# The purpose of this file is to run experiments to 
#
#
# Ran experiments with:
#
#   Baselines:
#     python run_experiments_v2.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --epochs=4 --use_wandb
#     python run_experiments_v2.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --epochs=4 --full_sequence --use_wandb
#
#   Memory integration methods:
#     python run_experiments_v2.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --integrate_memory --epochs=4 --integrate_memory_version=0 --use_wandb
#     python run_experiments_v2.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --integrate_memory --epochs=4 --integrate_memory_version=1 --use_wandb
#     python run_experiments_v2.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --integrate_memory --epochs=4 --integrate_memory_version=2 --use_wandb
#     python run_experiments_v2.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --integrate_memory --epochs=4 --integrate_memory_version=3 --use_wandb
# 
# python run_experiments_v3.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --epochs=1 --multiple_memories --memory_pool_size=1 --use_wandb
# python run_experiments_v3.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --epochs=1 --multiple_memories --memory_pool_size=2 --use_wandb
# python run_experiments_v3.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --epochs=1 --multiple_memories --memory_pool_size=4 --use_wandb
# python run_experiments_v3.py --batch_size=64 --d_model=768 --expansion_factor=2 --max_length=128 --epochs=1 --multiple_memories --memory_pool_size=8 --use_wandb


import argparse
import copy
import math

from datasets import load_dataset
from einops import rearrange
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import wandb

from model import BatchLinear, MultiLayerRNN
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
        texts.append(text)
    
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
    dataset['train'] = dataset['train'].select(range(1000000))

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


class MultiLayerRNNWithAttn(MultiLayerRNN):
    
    def __init__(self, *args, key_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Create attention layers for each GRU layer
        key_dim = key_dim or self.d_gru_hidden
        self.query_proj = BatchLinear(self.d_gru_hidden, key_dim, len(self.layers))
        self.key_proj = BatchLinear(self.d_gru_hidden, key_dim, len(self.layers))
        
        # Scale factor for attention
        self.scale = math.sqrt(key_dim)
        
        self.mod_proj = BatchLinear(2 * self.d_gru_hidden, self.d_gru_hidden, len(self.layers))
        self.alpha_proj = BatchLinear(2 * self.d_gru_hidden, self.d_gru_hidden, len(self.layers))
        
    def query_memories(self, current_state: torch.Tensor, memories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Integrate memories with current state using attention mechanism.
        
        Args:
            current_state: Current hidden state tensor of shape (batch_size, n_gru_layers, hidden_dim)
            memories: Memory tensor of shape (batch_size, n_memories, n_gru_layers, hidden_dim)
            
        Returns:
            tuple of:
                - Mixed memories tensor of shape (batch_size, n_gru_layers, hidden_dim)
                - Attention weights of shape (batch_size, n_gru_layers, n_memories)
        """        
        # Compute queries from current state
        # Shape: (batch_size, n_layers, hidden_dim)
        queries = self.query_proj(current_state)
        
        # Compute keys from memories
        # Shape: (batch_size, n_memories, n_layers, hidden_dim)
        keys = self.key_proj(memories)
        
        # Compute attention scores
        # (batch_size, n_layers, n_memories, hidden_dim) @ (batch_size, n_layers, hidden_dim, 1)
        # Shape: (batch_size, n_layers, n_memories)
        scores = (keys.transpose(1, 2) @ queries.unsqueeze(3)).squeeze(3) / self.scale
        
        if args.sparse_attention:
            # Select only one memory (i.e. one-hot vectors)
            # Shape: (batch_size, n_layers, n_memories)
            attn_weights = F.gumbel_softmax(scores, dim=2, hard=True)
        else:
            # Apply softmax to get attention weights
            # Shape: (batch_size, n_layers, n_memories) 
            attn_weights = torch.softmax(scores, dim=2)

        # Mix memories using attention weights
        # Shape: (batch_size, n_layers, hidden_dim)
        mixed_memories = attn_weights.permute(0, 2, 1).unsqueeze(3) * memories
        mixed_memories = mixed_memories.sum(dim=1)
        
        return mixed_memories, attn_weights
    
    def integrate_memories(self, current_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        """Interpolate between current state and memory state."""
        alpha_inputs = torch.cat([current_states, memory_states], dim=2)
        alpha = torch.sigmoid(self.alpha_proj(alpha_inputs))
        modified_states = alpha * memory_states + (1 - alpha) * current_states
        return modified_states


def compute_losses_and_metrics(model, input_ids, target_ids, args):
    batch_size = input_ids.shape[0]

    memory_input_ids, query_input_ids, prediction_input_ids = torch.tensor_split(input_ids, [input_ids.shape[1]//2, 3*input_ids.shape[1]//4], dim=1)
    memory_target_ids, query_target_ids, prediction_target_ids = torch.tensor_split(target_ids, [target_ids.shape[1]//2, 3*target_ids.shape[1]//4], dim=1)

    ### Form and train memories ###
    
    _, memory_hidden_states = model(memory_input_ids)
    
    if args.memory_recon_loss:
        bos_tokens = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).repeat(batch_size, 1).to(device)
        memory_input_ids = torch.cat([bos_tokens, memory_input_ids], dim=1)
        ignore_tokens = torch.tensor([[-100]], dtype=torch.long).repeat(batch_size, 1).to(device)
        memory_target_ids = torch.cat([ignore_tokens, memory_target_ids], dim=1)
        
        memory_logits, _ = model(memory_input_ids, memory_hidden_states)

        recon_loss = nn.functional.cross_entropy(
            memory_logits.view(-1, memory_logits.size(-1)), 
            memory_target_ids.reshape(-1),
            ignore_index=-100,
        )
    else:
        recon_loss = torch.tensor(float('nan'))

    ### Use query states to retrieve matching memory states ###
        
    _, query_states = model(query_input_ids)
    
    
    if args.multiple_memories and args.memory_pool_size >= 1:
        memory_pool = rearrange(memory_hidden_states, 'l b 1 d -> 1 b l d').repeat(batch_size, 1, 1, 1)
        
        # Select indices ([0, i:i+args.memory_pool_size], [1, i:i+args.memory_pool_size], ...)
        dim1_indices =  torch.arange(batch_size).unsqueeze(1)
        dim2_indices = ((
            torch.arange(args.memory_pool_size).unsqueeze(0).repeat(batch_size, 1) \
            + torch.arange(batch_size).unsqueeze(1)
        ) % batch_size)
        memory_pool = memory_pool[dim1_indices, dim2_indices]
    else:
        memory_pool = rearrange(memory_hidden_states, 'l b 1 d -> b 1 l d')

    retrieved_states, attn_weights = model.query_memories(
        rearrange(query_states, 'l b 1 d -> b l d'), memory_pool)
    
    if args.multiple_memories:
        correct_weights = torch.zeros((batch_size, len(model.layers)))
        attn_accuracy = (attn_weights.detach().cpu().argmax(dim=2) == correct_weights).float().mean()
    else:
        attn_accuracy = torch.tensor(float('nan'))

    integrated_states = model.integrate_memories(
        rearrange(query_states, 'l b 1 d -> b l d'), retrieved_states)
    integrated_states = rearrange(integrated_states, 'b l d -> l b 1 d')
        

    ### Use retrieved states to predict next token ###
    
    prediction_logits, _ = model(prediction_input_ids, integrated_states)
    
    prediction_loss = nn.functional.cross_entropy(
        prediction_logits.view(-1, prediction_logits.size(-1)), 
        prediction_target_ids.reshape(-1),
        ignore_index=-100,
    )
    
    combined_loss = recon_loss + prediction_loss
    
    # Calculate accuracy ignoring padding tokens
    mask = (prediction_target_ids != -100)
    correct = (prediction_logits.argmax(dim=-1) == prediction_target_ids) * mask
    accuracy = correct.sum().float() / mask.sum()
            
    return {
        'combined_loss': combined_loss,
        'recon_loss': recon_loss,
        'prediction_loss': prediction_loss,
        'attn_accuracy': attn_accuracy,
        'prediction_accuracy': accuracy,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-layer RNN model on the TinyStories dataset")
    parser.add_argument('--d_model', type=int, default=512,
                        help='Hidden dimension size of the model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=500,
                        help='Run validation every N training steps')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--max_length', type=int, default=128,
                        help='Maximum sequence length for truncation')
    parser.add_argument('--expansion_factor', type=float, default=1.0,
                        help='Factor to expand hidden dimension in the GRU layers')
    parser.add_argument('--multiple_memories', action='store_true', default=False,
                        help='Model must choose which hidden state to use from a batch of hidden states.'
                             'Exactly one of the hidden states from the pool of \{batch_size\} hidden states'
                             'will be useful.')
    parser.add_argument('--memory_recon_loss', action='store_true', default=False,
                        help='Use a reconstruction loss on the memory states.')
    parser.add_argument('--memory_pool_size', type=int, default=None,
                        help='Number of memories to attend over. Default is batch size.')
    parser.add_argument('--sparse_attention', action='store_true', default=False,
                        help='Use a sparse attention mechanism to attend over the memory pool.')
    
    args = parser.parse_args()
    args.memory_pool_size = args.memory_pool_size or args.batch_size
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader, val_loader = prepare_dataloaders(tokenizer, args)
    
    model_kwargs = dict(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
        use_min_gru=False,
        n_heads=1,
        mlp_ratio=4.0,
        key_dim=args.d_model // 4
    )

    # Initialize model
    model = MultiLayerRNNWithAttn(**model_kwargs).to(device)

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
        wandb.init(project='story-gru-testing-v3', config=args)

    ### Start Training ###

    total_steps = 0
    for epoch in range(args.epochs):
        # Train
        model.train()
        total_loss = 0
        
        batch_memory_losses = []
        batch_prediction_losses = []
        batch_accuracies = []
        batch_attn_accuracies = []
        progress = tqdm(train_loader, desc=f'Training Epoch {epoch}')
        for idx, batch in enumerate(progress):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['labels'].to(device)
            
            losses_and_metrics = compute_losses_and_metrics(model, input_ids, target_ids, args)
    
            optimizer.zero_grad()
            losses_and_metrics['combined_loss'].backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += losses_and_metrics['combined_loss'].item()
            total_steps += 1
            
            batch_memory_losses.append(losses_and_metrics['recon_loss'].item())
            batch_prediction_losses.append(losses_and_metrics['prediction_loss'].item())
            batch_accuracies.append(losses_and_metrics['prediction_accuracy'].item())
            if args.multiple_memories:
                batch_attn_accuracies.append(losses_and_metrics['attn_accuracy'].item())

            progress.set_postfix({'recon_loss': losses_and_metrics['recon_loss'].item(), 
                                  'prediction_loss': losses_and_metrics['prediction_loss'].item(), 
                                  'prediction_accuracy': losses_and_metrics['prediction_accuracy'].item(), 
                                  'attn_accuracy': losses_and_metrics['attn_accuracy'].item()})
            
            if args.use_wandb and idx % 20 == 0:
                wandb.log({
                    'train_memory_loss': torch.tensor(batch_memory_losses).mean().item(),
                    'train_prediction_loss': torch.tensor(batch_prediction_losses).mean().item(),
                    'train_accuracy': torch.tensor(batch_accuracies).mean().item(),
                    'train_attn_accuracy': torch.tensor(batch_attn_accuracies).mean().item(),
                    'epoch': epoch,
                    'train_step': total_steps,
                })
                batch_memory_losses = []
                batch_prediction_losses = []
                batch_accuracies = []
                batch_attn_accuracies = []
                    

            # Evaluate based on steps instead of epochs
            if total_steps % args.eval_every == 0:
                model.eval()
                total_loss = 0
                val_memory_losses = []
                val_prediction_losses = []
                val_accuracies = []
                val_attn_accuracies = []
                
                train_loss = total_loss / (idx + 1)  # Current epoch's average loss
                
                for batch in tqdm(val_loader, desc='Evaluating'):
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['labels'].to(device)
                    
                    with torch.no_grad():
                        losses_and_metrics = compute_losses_and_metrics(model, input_ids, target_ids, args)
                    
                    total_loss += losses_and_metrics['combined_loss'].item()
                    
                    val_memory_losses.append(losses_and_metrics['recon_loss'].item())
                    val_prediction_losses.append(losses_and_metrics['prediction_loss'].item())
                    val_accuracies.append(losses_and_metrics['prediction_accuracy'].item())
                    if args.multiple_memories:
                        val_attn_accuracies.append(losses_and_metrics['attn_accuracy'].item())
                    
                val_loss = total_loss / len(val_loader)
                val_accuracy = torch.tensor(val_accuracies).mean().item()
                val_memory_loss = torch.tensor(val_memory_losses).mean().item()
                val_prediction_loss = torch.tensor(val_prediction_losses).mean().item()
                val_attn_accuracy = torch.tensor(val_attn_accuracies).mean().item() if val_attn_accuracies else 0.0
                
                print(f'\nStep {total_steps}:')
                print(f'Train Loss: {train_loss:.4f}')
                print(f'Val Loss: {val_loss:.4f}')
                print(f'Val Memory Loss: {val_memory_loss:.4f}')
                print(f'Val Prediction Loss: {val_prediction_loss:.4f}')
                print(f'Val Accuracy: {val_accuracy:.4f}')
                if args.multiple_memories:
                    print(f'Val Attention Accuracy: {val_attn_accuracy:.4f}')
                
                if args.use_wandb:
                    wandb.log({
                        'val_loss': val_loss,
                        'val_memory_loss': val_memory_loss,
                        'val_prediction_loss': val_prediction_loss,
                        'val_accuracy': val_accuracy,
                        'val_attn_accuracy': val_attn_accuracy,
                        'epoch': epoch,
                        'step': total_steps
                    })
                    
                model.train()
                torch.cuda.empty_cache()