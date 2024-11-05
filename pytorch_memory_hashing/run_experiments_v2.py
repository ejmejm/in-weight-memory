### Experiment Set 2 ###
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


import argparse
import copy
import math

from datasets import load_dataset
from einops import rearrange
from transformers import GPT2Tokenizer
import torch
import torch.nn as nn
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


class MultiLayerRNNWithAttn(MultiLayerRNN):
    
    def __init__(self, *args, key_dim=None, integrate_memory_version='0', **kwargs):
        super().__init__(*args, **kwargs)
        # Create attention layers for each GRU layer
        key_dim = key_dim or self.d_gru_hidden
        self.query_proj = BatchLinear(self.d_gru_hidden, key_dim, len(self.layers))
        self.key_proj = BatchLinear(self.d_gru_hidden, key_dim, len(self.layers))
        
        # Scale factor for attention
        self.scale = math.sqrt(key_dim)
        
        # mlp_ratio = getattr(self, 'mlp_ratio', 4.0)
        
        # hidden_dim = int(self.d_gru_hidden * mlp_ratio)
        # self.gate_proj = BatchLinear(2 * self.d_gru_hidden, hidden_dim, len(self.layers), bias=False)
        # self.up_proj = BatchLinear(2 * self.d_gru_hidden, hidden_dim, len(self.layers), bias=False)
        # self.down_proj = BatchLinear(hidden_dim, self.d_gru_hidden, len(self.layers), bias=False)
        # self.act_fn = nn.SiLU()
        
        self.mod_proj = BatchLinear(2 * self.d_gru_hidden, self.d_gru_hidden, len(self.layers))
        self.alpha_proj = BatchLinear(2 * self.d_gru_hidden, self.d_gru_hidden, len(self.layers))
        
        integrate_memories_version_map = {
            '0': self.integrate_memories_v0,
            '1': self.integrate_memories_v1,
            '2': self.integrate_memories_v2,
            '3': self.integrate_memories_v3,
        }
        print(f'Using integrate memories version {integrate_memory_version}')
        self.integrate_memories = integrate_memories_version_map[integrate_memory_version]
        
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
        
        # Apply softmax to get attention weights
        # Shape: (batch_size, n_layers, n_memories) 
        attn_weights = torch.softmax(scores, dim=2)
        
        # Mix memories using attention weights
        # Shape: (batch_size, n_layers, hidden_dim)
        mixed_memories = attn_weights.permute(0, 2, 1).unsqueeze(3) * memories
        mixed_memories = mixed_memories.sum(dim=1)
        
        return mixed_memories, attn_weights
        
    # def integrate_memories(self, current_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
    #     """Integrate memories with current state using MLP with gating.
        
    #     Args:
    #         current_states: Current hidden state tensor of shape (batch_size, n_gru_layers, hidden_dim)
    #         memory_states: Memory tensor of shape (batch_size, n_gru_layers, hidden_dim)
            
    #     Returns:
    #         Modified current states tensor of shape (batch_size, n_gru_layers, hidden_dim)
    #     """
    #     input_states = torch.cat([current_states, memory_states], dim=2)
        
    #     # MLP with gating
    #     gate_output = self.act_fn(self.gate_proj(input_states))
    #     up_output = self.up_proj(input_states)
    #     state_modifications = self.down_proj(gate_output * up_output)

    #     modified_states = torch.atanh(current_states) + state_modifications
    #     modified_states = torch.tanh(modified_states)
        
    #     return modified_states

    def integrate_memories_v0(self, current_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        input_states = torch.cat([current_states, memory_states], dim=2)
        
        modified_state = self.mod_proj(input_states)
        alpha_inputs = torch.cat([current_states, modified_state], dim=2)
        alpha = torch.sigmoid(self.alpha_proj(alpha_inputs))
        
        # Interpolate between current state and modified state
        modified_states = alpha * torch.tanh(modified_state) + (1 - alpha) * current_states
        
        return modified_states
    
    def integrate_memories_v1(self, current_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        """Create a new state with a linear projection, then interpolate with the current state."""
        input_states = torch.cat([current_states, memory_states], dim=2)
        
        modified_state = torch.tanh(self.mod_proj(input_states))
        alpha_inputs = torch.cat([current_states, modified_state], dim=2)
        alpha = torch.sigmoid(self.alpha_proj(alpha_inputs))
        
        # Interpolate between current state and modified state
        modified_states = alpha * modified_state + (1 - alpha) * current_states
        
        return modified_states

    # def integrate_memories_v2(self, current_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
    #     input_states = torch.cat([current_states, memory_states], dim=2)
        
    #     state_modifications = self.mod_proj(input_states)
    #     modified_states = torch.atanh(torch.clamp(current_states, -0.999999, 0.999999)) + state_modifications
    #     modified_states = torch.tanh(modified_states)
        
    #     alpha_inputs = torch.cat([current_states, modified_states], dim=2)
    #     alpha = torch.sigmoid(self.alpha_proj(alpha_inputs))
        
    #     # Interpolate between current state and modified state
    #     modified_states = alpha * torch.tanh(modified_states) + (1 - alpha) * current_states
        
    #     return modified_states
    
    def integrate_memories_v2(self, current_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        """Unscale and modify the current state with a residual connection before passing back through a tanh."""
        current_state_logits = torch.atanh(torch.clamp(current_states, -0.999999, 0.999999))
        mem_state_logits = torch.atanh(torch.clamp(memory_states, -0.999999, 0.999999))
        input_states = torch.cat([current_state_logits, mem_state_logits], dim=2)
        
        state_modifications = self.mod_proj(input_states)
        gate_inputs = torch.cat([current_state_logits, state_modifications], dim=2)
        gate_values = torch.sigmoid(self.alpha_proj(gate_inputs))

        modified_states = current_state_logits + gate_values * state_modifications
        modified_states = torch.tanh(modified_states)
        
        return modified_states
    
    def integrate_memories_v3(self, current_states: torch.Tensor, memory_states: torch.Tensor) -> torch.Tensor:
        """Interpolate between current state and memory state."""
        alpha_inputs = torch.cat([current_states, memory_states], dim=2)
        alpha = torch.sigmoid(self.alpha_proj(alpha_inputs))
        modified_states = alpha * memory_states + (1 - alpha) * current_states
        return modified_states


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
    parser.add_argument('--full_sequence', action='store_true', default=False,
                        help='Use the full sequence with no need for memory')
    parser.add_argument('--integrate_memory', action='store_true', default=False,
                        help='Model must choose which hidden state to use from a batch of hidden states.'
                             'Exactly one of the hidden states from the pool of \{batch_size\} hidden states'
                             'will be useful.')
    parser.add_argument('--integrate_memory_version', type=str, default='0',
                        help='Version of the integrate memories function to use.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # tokenizer = CharacterTokenizer()

    train_loader, val_loader = prepare_dataloaders(tokenizer, args)

    model_cls = MultiLayerRNNWithAttn if args.integrate_memory else MultiLayerRNN
    
    model_kwargs = dict(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
        use_min_gru=False,
        n_heads=1,
        mlp_ratio=4.0,
    )
    
    if args.integrate_memory:
        model_kwargs['key_dim'] = args.d_model // 4
        model_kwargs['integrate_memory_version'] = args.integrate_memory_version

    # Initialize model
    model = model_cls(**model_kwargs).to(device)

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
        wandb.init(project='story-gru-testing-v2', config=args)

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
            batch_size = input_ids.shape[0]

            memory_input_ids, query_input_ids, prediction_input_ids = torch.tensor_split(input_ids, [input_ids.shape[1]//2, 3*input_ids.shape[1]//4], dim=1)
            memory_target_ids, query_target_ids, prediction_target_ids = torch.tensor_split(target_ids, [target_ids.shape[1]//2, 3*target_ids.shape[1]//4], dim=1)
            
            if args.integrate_memory:
                ### Form and train memories ###
                
                # TODO: Consider prepending a memory token to the memory input ids
                _, memory_hidden_states = model(memory_input_ids)
                
                # bos_tokens = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).repeat(batch_size, 1).to(device)
                # memory_input_ids = torch.cat([bos_tokens, memory_input_ids], dim=1)
                # ignore_tokens = torch.tensor([[-100]], dtype=torch.long).repeat(batch_size, 1).to(device)
                # memory_target_ids = torch.cat([ignore_tokens, memory_target_ids], dim=1)
                
                # memory_logits, _ = model(memory_input_ids, memory_hidden_states)

                # recon_loss = nn.functional.cross_entropy(
                #     memory_logits.view(-1, memory_logits.size(-1)), 
                #     memory_target_ids.reshape(-1),
                #     ignore_index=-100,
                # )
                recon_loss = torch.tensor(0.0)
                attn_accuracy = torch.tensor(0.0)

                ### Use query states to retrieve matching memory states ###
                    
                _, query_states = model(query_input_ids)
                
                
                # retrieved_states, attn_weights = model.query_memories(
                #     rearrange(query_states, 'l b 1 d -> b l d'),
                #     # rearrange(memory_hidden_states, 'l b 1 d -> 1 b l d').repeat(batch_size, 1, 1, 1),
                #     rearrange(memory_hidden_states, 'l b 1 d -> b 1 l d'),
                # )
                
                # correct_weights = torch.arange(batch_size).unsqueeze(1).repeat(1, len(model.layers))
                # attn_accuracy = (attn_weights.detach().cpu().argmax(dim=2) == correct_weights).float().mean()
                # batch_attn_accuracies.append(attn_accuracy.item())
            
                integrated_states = model.integrate_memories(
                    rearrange(query_states, 'l b 1 d -> b l d'), rearrange(memory_hidden_states, 'l b 1 d -> b l d'))
                    # retrieved_states)
                
                integrated_states = rearrange(integrated_states, 'b l d -> l b 1 d')
                
            else:
                if args.full_sequence:
                    _, start_states = model(memory_input_ids)
                else:
                    start_states = None
                
                recon_loss = torch.tensor(0.0)
                attn_accuracy = torch.tensor(0.0)
                _, query_states = model(query_input_ids, start_states)
                integrated_states = query_states

            ### Use retrieved states to predict next token ###
            prediction_logits, _ = model(prediction_input_ids, integrated_states)
            
            prediction_loss = nn.functional.cross_entropy(
                prediction_logits.view(-1, prediction_logits.size(-1)), 
                prediction_target_ids.reshape(-1),
                ignore_index=-100,
            )
            
            combined_loss = recon_loss + prediction_loss
    
            optimizer.zero_grad()
            combined_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += combined_loss.item()
            total_steps += 1
            # Calculate accuracy ignoring padding tokens
            mask = (prediction_target_ids != -100)
            correct = (prediction_logits.argmax(dim=-1) == prediction_target_ids) * mask
            accuracy = correct.sum().float() / mask.sum()
            
            batch_memory_losses.append(recon_loss.item())
            batch_prediction_losses.append(prediction_loss.item())
            batch_accuracies.append(accuracy.item())

            progress.set_postfix({'recon_loss': recon_loss.item(), 'prediction_loss': prediction_loss.item(), 
                                'accuracy': accuracy.item(), 'attn_accuracy': attn_accuracy.item()})
            
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
                total_accuracy = 0
                val_memory_losses = []
                val_prediction_losses = []
                val_accuracies = []
                val_attn_accuracies = []
                
                train_loss = total_loss / (idx + 1)  # Current epoch's average loss
                
                for batch in tqdm(val_loader, desc='Evaluating'):
                    input_ids = batch['input_ids'].to(device)
                    target_ids = batch['labels'].to(device)
                    batch_size = input_ids.shape[0]
                    
                    # memory_input_ids, query_input_ids, prediction_input_ids = input_ids.chunk(3, dim=1)
                    # memory_target_ids, query_target_ids, prediction_target_ids = target_ids.chunk(3, dim=1)

                    memory_input_ids, query_input_ids, prediction_input_ids = torch.tensor_split(input_ids, [input_ids.shape[1]//2, 3*input_ids.shape[1]//4], dim=1)
                    memory_target_ids, query_target_ids, prediction_target_ids = torch.tensor_split(target_ids, [target_ids.shape[1]//2, 3*target_ids.shape[1]//4], dim=1)

                    with torch.no_grad():
                        if args.integrate_memory:
                            # Form memories
                            _, memory_hidden_states = model(memory_input_ids)
                            # memory_logits, _ = model(memory_input_ids, memory_hidden_states)

                            # recon_loss = nn.functional.cross_entropy(
                            #     memory_logits.view(-1, memory_logits.size(-1)), 
                            #     memory_target_ids.reshape(-1),
                            #     ignore_index=-100,
                            # )
                            recon_loss = torch.tensor(0.0)

                            # Query and retrieve memories
                            _, query_states = model(query_input_ids)
                            
                            # retrieved_states, attn_weights = model.query_memories(
                            #     rearrange(query_states, 'l b 1 d -> b l d'),
                            #     rearrange(memory_hidden_states, 'l b 1 d -> 1 b l d').repeat(batch_size, 1, 1, 1),
                            # )
                            
                            # correct_weights = torch.arange(batch_size).unsqueeze(1).repeat(1, len(model.layers))
                            # attn_accuracy = (attn_weights.cpu().argmax(dim=2) == correct_weights).float().mean()
                            # val_attn_accuracies.append(attn_accuracy.item())

                            integrated_states = model.integrate_memories(
                                rearrange(query_states, 'l b 1 d -> b l d'), rearrange(memory_hidden_states, 'l b 1 d -> b l d'))
                                # retrieved_states)
                            
                            integrated_states = rearrange(integrated_states, 'b l d -> l b 1 d')
                        else:
                            if args.full_sequence:
                                _, start_states = model(memory_input_ids)
                            else:
                                start_states = None

                            recon_loss = torch.tensor(0.0)
                            attn_accuracy = torch.tensor(0.0)
                            _, query_states = model(query_input_ids, start_states)
                            integrated_states = query_states

                        # Make predictions
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
                    
                    total_loss += combined_loss.item()
                    total_accuracy += accuracy.item()
                    
                    val_memory_losses.append(recon_loss.item())
                    val_prediction_losses.append(prediction_loss.item())
                    val_accuracies.append(accuracy.item())
                    
                val_loss = total_loss / len(val_loader)
                val_accuracy = total_accuracy / len(val_loader)
                
                print(f'\nStep {total_steps}:')
                print(f'Train Loss: {train_loss:.4f}')
                print(f'Val Loss: {val_loss:.4f}')
                print(f'Val Memory Loss: {torch.tensor(val_memory_losses).mean().item():.4f}')
                print(f'Val Prediction Loss: {torch.tensor(val_prediction_losses).mean().item():.4f}')
                print(f'Val Accuracy: {val_accuracy:.4f}')
                if args.integrate_memory:
                    print(f'Val Attention Accuracy: {torch.tensor(val_attn_accuracies).mean().item():.4f}')
                
                if args.use_wandb:
                    wandb.log({
                        'val_loss': val_loss,
                        'val_memory_loss': torch.tensor(val_memory_losses).mean().item(),
                        'val_prediction_loss': torch.tensor(val_prediction_losses).mean().item(),
                        'val_accuracy': val_accuracy,
                        'val_attn_accuracy': torch.tensor(val_attn_accuracies).mean().item() if val_attn_accuracies else 0.0,
                        'epoch': epoch,
                        'step': total_steps
                    })
                    
                model.train()
                torch.cuda.empty_cache()