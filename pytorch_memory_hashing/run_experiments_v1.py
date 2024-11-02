### Experiment Set 1 ###
#
# The purpose of this file is to run experiments to better understand how basic choices in architecture, hyperparameters,
# and training procedure have on the copying task. Specifically, we looked at the following:
#
# - The difference between next-token-prediction on a single sequence vs. next-token-prediction on a repeated sequence
#   Note that the latter is simply the copying task, and the model should achieve higher accuracy (it did).
#
# - The difference between using a character-level tokenizer vs. using a word-level tokenizer (GPT2Tokenizer).
#   With this limited dataset, the former performs better because the model only has to learn ~100 tokens rather than 50k.
#
# - The difference between using a GRU vs. a minGRU. The former performs better despite results in the minGRU paper that
#   show around equal performance (in single and multi-layer models). With shorter sequences like the ones tested (64 tokens),
#   the minGRU doesn't have a major performance advantage either. For these reasons, we use the GRU for further experiments.
#
# - The difference between using a single layer model vs. a multi-layer model. The multi-layer architecture is similar to
#   a transformer architecture (specifically based on Llama 3), but with GRUs instead of attention. The multi-layer model
#   performs better than the single layer model, as expected.
#
# - The effects of adding multiple "heads" to a GRU while keeping the number of parameters about constant. The multi-head
#   GRU was slower to train, but achieved slightly higher accuracy. We would need more experiments to determine if the
#   results were statistically significant. For now, the complexity does not justify the potential minor performance gain.
#
# - The effects of modifying the recurrent state (memory) with an MLP after the first part of the sequence (where the agent
#   only needs to remember the sequence). The idea is that eventually the agent will want to recall different memories for
#   different parts of the sequence and for different purposes, so it makes sense to mix the prior memory with the current
#   recurrent state in some way. This only hindered the performance on the copying task. In this case, the recurrent state
#   is only used for a single purpose, and so testing this was likely premature. It should be revisited once we move to
#   more complex tasks.


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

from model import MultiLayerRNN, StoryNetwork


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
    texts = [tokenizer.bos_token + example['text'] + tokenizer.eos_token for example in batch]
    
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


def sigmoid_linear(x):
    """x < 0 -> sigmoid(x), x >= 0 -> x + 0.5"""
    return torch.where(x >= 0, x + 0.5, x.sigmoid())


def inverse_sigmoid_linear(x):
    """Inverse sigmoid linear function"""
    return torch.where(x >= 0.5, x - 0.5, torch.log(x / (1 - x)))
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--expansion_factor', type=float, default=1.0)
    parser.add_argument('--use_normal_gru', dest='use_min_gru', action='store_false')
    parser.add_argument('--repeat_sequence', action='store_true', default=False)
    parser.add_argument('--two_step_forward', action='store_true', default=False)
    parser.add_argument('--modify_memories', action='store_true', default=False)
    parser.add_argument('--model_type', type=str, default='multi_layer')
    
    return parser.parse_args()


class CharacterTokenizer:
    """Simple character-level tokenizer that treats each character as a token."""
    
    def __init__(self):
        # Create vocabulary from printable ASCII characters
        self.chars = [chr(i) for i in range(32, 127)] + ['\n']
        self.vocab = {char: i for i, char in enumerate(self.chars)}
        self.vocab['<pad>'] = len(self.vocab)  # Add padding token
        self.vocab['<bos>'] = len(self.vocab)  # Add beginning of sequence token
        self.vocab['<eos>'] = len(self.vocab)  # Add end of sequence token
        
        # Create reverse mapping
        self.id_to_char = {i: char for char, i in self.vocab.items()}
        
        # Special tokens
        self.pad_token = '<pad>'
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token_id = self.vocab[self.pad_token]
    
    def __len__(self) -> int:
        return len(self.vocab)
    
    def __call__(self, texts: List[str], truncation: bool = False, max_length: int = None, **kwargs) -> dict:
        """
        Tokenize a batch of texts into token ids.
        
        Args:
            texts: List of strings to tokenize
            truncation: Whether to truncate sequences longer than max_length
            max_length: Maximum sequence length (including special tokens)
            
        Returns:
            Dictionary with input_ids for each text
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_ids = []
        for text in texts:
            ids = [self.vocab.get(c, self.vocab['<pad>']) for c in text[:max_length]]
            all_ids.append(ids)
            
        return {'input_ids': all_ids}


if __name__ == '__main__':
    args = parse_args()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    # # Load tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token

    tokenizer = CharacterTokenizer()

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
    

    if args.model_type == 'multi_layer':
        model_cls = MultiLayerRNN
    elif args.model_type == 'story':
        model_cls = StoryNetwork


    class MemoryNetwork(model_cls):
        def __init__(self, *args, **kwargs):
            if 'expansion_factor' not in kwargs:
                kwargs['expansion_factor'] = 2.0
            super().__init__(*args, **kwargs)
            self.memory_dim = int(self.d_model * kwargs['expansion_factor'])
            mlp_ratio = getattr(self, 'mlp_ratio', 4.0)
            
            hidden_dim = int(self.memory_dim * mlp_ratio)
            self.gate_proj = nn.Linear(self.memory_dim, hidden_dim, bias=False)
            self.up_proj = nn.Linear(self.memory_dim, hidden_dim, bias=False)
            self.down_proj = nn.Linear(hidden_dim, self.memory_dim, bias=False)
            self.act_fn = nn.SiLU()
        
        def modify_memory_states(self, memory_states: List[torch.Tensor]) -> torch.Tensor:
            initial_states = torch.stack(memory_states)
            
            # MLP with gating
            gate_output = self.act_fn(self.gate_proj(initial_states))
            up_output = self.up_proj(initial_states)
            modified_states = self.down_proj(gate_output * up_output)
            
            # minGRU needs to have a positive recurrent state
            if self.use_min_gru:
                modified_states = inverse_sigmoid_linear(initial_states) + modified_states
                modified_states = sigmoid_linear(modified_states)
            else:
                modified_states = torch.atanh(initial_states) + modified_states
                modified_states = torch.tanh(modified_states)
            
            return modified_states # sigmoid_linear(self.memory_integration_layer(future_state))


    parent_model_cls = MemoryNetwork if args.modify_memories else model_cls

    # Initialize model
    model = parent_model_cls(
        vocab_size=len(tokenizer),
        d_model=args.d_model,
        expansion_factor=args.expansion_factor,
        use_min_gru=args.use_min_gru,
        n_heads=args.n_heads,
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

            if args.repeat_sequence and args.two_step_forward:
                _, prev_hidden_state = model(input_ids[:, :len(input_ids) // 2])
                
                if args.modify_memories:
                    prev_hidden_state = model.modify_memory_states(prev_hidden_state)
                    
                logits, _ = model(input_ids[:, len(input_ids) // 2:], prev_hidden_state)
                target_ids = target_ids[:, len(target_ids) // 2:]
            else:
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
                    if args.repeat_sequence and args.two_step_forward:
                        _, prev_hidden_state = model(input_ids[:, :len(input_ids) // 2])
                        
                        if args.modify_memories:
                            prev_hidden_state = model.modify_memory_states(prev_hidden_state)
                            
                        logits, _ = model(input_ids[:, len(input_ids) // 2:], prev_hidden_state)
                        target_ids = target_ids[:, len(target_ids) // 2:]
                    else:
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