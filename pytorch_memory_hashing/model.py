import torch
import torch.nn as nn
from minGRU_pytorch import minGRU


class StoryNetwork(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 512,
        expansion_factor: float = 2.0,
    ):
        """
        Creates a network with FC -> minGRU -> FC -> FC architecture
        
        Args:
            vocab_size: Size of vocabulary for embedding and final layer
            d_model: Internal dimension size
            expansion_factor: Expansion factor for minGRU
            leaky_relu_slope: Negative slope for LeakyReLU activations
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pre_gru = nn.Linear(d_model, d_model)
        self.leaky_relu = nn.LeakyReLU()
        self.gru = minGRU(dim=d_model, expansion_factor=expansion_factor)
        self.post_gru1 = nn.Linear(d_model, d_model)
        self.post_gru2 = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, prev_hidden: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            prev_hidden: Optional previous hidden state
            
        Returns:
            tuple of (output logits, next hidden state)
        """
        # (batch, seq_len) -> (batch, seq_len, d_model)
        x = self.embedding(x)
        
        # Pre GRU processing
        x = self.dropout(self.pre_gru(x))
        
        # Run through GRU
        if prev_hidden is not None:
            x, next_hidden = self.gru(x, prev_hidden, return_next_prev_hidden=True)
        else:
            x = self.gru(x)
            next_hidden = None
            
        # Post GRU processing
        x = self.dropout(self.leaky_relu(self.post_gru1(x)))
        x = self.post_gru2(x)  # No activation on final layer as it produces logits
        
        return x, next_hidden
