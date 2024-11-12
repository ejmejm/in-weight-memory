from typing import Callable, Optional, Tuple, List

from minGRU_pytorch import minGRU
import torch
import torch.nn as nn
import math


class BatchLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, n_layers: int, bias: bool = True):
        """Creates multiple linear layers that can be computed in parallel using batch matrix multiplication
        
        Args:
            in_features: Size of input features
            out_features: Size of output features
            n_layers: Number of parallel linear layers
            bias: Whether to include bias terms
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.n_layers = n_layers
        
        # Shape: (n_layers, in_features, out_features)
        self.weight = nn.Parameter(torch.empty(n_layers, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(n_layers, 1, out_features)) if bias else None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using the same approach as nn.Linear"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def _forward_single_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        x = x @ self.weight[layer_idx]
        if self.bias is not None:
            x = x + self.bias[layer_idx]
        return x
    
    def _forward_all_layers(self, x: torch.Tensor) -> torch.Tensor:
        # Save original shape and flatten all but last 2 dimensions
        orig_shape = x.shape
        x = x.reshape(-1, self.n_layers, self.in_features).transpose(0, 1) # (n_layers, batch_dim, in_features)
        
        # Batch matrix multiply
        x = torch.bmm(x, self.weight)
        
        if self.bias is not None:
            x = x + self.bias
            
        x = x.transpose(0, 1)  # (batch_dim, n_layers, out_features)
            
        # Restore original dimensions
        x = x.reshape(*orig_shape[:-2], self.n_layers, self.out_features)
        return x
    
    def forward(self, x: torch.Tensor, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., n_layers, in_features), or of shape (..., in_features) if layer_idx is specified
            layer_idx: Optional index to select a specific layer
            
        Returns:
            Output tensor of shape (..., n_layers, out_features), or of shape (..., out_features) if layer_idx is specified
        """
        if layer_idx is not None:
            return self._forward_single_layer(x, layer_idx)
        return self._forward_all_layers(x)
        


class MultiHeadGRU(nn.Module):
    def __init__(self, dim: int, n_heads: int, expansion_factor: float = 1.0):
        super().__init__()
        
        assert dim % n_heads == 0, f'dim {dim} must be divisible by n_heads {n_heads}'
        
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        dim_inner = int(self.head_dim * expansion_factor)

        # Create a GRU for each head
        self.grus = nn.ModuleList([
            nn.GRU(dim, dim_inner, batch_first=True)
            for _ in range(n_heads)
        ])

        if expansion_factor != 1.0:
            self.to_outs = BatchLinear(dim_inner, self.head_dim, n_heads, bias=False)
        else:
            self.to_outs = nn.Identity()

    def forward(
        self, 
        x: torch.Tensor, 
        prev_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape
        
        # Process each head separately
        outs = []
        next_hiddens = []
        
        for i, gru in enumerate(self.grus):
            head_prev_hidden = prev_hidden[:, i].transpose(0, 1) if prev_hidden is not None else None
            out, hidden = gru(x, head_prev_hidden)
            next_hidden = out[:, -1:, :]
            outs.append(out)
            next_hiddens.append(next_hidden.squeeze(1))
        
        # Stack outputs for batch processing
        out = torch.stack(outs, dim=2)  # (batch, seq_len, n_heads, dim_inner)
        
        # Process all outputs at once
        out = self.to_outs(out)  # (batch, seq_len, n_heads, head_dim)
        
        # Combine head outputs
        out = out.reshape(batch, seq_len, -1)  # (batch, seq_len, dim)
        next_prev_hidden = torch.stack(next_hiddens, dim=1)
            
        return out, next_prev_hidden


class GRUWrapper(nn.Module):
    def __init__(self, dim: int, expansion_factor: float = 1.0):
        super().__init__()
        dim_inner = int(dim * expansion_factor)
        self.gru = nn.GRU(dim, dim_inner, batch_first=True)
        self.to_out = nn.Linear(dim_inner, dim, bias=False) if expansion_factor != 1. else nn.Identity()

    def forward(
        self, 
        x: torch.Tensor, 
        prev_hidden: Optional[torch.Tensor] = None,
        modify_hidden_out_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            prev_hidden: Optional previous hidden state of shape (batch, 1, dim)
        """
        prev_hidden = prev_hidden.transpose(0, 1) if prev_hidden is not None else None
        out, hidden = self.gru(x, prev_hidden)
        next_prev_hidden = out[:, -1:, :]
        
        if modify_hidden_out_fn is not None:
            out = modify_hidden_out_fn(out)
        
        out = self.to_out(out)
            
        return out, next_prev_hidden


class StoryNetwork(nn.Module):
    def __init__(
        self, 
        vocab_size: int,
        d_model: int = 512,
        expansion_factor: float = 2.0,
        use_min_gru: bool = True,
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
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pre_gru = nn.Linear(d_model, d_model)
        self.leaky_relu = nn.LeakyReLU()
        if use_min_gru:
            self.gru = minGRU(dim=d_model, expansion_factor=expansion_factor)
        else:
            self.gru = GRUWrapper(dim=d_model, expansion_factor=expansion_factor)
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
            x, next_hidden = self.gru(x, return_next_prev_hidden=True)
            
        # Post GRU processing
        x = self.dropout(self.leaky_relu(self.post_gru1(x)))
        x = self.post_gru2(x)  # No activation on final layer as it produces logits
        
        return x, next_hidden
    

class RNNBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        expansion_factor: float = 2.0,
        use_min_gru: bool = True,
        mlp_ratio: float = 4.0,
        norm_cls: nn.Module = nn.LayerNorm,
        n_heads: int = 1,
    ):
        """
        Single RNN block following Llama 3 architecture
        
        Args:
            dim: Model dimension
            expansion_factor: Expansion factor for GRU
            use_min_gru: Whether to use minGRU or regular GRU
            mlp_ratio: Ratio for MLP hidden dimension
        """
        super().__init__()
        self.use_min_gru = use_min_gru
        
        self.rms_1 = norm_cls(dim)
        self.rms_2 = norm_cls(dim)
        
        # RNN layer
        if use_min_gru:
            self.rnn = minGRU(dim, expansion_factor)
        elif n_heads > 1:
            self.rnn = MultiHeadGRU(dim, n_heads, expansion_factor)
        else:
            self.rnn = GRUWrapper(dim, expansion_factor)
            
        # MLP components
        hidden_dim = int(dim * mlp_ratio)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.act_fn = nn.SiLU()

    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: Optional[torch.Tensor] = None,
        modify_hidden_out_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # First normalization and RNN
        residual = x
        x = self.rms_1(x)
        if self.use_min_gru:
            x, next_hidden = self.rnn(x, prev_hidden, return_next_prev_hidden=True)
        else:
            x, next_hidden = self.rnn(x, prev_hidden, modify_hidden_out_fn=modify_hidden_out_fn)
        x = residual + x
        
        # Second normalization and MLP
        residual = x
        x = self.rms_2(x)
        
        # MLP with gating
        gate_output = self.act_fn(self.gate_proj(x))
        up_output = self.up_proj(x)
        x = self.down_proj(gate_output * up_output)
        
        x = residual + x
        
        return x, next_hidden


class MultiLayerRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 4,
        expansion_factor: float = 2.0,
        use_min_gru: bool = True,
        mlp_ratio: float = 4.0,
        n_heads: int = 1,
    ):
        """
        Multi-layer RNN following Llama 3 architecture
        
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
            num_layers: Number of RNN blocks
            expansion_factor: Expansion factor for GRU layers
            use_min_gru: Whether to use minGRU or regular GRU
            mlp_ratio: Ratio for MLP hidden dimensions
        """
        super().__init__()
        
        assert n_heads <= 1 or not use_min_gru, 'Multi-head GRUs only supported with regular GRUs'
        
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.d_gru_hidden = int(d_model * expansion_factor)
        self.norm_cls = nn.LayerNorm
        self.mlp_ratio = mlp_ratio
        self.use_min_gru = use_min_gru
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            RNNBlock(
                dim=d_model,
                expansion_factor=expansion_factor,
                use_min_gru=use_min_gru,
                mlp_ratio=mlp_ratio,
                norm_cls=self.norm_cls,
                n_heads=n_heads,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = self.norm_cls(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(
        self,
        x: torch.Tensor,
        prev_hidden: Optional[List[torch.Tensor]] = None,
        modify_hidden_out_fn: Optional[Callable[[int, torch.Tensor], torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch, seq_len)
            prev_hidden: Optional list of previous hidden states for each layer
            
        Returns:
            tuple of (output logits, list of next hidden states)
        """
        x = self.embedding(x)
        
        next_hidden_states = []
        
        for i, layer in enumerate(self.layers):
            layer_prev_hidden = prev_hidden[i] if prev_hidden is not None else None
            if modify_hidden_out_fn is not None:
                layer_modify_hidden_out_fn = lambda hidden: modify_hidden_out_fn(i, hidden)
            else:
                layer_modify_hidden_out_fn = None
            x, next_hidden = layer(x, layer_prev_hidden, modify_hidden_out_fn=layer_modify_hidden_out_fn)
            next_hidden_states.append(next_hidden)
            
        x = self.norm(x)
        x = self.output(x)

        return x, torch.stack(next_hidden_states)

