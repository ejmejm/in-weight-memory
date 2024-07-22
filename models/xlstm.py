from typing import Any, Dict, List, Optional, Tuple, Union

import equinox as eqx
from equinox import nn
import jax
from jax import Array
from jaxtyping import PRNGKeyArray

from .mlstm import mLSTMBlock, mLSTMBlockState
from .slstm import sLSTMBlock, sLSTMBlockState


xLSTMState = Tuple[Union[mLSTMBlockState, sLSTMBlockState]]


class xLSTM(eqx.Module):
    vocab_size: int = eqx.field(static=True)
    hidden_dim: int = eqx.field(static=True)
    n_blocks: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    penultimate_norm: bool = eqx.field(static=True)

    blocks: List[Union[mLSTMBlock, sLSTMBlock]]
    embedding: nn.Embedding
    output_layer: nn.Linear
    layer_norm: Optional[nn.LayerNorm]

    def __init__(
            self,
            vocab_size: int,
            hidden_dim: int,
            n_blocks: int,
            n_heads: int,
            ms_ratio: Tuple[int, int],
            penultimate_norm: bool = True,
            mlstm_kwargs: Dict[str, Any] = None,
            slstm_kwargs: Dict[str, Any] = None,
            *,
            key: PRNGKeyArray,
        ):
        """Extracts features from input observations using a series of linear layers and LSTM cells.

        Args:
            rng (Array): The PRNG key used to initialize weights.
            vocab_size (int): The size of the vocabulary.
            hidden_dim (int): The dimensionality of the hidden state.
            n_blocks (int): The number of m/sLSTM blocks.
            n_heads (int): The number of heads in each block.
            ms_ratio (float): The ratio of mLSTM to sLSTM blocks.
            key (PRNGKeyArray): The PRNG key used to initialize weights.
        """
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.penultimate_norm = penultimate_norm

        mlstm_kwargs = mlstm_kwargs or {}
        slstm_kwargs = slstm_kwargs or {}

        gen_keys = jax.random.split(key, n_blocks + 2)
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim, key=gen_keys[0])

        block_types = [sLSTMBlock] * ms_ratio[1] + [mLSTMBlock] * ms_ratio[0]
        self.blocks = []
        for i in range(n_blocks):
            block_cls = block_types[i % len(block_types)]
            kwargs = mlstm_kwargs if block_cls == mLSTMBlock else slstm_kwargs
            block = block_cls(hidden_dim, n_heads=n_heads, key=gen_keys[i + 1], **kwargs)
            self.blocks.append(block)

        self.output_layer = nn.Linear(hidden_dim, vocab_size, key=gen_keys[-1])

        if penultimate_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = None

    def init_rnn_state(self) -> xLSTMState:
        return tuple([
            self.blocks[i].init_state()
            for i in range(self.n_blocks)
        ])
        
    @jax.remat
    def __call__(
        self,
        x: Array,
        rnn_state: Optional[xLSTMState] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[xLSTMState, Array]:
        """
        Args:
            x: The input, which should be a JAX scalar.
            rnn_state: The recurrent states for each block.
            key: Ignored; provided for compatibility with the rest of the Equinox API.
                (Keyword only argument.)

        Returns:
            A tuple containing the updated states and logits of shape `(vocab_size,)`.
        """
        
        if rnn_state is None:
            rnn_state = self.init_rnn_state()

        z = self.embedding(x)

        new_states = []
        for i, block in enumerate(self.blocks):
            block_input = z
            new_state, z = block(block_input, rnn_state[i])
            new_states.append(new_state)
            z += block_input # Residual connection between blocks

        if self.penultimate_norm:
            z = self.layer_norm(z)

        logits = self.output_layer(z)

        return tuple(new_states), logits

    def forward_sequence(self, rnn_state, xs):
        def step(rnn_state, x):
            rnn_state, y = self.__call__(x, rnn_state)
            return rnn_state, y
        state, ys = jax.lax.scan(step, rnn_state, xs)
        return state, ys