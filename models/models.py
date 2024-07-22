import math
from typing import List, Tuple, Optional, Sequence, Union

import equinox as eqx
from equinox import nn
import jax
from jax import Array
import jax.numpy as jnp

from .slstm import sLSTMBlock, sLSTMCell, sLSTMState
from .mlstm import mLSTMBlock, mLSTMCell, mLSTMState, mLSTMBlockState


LSTMState = Tuple[Union[Tuple[Array, Array], sLSTMState, mLSTMState, mLSTMBlockState]]
LSTM_CLS = sLSTMBlock


class SupervisedModel(eqx.Module):
    vocab_size: int = eqx.field(static=True)
    layer_sizes: Sequence[int] = eqx.field(static=True)
    recurrent_layer_indices: List[int] = eqx.field(static=True)
    output_dim: int = eqx.field(static=True)
    n_layers: int = eqx.field(static=True)
    layers: List[eqx.Module]
    embedding: eqx.Module

    def __init__(
            self,
            rng: Array,
            vocab_size: int,
            embedding_dim: int,
            layer_sizes: Sequence[int],
            output_dim: int,
            recurrent_layer_indices: List[int],
        ):
        """Extracts features from input observations using a series of linear layers and LSTM cells.

        Args:
            rng (Array): The PRNG key used to initialize weights.
            vocab_size (int): The size of the vocabulary.
            embedding_dim (int): The dimensionality of the embedding.
            layer_sizes (Sequence[int]): The sizes of all the layers.
            output_dim (int): The (1D) dimensionality of the output features.
            recurrent_layer_indices (List[int]): The indices of the layers that are LSTM cells.
        """
        self.vocab_size = vocab_size
        self.recurrent_layer_indices = recurrent_layer_indices
        self.output_dim = output_dim
        self.n_layers = len(layer_sizes) + 1

        gen_keys = jax.random.split(rng, len(layer_sizes) + 1)
        self.layer_sizes = [embedding_dim] + layer_sizes + [output_dim]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, key=gen_keys[0])
        
        self.layers = []
        for i in range(1, len(self.layer_sizes)):
            if self.layer_sizes[i] < 0:
                self.layers.append(nn.Identity())
            elif i - 1 in self.recurrent_layer_indices:
                if LSTM_CLS in (mLSTMBlock, sLSTMBlock):
                    assert self.layer_sizes[i-1] == self.layer_sizes[i], \
                        "mLSTM and sLSTM blocks require the same dimensionality for the input and output layers."
                    self.layers.append(LSTM_CLS(self.layer_sizes[i], key=gen_keys[i]))
                else:
                    self.layers.append(LSTM_CLS(self.layer_sizes[i-1], self.layer_sizes[i], key=gen_keys[i]))
            else:
                self.layers.append(nn.Linear(self.layer_sizes[i-1], self.layer_sizes[i], key=gen_keys[i]))

    def init_rnn_state(self):
        if LSTM_CLS == nn.LSTMCell:
            return tuple([(
                jnp.zeros(self.layer_sizes[i + 1]),
                jnp.zeros(self.layer_sizes[i + 1]))
                for i in self.recurrent_layer_indices
            ])
        else:
            return tuple([
                self.layers[i].init_state()
                for i in self.recurrent_layer_indices
            ])
        
    @jax.remat
    def __call__(self, x: Array, rnn_state: Optional[LSTMState] = None) -> Array:
        if rnn_state is None:
            rnn_state = self.init_rnn_state()

        z = self.embedding(x)

        recurrent_layer_idx = 0
        new_rnn_state = []
        for i in range(self.n_layers):
            layer = self.layers[i]

            if i in self.recurrent_layer_indices:
                rnn_state_i = rnn_state[recurrent_layer_idx]

                if LSTM_CLS == nn.LSTMCell:
                    out_rnn_state = layer(z, rnn_state_i)
                    z = out_rnn_state[0].flatten()
                else:
                    out_rnn_state, z = layer(z, rnn_state_i)

                new_rnn_state.append(out_rnn_state)
                recurrent_layer_idx += 1
            else:
                z = layer(z)

            if i < self.n_layers - 1:
                z = jax.nn.relu(z)

        return tuple(new_rnn_state), z

    def forward_sequence(self, rnn_state, xs):
        def step(rnn_state, x):
            rnn_state, y = self.__call__(x, rnn_state)
            return rnn_state, y
        state, ys = jax.lax.scan(step, rnn_state, xs)
        return state, ys