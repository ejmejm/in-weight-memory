import math
from typing import NamedTuple, Optional, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx
from equinox import nn
from equinox._misc import default_floating_dtype


mLSTMState = NamedTuple('mLSTMState', h=Array, c=Array, m=Array, n=Array)
mLSTMBlockState = NamedTuple('mLSTMBlockState', cell_state=mLSTMState, block_state=Array)


class mLSTMCell(eqx.Module, strict=True):
    """A single step of a Long-Short Term Memory unit (mLSTM).

    Example:
        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            cell: mLSTMCell

            def __init__(self, ...):
                self.cell = mLSTMCell(...)

            def __call__(self, xs):
                scan_fn = lambda state, input: (cell(input, state), None)
                rnn_state = self.cell.init_state()
                final_state, _ = jax.lax.scan(scan_fn, rnn_state, xs)
                return final_state
        ```
    """
    if_weights: Array
    o_weights: Array
    kq_weights: Array
    v_weights: Array
    kvq_bias: Optional[Array]
    if_bias: Optional[Array]
    o_bias: Optional[Array]
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    head_size: int = eqx.field(static=True)
    separate_value_input: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        n_heads: int = 1,
        dtype = None,
        separate_value_input: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        """
        Args:
            input_size: The dimensionality of the input vector at each time step.
            hidden_size: The dimensionality of the hidden state passed along between
                time steps.
            use_bias: Whether to add on a bias after each update.
            dtype: The dtype to use for all weights and biases in this LSTM cell.
                Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
                whether JAX is in 64-bit mode.
            separate_value_input: Whether to use a separate input for the value layer.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"

        self.head_size = hidden_size // n_heads
        self.n_heads = n_heads

        dtype = default_floating_dtype() if dtype is None else dtype
        ih_key, o_key, kv_key, q_key, ihb_key, ob_key, kvqb_key = jrandom.split(key, 7)
        lim = math.sqrt(1 / self.head_size)

        self.if_weights = jrandom.uniform(
            ih_key, (n_heads, 2, input_size), minval=-lim, maxval=lim, dtype=dtype)
        self.o_weights = jrandom.uniform(
            o_key, (n_heads, self.head_size, input_size), minval=-lim, maxval=lim, dtype=dtype)
        self.kq_weights = jrandom.uniform(
            kv_key, (n_heads, 2 * self.head_size, input_size), minval=-lim, maxval=lim, dtype=dtype)
        self.v_weights = jrandom.uniform(
            q_key, (n_heads, self.head_size, input_size), minval=-lim, maxval=lim, dtype=dtype)

        if use_bias:
            self.if_bias = jrandom.uniform(
                ihb_key, (n_heads, 2,), minval=-lim, maxval=lim, dtype=dtype
            )
            self.o_bias = jrandom.uniform(
                ob_key, (n_heads, self.head_size,), minval=-lim, maxval=lim, dtype=dtype
            )
            self.kvq_bias = jrandom.uniform(
                kvqb_key, (n_heads, 3, self.head_size,), minval=-lim, maxval=lim, dtype=dtype
            )
        else:
            self.bias = None
            self.kvq_bias = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.separate_value_input = separate_value_input

    def init_state(self) -> mLSTMState:
        return mLSTMState(
            jnp.zeros((self.n_heads, self.head_size)),
            jnp.zeros((self.n_heads, self.head_size, self.head_size)),
            jnp.zeros((self.n_heads,)),
            jnp.zeros((self.n_heads, self.head_size)),
        )

    @jax.named_scope("mLSTMCell")
    def __call__(
        self,
        input: Array,
        rnn_state: mLSTMState,
        v_input: Optional[Array] = None,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[mLSTMState, Array]:
        """
        Args:
            input: The input, which should be a JAX array of shape `(input_size,)`.
            rnn_state: A 3-tuple containing the h, c, and n states.
            v_input: The input to the value layer, which should be a JAX array of shape `(hidden_size,)`.
                `separate_value_input` must be set to True on class instantiation for this to take effect.
            key: Ignored; provided for compatibility with the rest of the Equinox API.
                (Keyword only argument.)

        Returns:
            The updated hidden state, which is a 2-tuple of JAX arrays, each of shape
            `(hidden_size,)`.
        """
        prev_h, prev_c, prev_m, prev_n = rnn_state

        # Calculate gate values
        i_f = jnp.inner(self.if_weights, input)
        o = self.o_weights @ input

        if self.use_bias:
            i_f += self.if_bias
            o += self.o_bias

        o = jnn.sigmoid(o)

        i, f = i_f.T
        m = jnp.maximum(f + prev_m, i)
        i = jnp.exp(i - m)
        f = jnp.exp(f + prev_m - m)

        if not self.separate_value_input:
            v_input = input

        # Calculate keys, values, and queries from normalized inputs
        kq = self.kq_weights @ input
        k, q = jnp.split(kq, 2, axis=1)
        v = self.v_weights @ v_input
        k *= 1.0 / jnp.sqrt(self.head_size)

        if self.use_bias:
            k_bias, v_bias, q_bias = self.kvq_bias.transpose(1, 0, 2)
            k += k_bias
            v += v_bias
            q += q_bias

        # Calculate recurrent values
        n = f[:, None] * prev_n + i[:, None] * k
        c = f[:, None, None] * prev_c + i[:, None, None] * v[:, :, None] @ k[:, None, :]
        h = (c @ q[:, :, None]).squeeze(2)
        h /= jnp.maximum(
            jnp.abs(n[:, None, :] @ q[:, :, None]).squeeze((1, 2,)),
            jnp.ones(self.n_heads, dtype=int),
        )[:, None]
        h = o * h

        return mLSTMState(h, c, m, n), h.reshape(self.hidden_size)


class mLSTMBlock(eqx.Module):
    """A block of scaled Long-Short Term Memory units (mLSTM) with normalization and projection layers.

    Example:
        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            block: mLSTMBlock

            def __init__(self, ...):
                self.block = mLSTMBlock(...)

            def __call__(self, xs):
                scan_fn = lambda state, input: (block(input, state), None)
                rnn_state = self.block.init_state()
                final_state, _ = jax.lax.scan(scan_fn, rnn_state, xs)
                return final_state
        ```
    """
    layer_norm: nn.LayerNorm
    conv: nn.Conv1d
    lstm_cell: mLSTMCell
    group_norm: nn.GroupNorm
    upscale_layer: nn.Linear
    downscale_layer: nn.Linear
    learnable_skip_params: Array

    upscale_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    projection_factor: float = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        n_heads: int = 4,
        projection_factor: float = 2.0,
        *,
        key: PRNGKeyArray,
    ):
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.projection_factor = projection_factor

        upscale_key, lstm_key, conv_key, downscale_key = jrandom.split(key, 4)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.upscale_size = int(projection_factor * hidden_size)
        self.upscale_layer = nn.Linear(hidden_size, 2 * self.upscale_size, key=upscale_key)
        self.conv = nn.Conv1d(self.upscale_size, self.upscale_size, 4, groups=self.upscale_size, key=conv_key)
        self.lstm_cell = mLSTMCell(
            self.upscale_size, self.upscale_size, n_heads=n_heads, separate_value_input=True, key=lstm_key)
        self.group_norm = nn.GroupNorm(n_heads, self.upscale_size)
        self.learnable_skip_params = jnp.ones(self.upscale_size)

        self.downscale_layer = nn.Linear(self.upscale_size, hidden_size, key=downscale_key)

    def init_state(self) -> mLSTMBlockState:
        """
        Initializes the hidden state for the sLSTM block.

        Returns:
            The initial hidden state, which is a the cell state and a 3-tuple of JAX arrays,
            each of shape `(n_heads, head_size)`.
        """
        return mLSTMBlockState(
            cell_state = self.lstm_cell.init_state(),
            block_state = jnp.zeros((3, self.upscale_size)),
        )

    @jax.named_scope("mLSTMBlock")
    def __call__(
        self,
        x: Array,
        rnn_state: mLSTMBlockState,
        *,
        key: Optional[PRNGKeyArray] = None,
    ) -> Tuple[mLSTMBlockState, Array]:
        """
        Args:
            x: The input, which should be a JAX array of shape `(hidden_size,)`.
            rnn_state: The current state of the RNN.
            key: Ignored; provided for compatibility with the rest of the Equinox API.
                (Keyword only argument.)

        Returns:
            A tuple containing the updated hidden state and the output of the block.
        """
        z = self.layer_norm(x)
        z = self.upscale_layer(z)
        z, skip_z = jnp.split(z, 2)

        # LSTM branch
        z_sequence = jnp.concatenate([rnn_state.block_state, z[None]], axis=0)
        new_block_state = z_sequence[1:]
        z = self.conv(z_sequence.T)
        z = z.squeeze(1)
        qk_input = jnn.swish(z)
        v_input = z
        new_cell_state, z = self.lstm_cell(qk_input, rnn_state.cell_state, v_input=v_input)
        z = self.group_norm(z)
        z = z + qk_input * self.learnable_skip_params

        # Branches converge
        z = z * jnn.swish(skip_z)
        z = self.downscale_layer(z)
        z = z + x # Final skip connection

        new_rnn_state = mLSTMBlockState(new_cell_state, new_block_state)

        return new_rnn_state, z
