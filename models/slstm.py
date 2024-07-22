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


sLSTMState = NamedTuple('sLSTMState', h=Array, c=Array, m=Array, n=Array)
sLSTMBlockState = NamedTuple('sLSTMBlockState', cell_state=sLSTMState, block_state=Array)


class sLSTMCell(eqx.Module, strict=True):
    """A single step of a scaled Long-Short Term Memory unit (sLSTM).

    Example:
        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            cell: sLSTMCell

            def __init__(self, ...):
                self.cell = sLSTMCell(...)

            def __call__(self, xs):
                scan_fn = lambda state, input: (cell(input, state), None)
                rnn_state = self.cell.init_state()
                final_state, _ = jax.lax.scan(scan_fn, rnn_state, xs)
                return final_state
        ```

    """
    weight_if: Array
    weight_zo: Array
    weight_hh: Array
    bias: Optional[Array]
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    head_size: int = eqx.field(static=True)
    separate_zo_input: bool = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        n_heads: int = 1,
        dtype = None,
        separate_zo_input: bool = False,
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
            separate_zo_input: Whether to use a different input for the cell (z) and output (o) gates.
            key: A `jax.random.PRNGKey` used to provide randomness for parameter
                initialisation. (Keyword only argument.)
        """
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"

        self.head_size = hidden_size // n_heads

        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey = jrandom.split(key, 3)
        lim = math.sqrt(1 / self.head_size)
        self.n_heads = n_heads

        self.weight_if = jrandom.uniform(
            ihkey, (n_heads, 2 * self.head_size, input_size), minval=-lim, maxval=lim, dtype=dtype)
        self.weight_zo = jrandom.uniform(
            ihkey, (n_heads, 2 * self.head_size, input_size), minval=-lim, maxval=lim, dtype=dtype)
        self.weight_hh = jrandom.uniform(
            # TODO: Add support for multiple heads
            hhkey, (n_heads, 4 * self.head_size, self.head_size), minval=-lim, maxval=lim, dtype=dtype)
        if use_bias:
            self.bias = jrandom.uniform(
                bkey, (n_heads, 4 * self.head_size,), minval=-lim, maxval=lim, dtype=dtype
            )
        else:
            self.bias = None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.separate_zo_input = separate_zo_input

    def init_state(self) -> sLSTMState:
        return sLSTMState(
            jnp.zeros((self.n_heads, self.head_size)),
            jnp.zeros((self.n_heads, self.head_size)),
            jnp.zeros((self.n_heads, self.head_size)),
            jnp.zeros((self.n_heads, self.head_size)),
        )

    @jax.named_scope("sLSTMCell")
    def __call__(
        self,
        input: Array,
        rnn_state: sLSTMState,
        *,
        zo_input: Optional[Array] = None,
        key: PRNGKeyArray = None,
    ) -> Tuple[sLSTMState, Array]:
        """
        Args:
            input: The input, which should be a JAX array of shape `(input_size,)`.
            hidden: The hidden state, which should be a 4-tuple of JAX arrays, each of
                shape `(n_heads, head_size)`.
            key: Ignored; provided for compatibility with the rest of the Equinox API.
                (Keyword only argument.)

        Returns:
            The updated hidden state, which is a 2-tuple of JAX arrays, each of shape
            `(n_heads, head_size)`.
        """
        prev_h, prev_c, prev_m, prev_n = rnn_state

        if not self.separate_zo_input:
            zo_input = input

        if_linear = self.weight_if @ input
        zo_linear = self.weight_zo @ zo_input
        combined_linear = jnp.concatenate([if_linear, zo_linear], axis=1)

        # (n_heads, 4 * head_size)
        lin = combined_linear + (self.weight_hh @ prev_h[..., None]).squeeze(2)

        if self.use_bias:
            lin = lin + self.bias

        i, f, z, o = jnp.split(lin, 4, axis=1)

        z = jnn.tanh(z)
        o = jnn.sigmoid(o)

        m = jnp.maximum(f + prev_m, i)
        i = jnp.exp(i - m)
        f = jnp.exp(f + prev_m - m)

        c = f * prev_c + i * z
        n = f * prev_n + i
        h = o * (c / n)
        
        return sLSTMState(h, c, m, n), h.reshape(self.hidden_size)


class sLSTMBlock(eqx.Module):
    """A block of scaled Long-Short Term Memory units (sLSTM) with normalization and projection layers.

    Example:
        This is often used by wrapping it into a `jax.lax.scan`. For example:

        ```python
        class Model(Module):
            block: sLSTMBlock

            def __init__(self, ...):
                self.block = sLSTMBlock(...)

            def __call__(self, xs):
                scan_fn = lambda state, input: (block(input, state), None)
                rnn_state = self.block.init_state()
                final_state, _ = jax.lax.scan(scan_fn, rnn_state, xs)
                return final_state
        ```

    Args:
        hidden_size: The dimensionality of the hidden state passed along between time steps.
        key: A `jax.random.PRNGKey` used to provide randomness for parameter initialisation.
        n_heads: The number of attention heads.
        projection_factor: The factor by which to scale the hidden size for the projection layers.

    Returns:
        A block of scaled Long-Short Term Memory units (sLSTM) with normalization and projection layers.
    """
    layer_norm: nn.LayerNorm
    conv: Optional[nn.Conv1d]
    lstm_cell: sLSTMCell
    group_norm: nn.GroupNorm
    upscale_layer: nn.Linear
    downscale_layer: nn.Linear

    upscale_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    projection_factor: float = eqx.field(static=True)
    use_conv: bool = eqx.field(static=True)

    def __init__(
            self,
            hidden_size: int,
            key: PRNGKeyArray,
            n_heads: int = 4,
            projection_factor: float = (4.0 / 3.0),
            use_conv: bool = True,
        ):
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.projection_factor = projection_factor
        self.use_conv = use_conv

        lstm_key, upscale_key, downscale_key, conv_key = jrandom.split(key, 4)

        self.layer_norm = nn.LayerNorm(hidden_size)
        self.lstm_cell = sLSTMCell(
            hidden_size, hidden_size, n_heads=n_heads, separate_zo_input=use_conv, key=lstm_key)
        self.group_norm = nn.GroupNorm(n_heads, hidden_size)

        self.upscale_size = int(projection_factor * hidden_size)
        self.upscale_layer = nn.Linear(hidden_size, 2 * self.upscale_size, key=upscale_key)
        self.downscale_layer = nn.Linear(self.upscale_size, hidden_size, key=downscale_key)

        if use_conv:
            self.conv = nn.Conv1d(
                self.hidden_size, self.hidden_size, 4, groups=self.hidden_size, key=conv_key)
        else:
            self.conv = None

    def init_state(self) -> sLSTMBlockState:
        """
        Initializes the hidden state for the mLSTM block.

        Returns:
            The initial hidden state, which is the cell state and a 3-tuple of JAX arrays,
            each of shape `(n_heads, head_size)`.
        """
        return sLSTMBlockState(
            cell_state = self.lstm_cell.init_state(),
            block_state = jnp.zeros((3, self.hidden_size)),
        )

    @jax.named_scope("sLSTMBlock")
    def __call__(self, x: Array, rnn_state: sLSTMBlockState) -> Tuple[sLSTMBlockState, Array]:
        """
        Args:
            x: The input, which should be a JAX array of shape `(hidden_size,)`.
            rnn_state: The rnn state, which should be a 4-tuple of JAX arrays, each of
                shape `(n_heads, head_size)`.

        Returns:
            A tuple containing the updated hidden state and the output of the block.
        """
        x_normalized = self.layer_norm(x)

        if self.use_conv:
            z_sequence = jnp.concatenate([rnn_state.block_state, x_normalized[None]], axis=0)
            new_block_state = z_sequence[1:]
            z = self.conv(z_sequence.T)
            z = z.squeeze(1)

            if_input = jnn.swish(z)
            zo_input = x_normalized

            new_cell_state, z = self.lstm_cell(if_input, rnn_state.cell_state, zo_input=zo_input)
        else:
            new_block_state = rnn_state.block_state
            new_cell_state, z = self.lstm_cell(x_normalized, rnn_state.cell_state)

        z = self.group_norm(z)
        z = self.upscale_layer(z)
        upscale_1, upscale_2 = jnp.split(z, 2)
        upscale_1 = jnn.gelu(upscale_1)
        z = upscale_1 * upscale_2 # Self gating
        z = self.downscale_layer(z)
        z += x # Skip connection

        new_rnn_state = sLSTMBlockState(new_cell_state, new_block_state)

        return new_rnn_state, z