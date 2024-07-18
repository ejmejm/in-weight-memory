import math
from typing import Optional

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx
from equinox._misc import default_floating_dtype


class mLSTMCell(eqx.Module, strict=True):
    """A single step of a Long-Short Term Memory unit (mLSTM).

    !!! example

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
    kvq_weights: Array
    kvq_bias: Optional[Array]
    if_bias: Optional[Array]
    o_bias: Optional[Array]
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    n_heads: int = eqx.field(static=True)
    head_size: int = eqx.field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        n_heads: int = 1,
        dtype = None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `dtype`: The dtype to use for all weights and biases in this LSTM cell.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
            whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"

        self.head_size = hidden_size // n_heads
        self.n_heads = n_heads

        dtype = default_floating_dtype() if dtype is None else dtype
        ih_key, o_key, kvq_key, ihb_key, ob_key, kvqb_key = jrandom.split(key, 6)
        lim = math.sqrt(1 / self.head_size)

        self.if_weights = jrandom.uniform(
            ih_key, (n_heads, 2, input_size), minval=-lim, maxval=lim, dtype=dtype)
        self.o_weights = jrandom.uniform(
            o_key, (n_heads, self.head_size, input_size), minval=-lim, maxval=lim, dtype=dtype)
        self.kvq_weights = jrandom.uniform(
            kvq_key, (n_heads, 3 * self.head_size, input_size), minval=-lim, maxval=lim, dtype=dtype)

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

    def init_state(self):
        return tuple([
            jnp.zeros((self.n_heads, self.head_size)),
            jnp.zeros((self.n_heads, self.head_size, self.head_size)),
            jnp.zeros((self.n_heads, self.head_size)),
        ])

    @jax.named_scope("mLSTMCell")
    def __call__(self, input, hidden, *, key=None):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a 2-tuple of JAX arrays, each of
            shape `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a 2-tuple of JAX arrays, each of shape
        `(hidden_size,)`.
        """
        prev_h, prev_c, prev_n = hidden

        # Calculate gate values
        i_f = jnp.inner(self.if_weights, input)
        o = self.o_weights @ input

        if self.use_bias:
            i_f += self.if_bias
            o += self.o_bias

        i, f = i_f.T
        i = jnp.exp(i)
        f = jnp.exp(f)
        o = jnn.sigmoid(o)

        # Calculate keys, values, and queries from normalized inputs
        kvq = self.kvq_weights @ input # TODO: Don't forget to normalize this input in the block
        k, v, q = jnp.split(kvq, 3, axis=1)
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

        return h, c, n
