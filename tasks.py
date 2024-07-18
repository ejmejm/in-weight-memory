from typing import Dict

import jax
import jax.numpy as jnp


VOCAB = 'abcdefghijklmnopqrstuvwxyz0123456789=,|'


def decode_sequence(sequence: jnp.ndarray) -> str:
    return ''.join([VOCAB[i] for i in sequence])


def gen_train_sequence(
    rng: jax.random.PRNGKey, name_length: int, val_length: int,
    n_vars: int, name_vocab_size: int = 26, val_vocab_size: int = 10,
) -> Dict[str, jnp.ndarray]:
    """Generate a training sequence.

    Example sequence: "a=1,b=2,c=3|c=3,a=1,b=2"

    Args:
        rng: random key.
        name_length: number of characters for each variable name.
        val_length: number of digits for each variable value.
        n_vars: number of variables.
        name_vocab_size: size of the name vocabulary (default 26 to mimic alphabet).
        val_vocab_size: size of the value vocabulary (default 10 to mimic base 10 digits).

    Returns:
        A dictionary containing the following:
            - input_ids: all ids in the sequence except for the last.
            - target_ids: all ids in the sequence except for the first.
            - loss_mask: 1 for all value parts of the output sequence and 0 elsewhere.
    """
    # 3 special tokens: "=", ",", and "|"
    equal_signs = jnp.full((n_vars, 1), name_vocab_size + val_vocab_size)
    commas = jnp.full((n_vars, 1), name_vocab_size + val_vocab_size + 1)
    vertical_bar = jnp.full((1), name_vocab_size + val_vocab_size + 2)

    rngs = jax.random.split(rng, 3)

    # Create the sequence of initial variables
    # e.g. "a=1,b=2,c=3"
    name_seq = jax.random.randint(rngs[0], shape=(n_vars, name_length), minval=0, maxval=name_vocab_size)
    val_seq = jax.random.randint(
        rngs[1], shape=(n_vars, val_length), minval=name_vocab_size, maxval=name_vocab_size+val_vocab_size)
    input_vals = jnp.concatenate([name_seq, equal_signs, val_seq, commas], axis=1)
    input_seq = input_vals.flatten()[:-1] # Get rid of last comma

    # Create a permutation of the variables for the model to predict
    # e.g. "c=3,a=1,b=2"
    permutation = jax.random.permutation(rngs[2], n_vars)
    name_seq = name_seq[permutation]
    val_seq = val_seq[permutation]
    output_vals = jnp.concatenate([name_seq, equal_signs, val_seq, commas], axis=1)
    output_seq = output_vals.flatten()[:-1] # Get rid of last comma

    # Combine into the full sequence
    full_seq = jnp.concatenate([input_seq, vertical_bar, output_seq])
    input_ids = full_seq[:-1]
    target_ids = full_seq[1:]

    # Loss mask is 1 for all value parts of the output sequence and 0 elsewhere
    val_mask = jnp.where(
        jnp.less_equal(name_vocab_size, output_seq)
        & jnp.less(output_seq, name_vocab_size + val_vocab_size), 1, 0)
    loss_mask = jnp.concatenate([
        jnp.zeros(input_seq.shape[0]),
        val_mask,
    ])

    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "loss_mask": loss_mask,
    }