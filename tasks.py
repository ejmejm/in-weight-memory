from typing import Dict

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Integer, Scalar

from utils import tree_replace


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


class ContinualARState(eqx.Module):
    """
    This class represents the state for a continual associative recall task. 
    The task involves creating and recalling associations between variable names and values. 
    The state includes parameters for the environment, such as the lengths and vocabularies of the names and values, 
    as well as mutable state variables that track the current associations and sequences. 
    The task tests the model's ability to recall existing associations and create new ones, 
    with a certain probability of recalling an existing association versus creating a new one.
    """

    # Env params
    name_length: int = eqx.field(static=True)
    val_length: int = eqx.field(static=True)
    name_vocab_size: int = eqx.field(static=True)
    val_vocab_size: int = eqx.field(static=True)
    max_vars: int = eqx.field(static=True)
    test_recall_prob: float = eqx.field(static=True)
    fill_before_recall: bool = eqx.field(static=True)

    # Env mutable state
    rng: jax.random.PRNGKey
    curr_names: Integer[Array, "max_vars"]
    curr_vals: Integer[Array, "max_vars"]
    n_curr_vars: Integer[Scalar, ""]
    var_insert_idx: Integer[Scalar, ""]
    curr_sequence: Integer[Array, "name_length + val_length + 1"]
    curr_sequence_len: Integer[Scalar, ""]
    curr_sequence_idx: Integer[Scalar, ""]


    def __init__(
            self,
            rng: jax.random.PRNGKey,
            name_length: int = 2,
            val_length: int = 2,
            name_vocab_size: int = 26,
            val_vocab_size: int = 10,
            max_vars: int = 10,
            test_recall_prob: float = 0.5,
            fill_before_recall: bool = False,
    ):
        self.rng = rng
        self.name_length = name_length
        self.val_length = val_length
        self.name_vocab_size = name_vocab_size
        self.val_vocab_size = val_vocab_size
        self.max_vars = max_vars
        self.test_recall_prob = test_recall_prob
        self.fill_before_recall = fill_before_recall

        self.curr_names = jnp.zeros((max_vars, name_length), dtype=jnp.int32)
        self.curr_vals = jnp.zeros((max_vars, val_length), dtype=jnp.int32)
        self.var_insert_idx = jnp.array(0, dtype=jnp.int32)
        self.n_curr_vars = jnp.array(0, dtype=jnp.int32)
        self.curr_sequence = jnp.zeros((name_length + val_length + 2), dtype=jnp.int32)
        self.curr_sequence_len = jnp.array(0, dtype=jnp.int32)
        self.curr_sequence_idx = jnp.array(0, dtype=jnp.int32)


def next_associative_recall_obs(state: ContinualARState) -> Dict[str, Array]:
    """
    Generate the next observation for the continual associative recall task.

    The task involves creating and recalling associations between variable names and values.
    New observations are generated by either recalling an existing association or creating a new one.
    The function updates the state with the new sequence and increments the sequence index.

    Args:
        state (ContinualARState): The current state of the task.

    Returns:
        Tuple[ContinualARState, Dict[str, Array]]: The updated state and a dictionary containing:
            - input_id: The current input id.
            - target_id: The target id for the next step.
            - loss_mask: 1 for all value parts of the output sequence and 0 elsewhere.
    """

    def reset_sequence(state):
        rng, name_rng, val_rng, association_rng = jax.random.split(state.rng, 4)
        
        state = tree_replace(
            state,
            rng = rng,
            curr_sequence_idx = jnp.array(0, dtype=jnp.int32), # First observation
        )
        
        def test_association(state):
            idx = jax.random.randint(association_rng, (), minval=0, maxval=state.n_curr_vars)
            name = state.curr_names[idx]
            val = state.curr_vals[idx]
            return name, val, state

        def create_new_association(state):
            name = jax.random.randint(name_rng, shape=(state.name_length,), minval=0, maxval=state.name_vocab_size)
            val = jax.random.randint(
                val_rng, shape=(state.val_length,), minval=state.name_vocab_size,
                maxval=state.name_vocab_size+state.val_vocab_size,
            )

            # Check if this variable already exists
            def update_existing_value(existing_mask):
                idx = jnp.where(existing_mask, size=state.curr_names.shape[0])[0]
                new_curr_vals = state.curr_vals.at[idx].set(val)
                return dict(
                    curr_names=state.curr_names,
                    curr_vals=new_curr_vals,
                    n_curr_vars=state.n_curr_vars,
                    var_insert_idx=state.var_insert_idx,
                )

            def record_new_association(_):
                new_curr_names = state.curr_names.at[state.n_curr_vars].set(name)
                new_curr_vals = state.curr_vals.at[state.n_curr_vars].set(val)
                return dict(
                    curr_names = new_curr_names,
                    curr_vals = new_curr_vals,
                    n_curr_vars = jnp.minimum(state.n_curr_vars + 1, state.max_vars),
                    var_insert_idx = (state.var_insert_idx + 1) % state.max_vars,
                )

            existing_mask = jnp.all(state.curr_names == name, axis=1)
            update_vars = jax.lax.cond(
                jnp.any(existing_mask),
                update_existing_value,
                record_new_association,
                existing_mask,
            )
            state = tree_replace(state, **update_vars)
            return name, val, state

        # `test_recall_prob` chance of testing an existing association, otherwise create a new one
        test_recall_gate = state.n_curr_vars == state.max_vars if state.fill_before_recall else state.n_curr_vars > 0
        name, val, state = jax.lax.cond(
            jax.lax.bitwise_and(test_recall_gate, jax.random.uniform(rng) < state.test_recall_prob),
            test_association,
            create_new_association,
            state,
        )

        sequence_padding = jnp.zeros((state.curr_sequence.shape[0] - name.shape[0] - val.shape[0]), dtype=jnp.int32)

        # The new sequence is var_name, var_value, padding
        new_sequence = jnp.concatenate([
            name, val, sequence_padding], axis=0)
        new_sequence_len = jnp.array(name.shape[0] + val.shape[0], dtype=jnp.int32)
        
        state = tree_replace(state, curr_sequence_len=new_sequence_len, curr_sequence=new_sequence)
        return state

    def increment_sequence_idx(state):
        return tree_replace(state, curr_sequence_idx=state.curr_sequence_idx + 1)

    state = jax.lax.cond(
        state.curr_sequence_idx >= state.curr_sequence_len - 1,
        reset_sequence,
        increment_sequence_idx,
        state,
    )

    # Generate the next observation
    input_id = state.curr_sequence[state.curr_sequence_idx]
    target_id = state.curr_sequence[state.curr_sequence_idx + 1]
    loss_mask = jnp.where(
        jax.lax.bitwise_and(
            state.name_length - 1 <= state.curr_sequence_idx,
            state.curr_sequence_idx < state.name_length + state.val_length - 1,
        ),
        1, 0,
    )

    return state, {
        "input_id": input_id,
        "target_id": target_id,
        "loss_mask": loss_mask,
    }


if __name__ == '__main__':
    rng = jax.random.PRNGKey(0)
    state = ContinualARState(rng)
    all_obs = []

    step_fn = jax.jit(next_associative_recall_obs)

    for _ in range(40):
        state, obs = step_fn(state)
        all_obs.append(obs)

    all_obs = jax.tree.map(lambda *args: jnp.stack(args), *all_obs)
    
    print(f'Input sequence:\n{decode_sequence(all_obs["input_id"])}\n')
    print(f'Target sequence:\n{decode_sequence(all_obs["target_id"])}\n')
    print(f'Loss mask:\n{all_obs["loss_mask"]}')