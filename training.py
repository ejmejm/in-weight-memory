from typing import Callable, Dict, Tuple

import equinox as eqx
import optax
import jax
from jax import Array
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from models.models import LSTMState
from utils import tree_replace


class TrainState(eqx.Module):
    rng: PRNGKeyArray
    opt_state: optax.OptState
    train_step: Array
    tx_update_fn: Callable = eqx.field(static=True)

    def __init__(
            self,
            rng: PRNGKeyArray,
            opt_state: optax.OptState,
            tx_update_fn: Callable,
        ):
        self.rng = rng
        self.opt_state = opt_state
        self.tx_update_fn = tx_update_fn
        self.train_step = jnp.array(0)


def supervised_loss_and_grads(
        model: eqx.Module,
        rnn_state: LSTMState,
        sequence: Dict[str, Array],
    ):
    def loss_fn(model: eqx.Module, rnn_state: LSTMState, subsequence: Dict[str, Array]):
        input_tokens = subsequence['input_ids']
        target_tokens = subsequence['target_ids']
        loss_mask = subsequence['loss_mask']

        rnn_state, pred_ids = model.forward_sequence(rnn_state, input_tokens)
        loss = optax.softmax_cross_entropy_with_integer_labels(pred_ids, target_tokens) * loss_mask
        n_targets = jnp.sum(loss_mask)
        loss = jax.lax.cond(n_targets > 0, lambda x: jnp.sum(x), lambda _: 0.0, loss)
        return loss, (n_targets, rnn_state)
    
    value_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    # Compute loss and gradients over the entire sequence
    (loss, (total_labels, rnn_state)), grads = value_grad_fn(model, rnn_state, sequence)

    # Normalize the loss and gradients
    loss /= total_labels
    grads = jax.tree_map(lambda x: x / total_labels, grads)

    return loss, grads, rnn_state


def supervised_scan_loss_and_grads(
        model: eqx.Module,
        rnn_state: LSTMState,
        sequence: Dict[str, Array],
        tbptt_window: int = 4,
    ):
    def loss_fn(model: eqx.Module, rnn_state: LSTMState, subsequence: Dict[str, Array]):
        input_tokens = subsequence['input_ids']
        target_tokens = subsequence['target_ids']
        loss_mask = subsequence['loss_mask']

        rnn_state, pred_ids = model.forward_sequence(rnn_state, input_tokens)
        loss = optax.softmax_cross_entropy_with_integer_labels(pred_ids, target_tokens) * loss_mask
        n_targets = jnp.sum(loss_mask)
        loss = jax.lax.cond(n_targets > 0, lambda x: jnp.sum(x), lambda _: 0.0, loss)
        return loss, (n_targets, rnn_state)
    
    value_grad_fn = eqx.filter_value_and_grad(loss_fn, has_aux=True)

    def scannable_value_grad_fn(state: Tuple[Array, Array, eqx.Module, LSTMState], subsequence: Dict[str, Array]):
        loss_sum, total_labels, grads_sum, rnn_state = state
        (loss, (n_labels, rnn_state)), grads = value_grad_fn(model, rnn_state, subsequence)

        loss_sum += loss
        total_labels += n_labels
        grads_sum = jax.tree.map(lambda x, y: x + y, grads_sum, grads)
        
        # return (loss_sum, total_labels, grads_sum, rnn_state), loss
        return (loss, n_labels, grads, rnn_state), loss
        
    # Pad the sequence
    remainder = sequence['input_ids'].shape[0] % tbptt_window
    n_pad = (tbptt_window - remainder) % tbptt_window
    sequence = jax.tree.map(
        lambda x: jnp.pad(x, (0, n_pad), mode='constant'),
        sequence,
    )

    # Split the input sequence into subsequences of length tbptt_window
    subsequences = jax.tree.map(
        lambda x: x.reshape((-1, tbptt_window)),
        sequence,
    )

    zero_grads = jax.tree.map(lambda x: jnp.zeros_like(x), model)

    (loss, total_labels, grads, rnn_state), _ = \
        jax.lax.scan(scannable_value_grad_fn, (0, 0, zero_grads, rnn_state), subsequences)

    loss /= total_labels
    grads = jax.tree.map(lambda x: x / total_labels, grads)

    return loss, grads, rnn_state


def apply_grads(
        train_state: TrainState,
        model: eqx.Module,
        grads: eqx.Module,
    ):
    # Replace nan grads with 0
    grads = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0, x), grads)
    updates, new_opt_state = train_state.tx_update_fn(grads, train_state.opt_state, model)
    new_model = eqx.apply_updates(model, updates)

    train_state = tree_replace(train_state, opt_state=new_opt_state)

    return train_state, new_model