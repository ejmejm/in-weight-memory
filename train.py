from functools import partial
import math
import time
from typing import Callable, Tuple

import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import omegaconf
from omegaconf import DictConfig
import optax
from tqdm import tqdm

from models.xlstm import xLSTM, xLSTMState
from tasks import ContinualARState, next_associative_recall_obs
from training import apply_grads, supervised_loss_and_grads, TrainState


# jax.config.update("jax_debug_nans", True)


def create_model(
        rng: PRNGKeyArray,
        model_config: DictConfig,
        half_precision: bool = False,
    ) -> eqx.Module:
    model = xLSTM(
        vocab_size = model_config.vocab_size,
        hidden_dim = model_config.hidden_dim,
        n_blocks = model_config.n_blocks,
        n_heads = model_config.n_heads,
        ms_ratio = model_config.ms_ratio,
        mlstm_kwargs = model_config.get('mlstm_kwargs', None),
        slstm_kwargs = model_config.get('slstm_kwargs', None),
        penultimate_norm = model_config.penultimate_norm,
        key = rng,
    )

    if half_precision:
        model = jax.tree.map(lambda x: x.astype(jnp.bfloat16), model)

    return model


def create_optimizer(
    model: eqx.Module, optimizer_config: DictConfig,
) -> Tuple[optax.GradientTransformation, optax.OptState]:
    optimizer = optax.adam(optimizer_config.learning_rate)
    opt_state = optimizer.init(model)
    return optimizer, opt_state


def init_environment(rng: PRNGKeyArray, env_config: DictConfig) -> Tuple[ContinualARState, Callable]:
    state = ContinualARState(rng = rng, **env_config)
    return state


def batch_train_iter(
        train_state: TrainState,
        env_states: ContinualARState,
        rnn_states: xLSTMState,
        model: eqx.Module,
        env_step_fn: Callable,
        train_config: DictConfig,
        half_precision: bool = False,
    ):
    # Generate `tbptt_window` long sequences for each env state in the batch
    batch_env_step_fn = jax.vmap((lambda state, _: env_step_fn(state)), in_axes=(0, None))
    env_states, train_sequences = jax.lax.scan(batch_env_step_fn, env_states, length=train_config.tbptt_window)
    train_sequences = jax.tree.map(jnp.transpose, train_sequences)
    if half_precision:
        train_sequences['loss_mask'] = train_sequences['loss_mask'].astype(jnp.bfloat16)
    
    batch_loss_and_grads = jax.vmap(supervised_loss_and_grads, (None, 0, 0))
    losses, grads, accuracies, rnn_states = batch_loss_and_grads(model, rnn_states, train_sequences)
    loss = jnp.mean(losses)
    accuracy = jnp.mean(accuracies)
    grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads) # Average over gradients
    
    train_state, model = apply_grads(
        train_state,
        model,
        grads,
    )

    return train_state, env_states, rnn_states, model, accuracy, loss


@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def train_loop(
        train_state: TrainState,
        env_states: ContinualARState,
        rnn_states: xLSTMState,
        model: eqx.Module,
        env_step_fn: Callable,
        train_config: DictConfig,
        train_steps: int,
        half_precision: bool = False,
    ):
    def train_step(carry, _):
        train_state, env_states, rnn_states, model = carry
        train_state, env_states, rnn_states, model, accuracy, loss = batch_train_iter(
            train_state, env_states, rnn_states, model, env_step_fn, train_config, half_precision)
        return (train_state, env_states, rnn_states, model), (accuracy, loss)

    (train_state, env_states, rnn_states, model), (accuracies, losses) = jax.lax.scan(
        train_step, (train_state, env_states, rnn_states, model), length=train_steps)
    return train_state, env_states, rnn_states, model, accuracies, losses


def train(
        train_state: TrainState,
        env_states: ContinualARState,
        rnn_states: xLSTMState,
        model: eqx.Module,
        env_step_fn: Callable,
        config: DictConfig,
    ):
    train_config = config.train
    steps_per_log = train_config.log_interval // (train_config.tbptt_window * train_config.batch_size)
    total_steps = train_config.steps // (train_config.tbptt_window * train_config.batch_size)
    
    env_steps_passed = 0
    train_steps_passed = 0

    with tqdm(total=train_config.steps) as pbar:
        for _ in range(total_steps // steps_per_log):
            train_state, env_states, rnn_states, model, accuracies, losses = train_loop(
                train_state, env_states, rnn_states, model, env_step_fn,
                train_config, steps_per_log, config.get('half_precision', False),
            )

            avg_loss = jnp.nanmean(losses)
            avg_accuracy = jnp.nanmean(accuracies)

            if jnp.isnan(avg_loss):
                print('Loss is nan, breakpoint!')

            train_steps_passed += steps_per_log
            env_steps_passed += steps_per_log * train_config.tbptt_window * train_config.batch_size

            pbar.update(steps_per_log * train_config.tbptt_window * train_config.batch_size)
            pbar.set_postfix({
                'avg_loss': avg_loss,
                'avg_accuracy': avg_accuracy,
                'train_steps': train_steps_passed,
                'env_steps': env_steps_passed
            })

            if config.wandb.get('enabled', False):
                import wandb
                wandb.log({
                    'loss': avg_loss,
                    'accuracy': avg_accuracy,
                    'train_step': train_steps_passed,
                    'env_step': env_steps_passed
                })


@hydra.main(config_path='conf', config_name='train_base')
def main(config: DictConfig) -> None:
    print('Config:\n', config)

    if config.wandb.get('enabled', False):
        import wandb
        wandb.config = omegaconf.OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True)
        wandb.init(entity=config.wandb.get('entity'), project=config.wandb.get('project'))

    rng = jax.random.PRNGKey(config.get('seed', time.time_ns()))
    model_key, env_key, rng = jax.random.split(rng, 3)

    # Prepare model
    model = create_model(model_key, config.model, config.half_precision)
    rnn_states = jax.vmap(lambda _: model.init_rnn_state())(jnp.arange(config.train.batch_size))
    if config.half_precision:
        rnn_states = jax.tree.map(lambda x: x.astype(jnp.bfloat16), rnn_states)

    print('# Model params:', sum(jax.tree.leaves(jax.tree.map(lambda x: math.prod(x.shape), model))))

    # Prepare optimizer
    optimizer, opt_state = create_optimizer(model, config.optimizer)

    # Prepare environment(s)
    env_keys = jax.random.split(env_key, config.train.batch_size)
    env_states = jax.vmap(init_environment, in_axes=(0, None))(env_keys, config.env)
    env_step_fn = next_associative_recall_obs

    # Train
    train_state = TrainState(rng, opt_state, optimizer.update)
    train(train_state, env_states, rnn_states, model, env_step_fn, config)


if __name__ == '__main__':
    main()
