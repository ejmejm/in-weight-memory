import equinox as eqx
import jax
import jax.numpy as jnp
import optax

from models import SupervisedModel
from tasks import gen_train_sequence
from training import apply_grads, supervised_loss_and_grads, TrainState
from utils import tree_replace


NAME_VOCAB_SIZE = 26
VAL_VOCAB_SIZE = 10
VOCAB_SIZE = NAME_VOCAB_SIZE + VAL_VOCAB_SIZE + 3


def train_loop(train_state, model, tbptt_window, iters=50):
    rngs = jax.random.split(train_state.rng, iters + 1)
    train_state = tree_replace(train_state, rng=rngs[0])
        
    gen_sequences = jax.vmap(gen_train_sequence, (0, None, None, None, None, None))

    seq_rngs = jnp.array([jax.random.PRNGKey(0) for _ in range(iters)])
    train_sequences = gen_sequences(
        rngs[1:],
        # seq_rngs,
        2, 2, 3, NAME_VOCAB_SIZE, VAL_VOCAB_SIZE)

    def train_iter(loop_state, train_sequence):
        train_state, model = loop_state
        loss, grads, _ = supervised_loss_and_grads(
            model,
            model.init_rnn_state(),
            train_sequence,
            tbptt_window,
        )
        train_state, new_model = apply_grads(
            train_state,
            model,
            grads,
        )
        return (train_state, new_model), loss

    (train_state, new_model), losses = jax.lax.scan(train_iter, (train_state, model), train_sequences)

    return train_state, new_model, losses


def batch_train_iter(train_state, model, tbptt_window, batch_size):
    rngs = jax.random.split(train_state.rng, batch_size + 1)
    train_state = tree_replace(train_state, rng=rngs[0])
        
    gen_sequences = jax.vmap(gen_train_sequence, (0, None, None, None, None, None))
    train_sequences = gen_sequences(
        rngs[1:], 2, 2, 3, NAME_VOCAB_SIZE, VAL_VOCAB_SIZE)
    
    v_loss_and_grads = jax.vmap(supervised_loss_and_grads, (None, None, 0, None))

    losses, grads, _ = v_loss_and_grads(
        model,
        model.init_rnn_state(),
        train_sequences,
        tbptt_window,
    )
    loss = jnp.mean(losses)
    # Average over gradients (which are each a pytree)
    grads = jax.tree.map(lambda x: jnp.mean(x, axis=0), grads)
    
    train_state, model = apply_grads(
        train_state,
        model,
        grads,
    )

    return train_state, model, loss


def batch_train_loop(train_state, model, tbptt_window, batch_size, n_steps):
    def scannable_update(state, _):
        train_state, model = state
        train_state, model, loss = batch_train_iter(train_state, model, tbptt_window, batch_size)
        return (train_state, model), loss
    
    (train_state, model), losses = jax.lax.scan(scannable_update, (train_state, model), None, n_steps)
    return train_state, model, losses


if __name__ == '__main__':

    learning_rate = 3e-4
    tbptt_window = 40

    vocab_size = VOCAB_SIZE
    output_dim = VOCAB_SIZE
    embedding_dim = 64
    layer_sizes = [64, 128]
    recurrent_layer_indices = [1]


    rng = jax.random.PRNGKey(0)
    model_key, train_key, rng = jax.random.split(rng, 3)

    model = SupervisedModel(
        rng = model_key,
        vocab_size = vocab_size,
        embedding_dim = embedding_dim,
        layer_sizes = layer_sizes,
        output_dim = output_dim,
        recurrent_layer_indices = recurrent_layer_indices,
    )

    print(model)
    
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(model)

    train_state = TrainState(
        rng = train_key,
        opt_state = opt_state,
        tx_update_fn = optimizer.update,
    )

    jax_cpu = jax.devices(backend='cpu')[0]
    train_fn = eqx.filter_jit(train_loop, device=jax_cpu)

    train_state, model, losses = train_fn(train_state, model, tbptt_window, iters=1000)
    print(f"Losses: {losses}")

    print("Training completed. Final loss:", losses[-1])


    # train_fn = eqx.filter_jit(batch_train_loop)

    # train_state, model, losses = train_fn(train_state, model, tbptt_window, batch_size=32, n_steps=1000)
    # print(f"Losses: {losses}")
