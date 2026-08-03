"""Training state management for A2C agent"""

import jax
import jax.numpy as jnp
from flax import struct
import optax
from typing import Any


@struct.dataclass
class TrainState:
    """Training state containing all mutable components"""
    params: Any
    opt_state: Any
    rng_key: jnp.ndarray
    # RNN hidden states for all environments (NUM_ENVS, hidden_size)
    actor_hidden: jnp.ndarray
    critic_hidden: jnp.ndarray
    # Previous step info for network input
    prev_obs: jnp.ndarray      # (NUM_ENVS, obs_size)
    prev_action: jnp.ndarray   # (NUM_ENVS,)
    prev_reward: jnp.ndarray   # (NUM_ENVS,)
    learning_rate: float
    grads: jnp.ndarray
    # Persistent per-(obs, action) credit filters (continue across update blocks). (NUM_ENVS, obs_size, action_size)
    action_elig: Any = None            # per-action low-pass of the triggering observation
    action_credit_reward: Any = None   # low-passed (obs eligibility * reward) -> reward-stream credit matrix
    action_credit_pred: Any = None     # low-passed (obs eligibility * reward prediction) -> prediction-stream credit matrix
    critic_lr_scale: float = 1.0  # critic (reward-predictor) LR multiplier; <1 -> critic learns slower


def _critic_label_tree(params):
    """Label each param leaf 'critic' (reward-predictor MLP) or 'actor' (everything else),
    so the two groups can be given different learning rates via optax.multi_transform."""
    def label(path, _leaf):
        keys = [getattr(k, 'key', str(k)) for k in path]
        return 'critic' if any('reward_pred' in str(k) for k in keys) else 'actor'
    return jax.tree_util.tree_map_with_path(label, params)


def make_optimizer(params: Any, learning_rate: float, critic_lr_scale: float = 1.0):
    """Adam with a separate (typically smaller) learning rate for the critic/reward-predictor
    params, wrapped in the shared global-norm clip + apply_if_finite guard."""
    tx = optax.multi_transform(
        {
            'actor': optax.adam(learning_rate),
            'critic': optax.adam(learning_rate * critic_lr_scale),
        },
        _critic_label_tree(params),
    )
    return optax.chain(
        optax.clip_by_global_norm(0.5),   # try values 0.3 – 1.0 depending on stability
        optax.apply_if_finite(tx, max_consecutive_errors=100),
    )


def init_opt(params : Any, learning_rate : float, critic_lr_scale : float = 1.0):
    # Initialize optimizer
    optimizer = make_optimizer(params, learning_rate, critic_lr_scale)
    opt_state = optimizer.init(params)
    return opt_state

def create_train_state(
    rng_key: jnp.ndarray,
    obs_size: int,
    hidden_size: int,
    num_envs: int,
    learning_rate: float,
    params: Any,
    critic_lr_scale: float = 1.0,
) -> TrainState:
    """Initialize training state"""
    
    # Initialize hidden states for all environments
    actor_hidden = jnp.zeros((num_envs, hidden_size))
    critic_hidden = jnp.zeros((num_envs, hidden_size))
    
    # Initialize previous step info
    prev_obs = jnp.zeros((num_envs, obs_size))
    prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((num_envs,))

    # Persistent per-(obs, action) credit filters (action_size fixed at 2)
    action_elig = jnp.zeros((num_envs, obs_size, 2))
    action_credit_reward = jnp.zeros((num_envs, obs_size, 2))
    action_credit_pred = jnp.zeros((num_envs, obs_size, 2))

    return TrainState(
        params=params,
        opt_state=init_opt(params, learning_rate, critic_lr_scale),
        rng_key=rng_key,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        prev_obs=prev_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        learning_rate=learning_rate,
        grads=None,
        action_elig=action_elig,
        action_credit_reward=action_credit_reward,
        action_credit_pred=action_credit_pred,
        critic_lr_scale=critic_lr_scale,
    )