"""Training state management for A2C agent"""

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


def init_opt(params : Any, learning_rate : float):
    # Initialize optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),   # try values 0.3 – 1.0 depending on stability
        optax.apply_if_finite(
            optax.adam(learning_rate),
            max_consecutive_errors=100,
        ),
    )
    opt_state = optimizer.init(params)
    return opt_state

def create_train_state(
    rng_key: jnp.ndarray,
    obs_size: int,
    hidden_size: int,
    num_envs: int,
    learning_rate: float,
    params: Any,
) -> TrainState:
    """Initialize training state"""
    
    # Initialize hidden states for all environments
    actor_hidden = jnp.zeros((num_envs, hidden_size))
    critic_hidden = jnp.zeros((num_envs, hidden_size))
    
    # Initialize previous step info
    prev_obs = jnp.zeros((num_envs, obs_size))
    prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((num_envs,))

    
    return TrainState(
        params=params,
        opt_state=init_opt(params, learning_rate),
        rng_key=rng_key,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        prev_obs=prev_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        learning_rate=learning_rate,
        grads=None,
    )