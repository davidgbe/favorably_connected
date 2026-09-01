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
    luck_filter: Any = None            # (NUM_ENVS,) persistent EMA of (reward - reward_prediction): "recent luck"
    # reward-predictor LR multiplier; <1 -> reward-pred net learns slower. Static (pytree_node=False)
    # so make_optimizer can branch on it inside jit (it selects plain-Adam vs multi_transform).
    reward_pred_lr_scale: float = struct.field(pytree_node=False, default=1.0)


def _reward_pred_label_tree(params):
    """Label each param leaf 'reward_pred' (reward-predictor MLP) or 'other' (everything else),
    so the two groups can be given different learning rates via optax.multi_transform."""
    def label(path, _leaf):
        keys = [getattr(k, 'key', str(k)) for k in path]
        return 'reward_pred' if any('reward_pred' in str(k) for k in keys) else 'other'
    return jax.tree_util.tree_map_with_path(label, params)


def make_optimizer(params: Any, learning_rate: float, reward_pred_lr_scale: float = 1.0):
    """Adam wrapped in the shared global-norm clip + apply_if_finite guard.

    When reward_pred_lr_scale == 1.0 (the default) this is a single plain Adam chain -- structurally
    identical to what train_steps that build the optimizer inline produce, so their opt_state stays
    compatible. Only when a non-unit scale is requested do we split into a multi_transform that gives
    the reward-predictor params a separately-scaled learning rate. (reward_pred_lr_scale must be a
    static Python value, since this branch is taken inside jit in some train_steps.)"""
    if reward_pred_lr_scale == 1.0:
        tx = optax.adam(learning_rate)
    else:
        tx = optax.multi_transform(
            {
                'other': optax.adam(learning_rate),
                'reward_pred': optax.adam(learning_rate * reward_pred_lr_scale),
            },
            _reward_pred_label_tree(params),
        )
    return optax.chain(
        optax.clip_by_global_norm(0.5),   # try values 0.3 – 1.0 depending on stability
        optax.apply_if_finite(tx, max_consecutive_errors=100),
    )


def init_opt(params : Any, learning_rate : float, reward_pred_lr_scale : float = 1.0):
    # Initialize optimizer
    optimizer = make_optimizer(params, learning_rate, reward_pred_lr_scale)
    opt_state = optimizer.init(params)
    return opt_state

def create_train_state(
    rng_key: jnp.ndarray,
    obs_size: int,
    hidden_size: int,
    num_envs: int,
    learning_rate: float,
    params: Any,
    reward_pred_lr_scale: float = 1.0,
) -> TrainState:
    """Initialize training state"""
    
    # Initialize hidden states for all environments
    actor_hidden = jnp.zeros((num_envs, hidden_size))
    critic_hidden = jnp.zeros((num_envs, hidden_size))
    
    # Initialize previous step info
    prev_obs = jnp.zeros((num_envs, obs_size))
    prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((num_envs,))

    # Persistent per-(feature, action) credit filters (action_size fixed at 2). The credit feature
    # dimension is obs_size + 1: the observation plus the reward history channel (see compute_a2c_loss).
    credit_feat_dim = obs_size + 1
    action_elig = jnp.zeros((num_envs, credit_feat_dim, 2))
    action_credit_reward = jnp.zeros((num_envs, credit_feat_dim, 2))
    action_credit_pred = jnp.zeros((num_envs, credit_feat_dim, 2))
    luck_filter = jnp.zeros((num_envs,))

    return TrainState(
        params=params,
        opt_state=init_opt(params, learning_rate, reward_pred_lr_scale),
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
        luck_filter=luck_filter,
        reward_pred_lr_scale=reward_pred_lr_scale,
    )