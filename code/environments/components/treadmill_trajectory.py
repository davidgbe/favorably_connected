import jax
import jax.numpy as jnp
from jax import random, lax
from flax import struct
from typing import Tuple, Dict, Any, Optional
from functools import partial

from environments.treadmill_env_jax import (
    TreadmillEnvironment, 
    TreadmillEnvParams, 
    TreadmillEnvState, 
)

from environments.components.train_state import TrainState
from agents.a2c_rnn_flax import A2CRNNFlax


@struct.dataclass
class TrajectoryData:
    """Data collected during trajectory rollout"""
    action: jnp.ndarray
    current_patch_num: jnp.ndarray
    current_position: jnp.ndarray
    current_patch_start: jnp.ndarray
    agent_in_patch: jnp.ndarray
    reward_bounds: jnp.ndarray
    reward_site_idx: jnp.ndarray
    current_reward_site_attempted: jnp.ndarray
    patch_reward_param: jnp.ndarray
    reward: jnp.ndarray

    observations: jnp.ndarray  # (num_envs, n_steps, obs_size)
    actions: jnp.ndarray       # (num_envs, n_steps)
    rewards: jnp.ndarray       # (num_envs, n_steps)
    logits: jnp.ndarray       # (num_envs, n_steps, action_size)
    values: jnp.ndarray       # (num_envs, n_steps)
    dones: jnp.ndarray        # (num_envs, n_steps)

    actor_hidden: jnp.ndarray
    critic_hidden: jnp.ndarray

    environment_quality: jnp.ndarray # track expected environment quality
    pred_environment_quality: jnp.ndarray
    pred_obs: jnp.ndarray
    exp_filtered_reward_rate: jnp.ndarray
    pred_reward_rate: jnp.ndarray


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size', 'n_steps', 'obs_size'])
def collect_trajectory(
    train_state: TrainState,
    env_states: TreadmillEnvState, 
    env_params: TreadmillEnvParams,
    input_noise_std: float,
    unit_noise_std: float,
    rnn_type: str,
    hidden_size: int,
    obs_size: int,
    n_steps: int,
) -> Tuple[TrajectoryData, TrainState, TreadmillEnvState]:
    """Collect trajectory using lax.scan over time steps"""
    
    network = A2CRNNFlax(
        action_size=2,  # Fixed ACTION_SIZE
        hidden_size=hidden_size,  # This should come from config
        unit_noise_std=unit_noise_std,   # This should come from config,
        rnn_type=rnn_type,
        obs_size=obs_size,
    )
    
    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()
    
    def scan_step(carry, _):
        train_state, env_states = carry
        rng_key = train_state.rng_key
        
        # Sample actions using current observations (from previous step)
        # Create network input: [current_obs, prev_obs, prev_action, prev_reward]
        prev_action_one_hot = jax.nn.one_hot(train_state.prev_action, num_classes=2)  # Assuming 2 actions
        network_input = jnp.concatenate([
            train_state.prev_obs,                    # Current observations  
            prev_action_one_hot,  # Previous action as scalar
            train_state.prev_reward[..., None],      # Previous reward
        ], axis=-1)
        
        # Add input noise
        rng_key, noise_key = random.split(rng_key)
        obs_noise = random.normal(noise_key, network_input.shape) * input_noise_std
        network_input = network_input + obs_noise

        rng_key, network_noise_key = random.split(rng_key)

        # Forward pass through network
        logits, values, new_actor_hidden, new_critic_hidden, pred_env_quality, pred_obs, pred_reward_rate = network.apply(
            train_state.params,
            network_input,
            train_state.actor_hidden,
            train_state.critic_hidden,
            rngs={'noise': network_noise_key} if train_state.params  else {}  # Simple check
        )
        
        # Sample actions
        rng_key, action_key = random.split(rng_key)
        action_keys = random.split(action_key, logits.shape[0])
        actions = jax.vmap(
            lambda key, logit: random.categorical(key, logit)
        )(action_keys, logits)
        
        # Step environments with sampled actions
        rng_key, step_key = random.split(rng_key)
        step_keys = random.split(step_key, actions.shape[0])
        step_results = jax.vmap(
            lambda key, state, action: step_fn(key, state, action, env_params)
        )(step_keys, env_states, actions)
        
        # Unpack step results
        new_obs, new_env_states, rewards, dones, infos = step_results

        alpha = 0.9997
        new_reward_rate = alpha * new_env_states.exp_filtered_reward_rate + (1.0 - alpha) * rewards

        new_env_states = new_env_states.replace(
            exp_filtered_reward_rate=new_reward_rate,
        )
        
        # Update train state with new info
        new_train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=new_actor_hidden,
            critic_hidden=new_critic_hidden,
            prev_obs=new_obs,        # Store current obs as previous for next step
            prev_action=actions,     # Store current action as previous
            prev_reward=rewards,     # Store current reward as previous
        )
        
        # Return step data (using the observations that were actually used for decisions)
        step_data = {
            'observations': train_state.prev_obs,  # The obs that led to these actions
            'actions': actions,
            'rewards': rewards, 
            'logits': logits,
            'values': values,
            'dones': dones,
            'actor_hidden': train_state.actor_hidden,
            'critic_hidden': train_state.critic_hidden,
            'pred_environment_quality': pred_env_quality,
            'pred_obs': pred_obs,
            'exp_filtered_reward_rate': new_env_states.exp_filtered_reward_rate,
            'pred_reward_rate': pred_reward_rate,
        } | infos
        
        return (new_train_state, new_env_states), step_data
    
    # Run scan over time steps using compile-time constant
    (final_train_state, final_env_states), trajectory_data = lax.scan(
        scan_step,
        (train_state, env_states),
        None,
        length=n_steps  # Now a compile-time constant
    )
    
    # Reshape trajectory data from (n_steps, num_envs, ...) to (num_envs, n_steps, ...)
    trajectory_data = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), trajectory_data
    )
    
    trajectory = TrajectoryData(**trajectory_data)
    
    return trajectory, final_train_state, final_env_states