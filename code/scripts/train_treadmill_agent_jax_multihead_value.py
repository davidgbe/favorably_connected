import sys
import os
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))


# Fix for JAX/Optax version compatibility
import jax
import jax.numpy as jnp
# jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_debug_nans', True)

# Handle DeviceArray deprecation
if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = jax.Array

from jax import random, lax
from flax import struct, serialization
from flax.training import checkpoints
from flax import linen as nn
from flax.traverse_util import flatten_dict
import optax
from typing import Tuple, Dict, Any, Optional
from functools import partial
import numpy as np
from tqdm.auto import trange
import argparse
import pickle
from datetime import datetime
from aux_funcs import zero_pad
from agents.a2c_rnn_flax_multihead_value import A2CRNNFlax, init_network_and_params
from environments.components.train_state import TrainState, create_train_state, init_opt
from environments.components.treadmill_trajectory_multi_value import TrajectoryData, collect_trajectory
# Import your existing JAX environment
from environments.treadmill_env_jax import (
    TreadmillEnvironment, 
    TreadmillEnvParams, 
    TreadmillEnvState, 
    treadmill_session_default_params
)
from pprint import pprint

# Compile-time constants for JAX JIT compatibility
N_UPDATES_PER_SESSION = 50
N_STEPS_PER_UPDATE = 400


# PARSE ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument('--exp_title', metavar='et', type=str, default='run')
parser.add_argument('--noise_var', metavar='nv', type=float, default=1e-4)
parser.add_argument('--activity_reg', metavar='ar', type=float, default=1)
parser.add_argument('--gamma', metavar='g', type=float, default=0.997)
parser.add_argument('--curr_style', metavar='cs', type=str, default='fixed')
parser.add_argument('--reward_func', metavar='rf', type=str, default='exp')
parser.add_argument('--agent_type', metavar='at', type=str, default='split')
parser.add_argument('--rnn_type', metavar='rt', type=str, default='VANILLA')
parser.add_argument('--seed', metavar='s', type=int, default=0)
# New test mode arguments
parser.add_argument('--test', action='store_true', help='Run in test/evaluation mode')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for testing')
parser.add_argument('--test_sessions', type=int, default=30, help='Number of episodes to run in test mode')
parser.add_argument('--save_trajectories', action='store_true', help='Save trajectory data during testing')
args = parser.parse_args()

# Import your existing JAX environment
from environments.treadmill_env_jax import (
    TreadmillEnvironment, 
    TreadmillEnvParams, 
    TreadmillEnvState, 
    treadmill_session_default_params
)


def compute_a2c_loss(
    params: Any,
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    gamma: float,
    critic_weight: float,
    entropy_weight: float,
    env_prediction_weight: float,
    actor_value_weight: float,
    input_noise_std: float,
    var_noise: float,
    rnn_type: str,
    hidden_size: int,
    actor_gammas: jnp.ndarray,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Compute A2C loss by calling collect_trajectory with the params argument"""
    
    # Create a modified train_state that uses the params we're taking gradients w.r.t.
    modified_train_state = train_state.replace(params=params)
    
    # Call collect_trajectory - now it will use the params argument
    trajectory, final_train_state, final_env_states = collect_trajectory(
        modified_train_state,
        env_states,
        env_params,
        input_noise_std,
        var_noise,
        rnn_type,
        hidden_size,
        N_STEPS_PER_UPDATE,
    )
    
    # Now compute loss using the collected trajectory
    vmapped_nstep_return = jax.vmap(compute_n_step_returns, (0, None, 0))
    returns = vmapped_nstep_return(trajectory.rewards, gamma, trajectory.values[:, -1])

    advantages = jnp.concatenate(
        (returns[:, 1:] - trajectory.values[:, :-1], jnp.zeros((returns.shape[0], 1))),
        axis=1
    )
    # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    # actor_gamma_returns = jax.vmap(vmapped_nstep_return, (None, 0, 1))(trajectory.rewards, actor_gammas, trajectory.actor_values[:, -1, :])
    # actor_gamma_returns = jnp.transpose(actor_gamma_returns, axes=(1, 2, 0)) # get shape (batch, time_step, gamma_number)

    vmapped_nstep_non_geometric_return = jax.vmap(compute_n_step_non_geometric_return, (0, None))
    actor_gamma_returns = jax.vmap(vmapped_nstep_non_geometric_return, (None, 0))(trajectory.rewards, actor_gammas)
    actor_gamma_returns = jnp.transpose(actor_gamma_returns, axes=(1, 2, 0)) # get shape (batch, time_step, gamma_number)

    print(actor_gamma_returns.mean(axis=0).shape)
    jax.debug.print('{x}', x=actor_gamma_returns.mean(axis=0))

    actor_advantages = jnp.concatenate(
        (
            actor_gamma_returns[:, 1:, :] - trajectory.actor_values[:, :-1, :],
            jnp.zeros((actor_gamma_returns.shape[0], 1, actor_gammas.shape[0])),
        ),
        axis=1,
    )

    print(actor_advantages.shape)
    
    # Actor loss (policy gradient)
    log_probs = jax.nn.log_softmax(trajectory.logits)
    chosen_log_probs = jnp.take_along_axis(
        log_probs,
        jax.lax.stop_gradient(trajectory.actions[..., None]),
        axis=-1
    ).squeeze(-1)
    
    actor_loss = -jnp.mean(chosen_log_probs * jax.lax.stop_gradient(advantages))
    
    # Critic loss
    critic_loss = jnp.mean(advantages ** 2)

    # Actor value loss
    actor_value_loss = jnp.dot(
        jnp.mean(actor_advantages[:, :200, :]**2, axis=(0, 1)),
        ((1 - actor_gammas)**2 / actor_gammas)**2
    ).mean()
    
    # asd
    env_quality_prediction_loss = jnp.mean((trajectory.pred_environment_quality - trajectory.environment_quality) ** 2)
    
    # Entropy loss
    probs = jax.nn.softmax(trajectory.logits)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    entropy_loss = -jnp.mean(entropy)

    activity_norm = (
        jnp.linalg.norm(trajectory.actor_hidden, axis=2).mean()
        + jnp.linalg.norm(trajectory.critic_hidden, axis=2).mean()
    )
    
    # for applying an L2 weight norm
    # param_leaves = jax.tree_util.tree_leaves(final_train_state.params)
    # l2_weight_norm = jnp.sum(jnp.array(
    #     jax.tree_util.tree_map(
    #         lambda p: jnp.sum(jnp.square(p)),
    #         param_leaves,
    #     )
    # ))

    # jax.debug.print('Env quality pred err: {x}', x=env_quality_prediction_loss)
    
    # Total loss
    total_loss = (
        actor_loss
        + critic_weight * critic_loss
        + entropy_weight * entropy_loss
        # + 1e-2 * activity_norm
        + 1e-4 * activity_norm
        + env_prediction_weight * env_quality_prediction_loss
        + actor_value_weight * actor_value_loss
    )

    # jax.debug.print('total loss {x}', x=total_loss)
    # jax.debug.print('actor_value loss {x}', x= 2000 * actor_value_loss)
    # jax.debug.print('critic loss {x}', x= critic_weight * critic_loss)

    # jax.debug.print('total: {x}', x=total_loss)
    
    metrics = {
        'total_loss': total_loss,
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'entropy_loss': entropy_loss,
        'activity_loss': activity_norm,
        'mean_reward': jnp.mean(trajectory.rewards),
        'final_train_state': jax.tree_util.tree_map(lax.stop_gradient, final_train_state),
        'final_env_states': jax.tree_util.tree_map(lax.stop_gradient, final_env_states),
    }
    
    return total_loss, metrics


def compute_n_step_non_geometric_return(rewards, gamma):

    def compute_return(carry, reward):
        (i, rolling_geometric_sum, rolling_non_geometric_sum) = carry 
        new_rolling_geometric_sum = reward + gamma * rolling_geometric_sum
        new_rolling_non_geometric_sum = reward + gamma * (rolling_geometric_sum + rolling_non_geometric_sum)
        return (i+1, new_rolling_geometric_sum, new_rolling_non_geometric_sum), new_rolling_non_geometric_sum
    
    _, returns = lax.scan(
        compute_return,
        (0, 0, 0),
        rewards,
        reverse=True,
    )
    return returns


def compute_n_step_returns(rewards, gamma, v_t):
    """
    Compute n-step returns for n=0 to max_n efficiently
    
    Args:
        rewards: (batch_size, time_steps) - rewards at each timestep
        gamma: discount factor
        
    Returns:
        n_step_returns: (batch_size, time_steps, max_n+1) where [:, :, n] contains n-step returns
    """

    def compute_return(carry, reward):
        (i, rolling_sum) = carry
        rolling_sum = reward + gamma * rolling_sum
        return (i+1, rolling_sum), rolling_sum
    
    _, returns = lax.scan(
        compute_return,
        (0, v_t),
        rewards,
        reverse=True,
    )
    return returns


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size'])
def train_step(
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    gamma: float,
    actor_gammas: jnp.array,
    critic_weight: float,
    entropy_weight: float,
    env_prediction_weight: float,
    actor_value_weight: float,
    input_noise_std: float,
    action_size: int,
    hidden_size: int,
    var_noise: float,
    rnn_type: str,
) -> Tuple[TrainState, TreadmillEnvState, Dict[str, jnp.ndarray]]:
    """Single training step"""
    
    # Compute gradients
    grad_fn = jax.grad(compute_a2c_loss, has_aux=True)
    grads, metrics = grad_fn(
        train_state.params,  # This is the params argument to compute_a2c_loss
        train_state,
        env_states,
        env_params,
        gamma,
        critic_weight,
        entropy_weight,
        env_prediction_weight,
        actor_value_weight,
        input_noise_std,
        var_noise,
        rnn_type,
        hidden_size,
        actor_gammas,
    )

    metrics['grad_norm'] = optax.global_norm(grads)
    
    # Apply updates
    optimizer = optax.chain(
        optax.clip_by_global_norm(0.5),   # try values 0.3 – 1.0 depending on stability
        optax.apply_if_finite(
            optax.rmsprop(train_state.learning_rate, eps_in_sqrt=False, decay=0.99),
            max_consecutive_errors=100,
        ),
    )
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)
    train_state.params
    # Get updated states from metrics
    final_train_state = metrics['final_train_state']
    final_env_states = metrics['final_env_states']
    
    # Update training state
    final_train_state = final_train_state.replace(
        params=new_params,
        opt_state=new_opt_state,
    )
    
    return final_train_state, final_env_states, metrics

# Configuration matching your original hyperparameters
@struct.dataclass
class TrainingConfig:
    actor_gammas: jnp.array

    # Environment 
    exp_name: str = ''
    num_envs: int = 64
    patch_types_per_env: int = 3
    obs_size: int = 4  # patch_types_per_env + 1
    action_size: int = 2
    dwell_time_for_reward: int = 6
    reward_site_len: int = 3
    input_noise_std: float = 0
    reward_param_style: int = 0
    reward_func_type: int = 0
    
    # Agent params  
    hidden_size: int = 128
    critic_weight: float = 0.0785
    entropy_weight: float = 1.02e-6 # 1.02e-06
    env_prediction_weight: float = 0.001
    actor_value_weight: float = 0
    gamma: float = 0.997 # 0.987
    learning_rate: float = 2.5e-5
    var_noise: float = 1e-4
    rnn_type: str = 'GRU'
    
    # Training params (runtime configurable)
    n_sessions: int = 5000
    
    # Logging
    output_state_save_rate: int = 100


@partial(jax.jit, static_argnames=['action_size', 'hidden_size', 'var_noise', 'rnn_type'])
def run_session_updates_with_metrics(
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    gamma: float,
    actor_gammas: jnp.array,
    critic_weight: float,
    entropy_weight: float,
    env_prediction_weight: float,
    actor_value_weight: float,
    input_noise_std: float,
    action_size: int,
    hidden_size: int,
    var_noise: float,
    rnn_type: str,
) -> Tuple[TrainState, TreadmillEnvState, Dict[str, jnp.ndarray]]:
    """Run all training updates with full metrics collection"""
    
    def update_step(carry, _):
        train_state, env_states = carry
        
        new_train_state, new_env_states, metrics = train_step(
            train_state=train_state,
            env_states=env_states, 
            env_params=env_params,
            gamma=gamma,
            actor_gammas=actor_gammas,
            critic_weight=critic_weight,
            entropy_weight=entropy_weight,
            env_prediction_weight=env_prediction_weight,
            actor_value_weight=actor_value_weight,
            input_noise_std=input_noise_std,
            action_size=action_size,
            hidden_size=hidden_size,
            var_noise=var_noise,
            rnn_type=rnn_type,
        )
        
        return (new_train_state, new_env_states), metrics
    
    # Run scan over all updates
    (final_train_state, final_env_states), all_metrics = lax.scan(
        update_step,
        (train_state, env_states),
        None,
        length=N_UPDATES_PER_SESSION,
    )

    jax.debug.print('grad_norm: {x}', x=all_metrics['grad_norm'])
    jax.debug.print('activity_norm: {x}', x=all_metrics['activity_loss'])
    
    return final_train_state, final_env_states, all_metrics


def train_a2c_jax(config: TrainingConfig = None, load_path: str = None, train: bool = True):
    """Main training function that matches your existing structure"""
    
    if config is None:
        config = TrainingConfig()
    
    print("Starting JAX A2C Training...")
    print(f"Num envs: {config.num_envs}")
    print(f"Sessions: {config.n_sessions}")
    print(f"Updates per session: {N_UPDATES_PER_SESSION}")
    print(f"Steps per update: {N_STEPS_PER_UPDATE}")
    
    # Initialize everything
    rng_key = random.key(args.seed)
    env_params = treadmill_session_default_params()
    env_params = env_params.replace(
        reward_param_style=config.reward_param_style,
        reward_func_type=config.reward_func_type,
    )

    net_init_key, rng_key = random.split(rng_key)

    network, params = init_network_and_params(
        hidden_size=config.hidden_size,
        action_size=config.action_size,
        obs_size=config.obs_size,
        rnn_type=config.rnn_type,
        var_noise=config.var_noise,
        rng_key=net_init_key,
    )
    
    # Create training state
    train_state = create_train_state(
        rng_key=rng_key,
        obs_size=config.obs_size,
        hidden_size=config.hidden_size,
        num_envs=config.num_envs,
        learning_rate=config.learning_rate,
        params=params,
    )

    # Load pretrained model if path is given
    if load_path is not None:
        print(f"Loading pretrained model from {load_path}")
        restored = checkpoints.restore_checkpoint(ckpt_dir=load_path, target=None)
        # restored can be just params or a dict depending on how saved
        if "params" in restored:
            params = restored["params"]
        else:
            params = restored
        train_state = train_state.replace(params=params)
    
    print(f"Initialized network with {config.hidden_size} hidden units")
    
    # Initialize environments
    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()
    rng_key, reset_key = random.split(train_state.rng_key)
    reset_keys = random.split(reset_key, config.num_envs)
    
    obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)
    train_state = train_state.replace(prev_obs=obs)
    print(f"Initialized {config.num_envs} environments")
    
    # Storage for logging (matching your original structure)
    all_session_rewards = []

    save_dir_rewards = Path(f'exp_reward_rates/{config.exp_name}').resolve()  # makes it absolute
    save_dir_rewards.mkdir(parents=True, exist_ok=True)

    save_dir = Path(f"checkpoints/{config.exp_name}").resolve()  # makes it absolute
    save_dir.mkdir(parents=True, exist_ok=True)

    save_dir_rewards = save_dir / '_reward_rates'
    save_dir_rewards.mkdir(parents=True, exist_ok=True)
    
    # Training loop (outer loop stays in Python for logging)
    for session_num in trange(config.n_sessions, desc='Sessions'):
        
        avg_rewards_per_update = np.empty((config.num_envs, N_UPDATES_PER_SESSION))
        all_info = []

        # Reset environment for new episode
        rng_key, reset_key = random.split(train_state.rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)

        # Reset hidden states
        train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            critic_hidden=jnp.zeros((config.num_envs, config.hidden_size)),
            prev_action=jnp.zeros((config.num_envs,), dtype=jnp.int32),
            prev_reward=jnp.zeros((config.num_envs,)),
            prev_obs=obs,
        )
        
        train_state, env_states, all_metrics = run_session_updates_with_metrics(
            train_state=train_state,
            env_states=env_states, 
            env_params=env_params,
            gamma=config.gamma,
            actor_gammas=config.actor_gammas,
            critic_weight=config.critic_weight,
            entropy_weight=config.entropy_weight,
            env_prediction_weight=config.env_prediction_weight,
            actor_value_weight=config.actor_value_weight,
            input_noise_std=config.input_noise_std,
            action_size=config.action_size,
            hidden_size=config.hidden_size,
            var_noise=config.var_noise,
            rnn_type=config.rnn_type,
        )

        # pprint(train_state.params)

        avg_rewards_per_update = all_metrics['mean_reward']
        grad_norms = all_metrics['grad_norm']
        print('grad_norms')
        print('mean:', jnp.mean(grad_norms), 'std:', jnp.std(grad_norms))
        # print(grad_norms)
        # weight_norms = all_metrics['weight_norm']
        # print('weight mean:', jnp.mean(weight_norms), 'std:', jnp.std(weight_norms))
        # losses = all_metrics['total_loss']
        # print('loss mean:', jnp.mean(losses), 'std:', jnp.std(losses))
        # print()

            
        # Session-level logging
        session_mean_reward = np.mean(avg_rewards_per_update)
        all_session_rewards.append(session_mean_reward)
        
        print(f'Session {session_num}: Avg reward = {session_mean_reward:.4f}')
        
        # Periodic saving (matching your original save rate)
        if session_num % config.output_state_save_rate == 0:
            print(f"Save checkpoint at session {session_num}")
            save_dir = Path(f"checkpoints/{config.exp_name}").resolve()  # makes it absolute
            checkpoints.save_checkpoint(
                ckpt_dir=str(save_dir),
                target={"params": train_state.params},  # or just train_state
                step=session_num,
                overwrite=True, # allows overwriting instead of accumulating
                keep=100,
            )

        sn = zero_pad(session_num, 6)
        with open(save_dir_rewards / f'{sn}', 'ab') as f:
            np.save(f, avg_rewards_per_update)
    
    print("Training completed!")
    return train_state, all_session_rewards


def evaluate_a2c_jax(config: TrainingConfig, checkpoint_path: str, save_trajectories: bool = False):
    """Evaluate trained A2C agent without gradient updates"""
    
    print("Starting JAX A2C Evaluation...")
    print(f"Loading checkpoint from: {checkpoint_path}")
    print(f"Num episodes: {config.n_sessions}")
    print(f"Save trajectories: {save_trajectories}")
    
    # Initialize everything
    rng_key = random.key(args.seed)
    env_params = treadmill_session_default_params()
    env_params = env_params.replace(
        reward_param_style=config.reward_param_style,
        reward_func_type=config.reward_func_type,
    )

    session_steps = N_UPDATES_PER_SESSION * N_STEPS_PER_UPDATE

    net_init_key, rng_key = random.split(rng_key)

    network, params = init_network_and_params(
        hidden_size=config.hidden_size,
        action_size=config.action_size,
        obs_size=config.obs_size,
        rnn_type=config.rnn_type,
        var_noise=config.var_noise,
        rng_key=net_init_key,
    )
    
    # Create training state (just for structure, won't be updated)
    train_state = create_train_state(
        rng_key=rng_key,
        obs_size=config.obs_size,
        hidden_size=config.hidden_size,
        num_envs=1,  # Use single environment for cleaner episode tracking
        learning_rate=config.learning_rate,
        params=params,
    )

    # Load trained model
    print(f"Loading trained model from {checkpoint_path}")
    restored = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=None)
    if "params" in restored:
        params = restored["params"]
    else:
        params = restored
    train_state = train_state.replace(params=params)
    print("Model loaded successfully")
    
    # Initialize environment
    reset_fn, step_fn, get_obs_fn = TreadmillEnvironment()
    
    # Storage for results
    all_episode_rewards = []
    all_trajectories = [] if save_trajectories else None
    
    # Run evaluation episodes
    for episode in trange(config.n_sessions, desc='Sessions'):

        # Reset environment for new episode
        rng_key, reset_key = random.split(train_state.rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)

        # Reset hidden states
        train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=jnp.zeros((1, config.hidden_size)),
            critic_hidden=jnp.zeros((1, config.hidden_size)),
            prev_action=jnp.zeros((1,), dtype=jnp.int32),
            prev_reward=jnp.zeros((1,)),
            prev_obs=obs,
        )

        # Run episode (using a reasonable episode length)
        trajectory, final_train_state, final_env_states = collect_trajectory(
            train_state=train_state,
            env_states=env_states,
            env_params=env_params,
            input_noise_std=config.input_noise_std,  # No noise during evaluation
            var_noise=config.var_noise,
            rnn_type=config.rnn_type,
            hidden_size=config.hidden_size,
            n_steps=session_steps,
        )
        
        # Extract episode metrics
        episode_reward = float(jnp.sum(trajectory.rewards))
        
        all_episode_rewards.append(episode_reward)
        
        # Save trajectory if requested
        if save_trajectories:
            # Convert JAX arrays to numpy for easier saving
            traj_no_batch = jax.tree_util.tree_map(lambda x: x[0], trajectory)
            trajectory_dict = serialization.to_state_dict(traj_no_batch)
            all_trajectories.append(trajectory_dict)
        
        # Update rng for next episode
        train_state = final_train_state
    
    # Compute summary statistics
    mean_reward_rate = np.mean(all_episode_rewards) / session_steps
    std_reward_rate = np.std(all_episode_rewards) / session_steps
    
    print("\nEvaluation Summary:")
    print(f"Mean episode reward rate: {mean_reward_rate:.4f} ± {std_reward_rate:.4f}")
    print(f"Min/Max reward rates: {np.min(all_episode_rewards) / session_steps:.4f} / {np.max(all_episode_rewards) / session_steps:.4f}")
    
    # Save results
    results = {
        'episode_rewards': all_episode_rewards,
        'mean_reward_rate': mean_reward_rate,
        'std_reward_rate': std_reward_rate,
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }
    
    # Create results directory
    results_dir = Path(f"results/{config.exp_name}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save summary results
    results_file = results_dir / f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    print(f"Results saved to {results_file}")
    
    # Save trajectories if requested
    if save_trajectories and all_trajectories:
        trajectories_file = results_dir / f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        with open(trajectories_file, 'wb') as f:
            pickle.dump(all_trajectories, f)
        print(f"Trajectories saved to {trajectories_file}")
    
    return results, all_trajectories


def reward_param_style_str_to_int(style):
    if style == 'fixed':
        return 0
    elif style == 'indep':
        return 1
    elif style == 'coupled':
        return 2
    else:
        raise NotImplementedError('reward param style not found')
    

def reward_func_type_str_to_int(func_type):
    if func_type == 'exp':
        return 0
    elif func_type == 'block':
        return 1
    elif func_type == 'markov':
        return 2
    else:
        raise NotImplementedError('reward func type not found')


def main():
    """Entry point for training or evaluation"""
    time_stamp = str(datetime.now()).replace(' ', '_')
    exp_name = f'{args.exp_title}_seed_{args.seed}_{time_stamp}'
    
    if args.test:
        # Test/Evaluation mode
        if args.checkpoint_path is None:
            print("Error: --checkpoint_path required for test mode")
            return
            
        # You can customize the config here
        config = TrainingConfig(
            exp_name=exp_name,
            n_sessions=args.test_sessions,
            num_envs=1,  # Single env for cleaner episode tracking
            hidden_size=64,
            rnn_type=args.rnn_type if args.rnn_type else 'VANILLA',
            reward_param_style=reward_param_style_str_to_int(args.curr_style),
            reward_func_type=reward_func_type_str_to_int(args.reward_func),
            var_noise=0,
            input_noise_std=0 #0.02,
        )
        
        print("Running in TEST mode")
        print(config)
        
        results, trajectories = evaluate_a2c_jax(
            config=config,
            checkpoint_path=os.path.join(os.getcwd(), args.checkpoint_path),
            save_trajectories=args.save_trajectories
        )
        
        print(f"\nTest completed! Mean reward: {results['mean_reward_rate']:.4f}")
        
    else:

        # Training mode (original behavior)
        config = TrainingConfig(
            exp_name=exp_name,
            n_sessions=5000,
            num_envs=128,
            learning_rate=1e-4, #1e-4 for GRU, 2e-5, smaller for relu
            entropy_weight=2.5e-3,# for relu 2.5e-5, GRU benefits from larger entropy bonus, like 2.5e-3
            critic_weight=0.05, # 0.5 originally for GRU, 0.04 for relu,
            env_prediction_weight=0, # 0.001,
            actor_value_weight=500,
            gamma=args.gamma,
            actor_gammas=jnp.array([0.95, 0.98, 0.99]),
            hidden_size=64, #64
            output_state_save_rate=50,
            rnn_type=args.rnn_type if args.rnn_type else 'VANILLA',
            reward_param_style=reward_param_style_str_to_int(args.curr_style),
            reward_func_type=reward_func_type_str_to_int(args.reward_func),
            var_noise=0,
            input_noise_std=0.02,
        )

        print("Running in TRAINING mode")
        print(config)
            
        final_train_state, rewards = train_a2c_jax(config)
        
        print("\nTraining Summary:")
        print(f"Final average reward: {np.mean(rewards[-10:]):.4f}")
        print(f"Best average reward: {np.max(rewards):.4f}")


if __name__ == "__main__":
    main()

    