import sys
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))


# Fix for JAX/Optax version compatibility
import jax
import jax.numpy as jnp

# Handle DeviceArray deprecation
if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = jax.Array

from jax import random, lax
import chex
from flax import struct
from flax.training import checkpoints
from flax import linen as nn
import optax
from typing import Tuple, Dict, Any, Optional
from functools import partial
import numpy as np
from tqdm.auto import trange

# Import your existing JAX environment
from environments.treadmill_env_jax import (
    TreadmillEnvironment, 
    TreadmillEnvParams, 
    TreadmillEnvState, 
    treadmill_session_default_params
)


@struct.dataclass
class TrainState:
    """Training state containing all mutable components"""
    params: Any
    opt_state: Any
    rng_key: chex.PRNGKey
    # RNN hidden states for all environments (NUM_ENVS, hidden_size)
    actor_hidden: jnp.ndarray
    critic_hidden: jnp.ndarray
    # Previous step info for network input
    prev_obs: jnp.ndarray      # (NUM_ENVS, obs_size)
    prev_action: jnp.ndarray   # (NUM_ENVS,)
    prev_reward: jnp.ndarray   # (NUM_ENVS,)
    learning_rate: float
    grads: jnp.ndarray


class A2CRNNFlax(nn.Module):
    """Flax implementation of A2C RNN network"""
    action_size: int
    hidden_size: int
    rnn_type: str = 'GRU'  # 'VANILLA' or 'GRU'
    var_noise: float = 1e-4
    
    def setup(self):
        if self.rnn_type == 'VANILLA':
            self.rnn_actor = VanillaRNNCell(self.hidden_size, self.var_noise)
            self.rnn_critic = VanillaRNNCell(self.hidden_size, self.var_noise)
        elif self.rnn_type == 'GRU':
            self.rnn_actor = nn.GRUCell(self.hidden_size)
            self.rnn_critic = nn.GRUCell(self.hidden_size)
        else:
            raise ValueError(f"Unknown RNN type: {self.rnn_type}")
            
        # Actor and critic heads
        self.actor = nn.Dense(self.action_size)
        self.critic = nn.Dense(1)
        
    def __call__(self, x, actor_hidden, critic_hidden):
        """
        Args:
            x: input (batch_size, input_dim)
            actor_hidden: (batch_size, hidden_size) 
            critic_hidden: (batch_size, hidden_size)
        Returns:
            logits, value, new_actor_hidden, new_critic_hidden
        """
        # Actor RNN
        new_actor_hidden, actor_outputs = self.rnn_actor(actor_hidden, x)
        logits = self.actor(actor_outputs)
        
        # Critic RNN  
        new_critic_hidden, critic_outputs = self.rnn_critic(critic_hidden, x)
        value = self.critic(critic_outputs).squeeze(-1)
        
        return logits, value, new_actor_hidden, new_critic_hidden


class VanillaRNNCell(nn.Module):
    """Vanilla RNN cell with noise injection matching PyTorch implementation"""
    hidden_size: int
    var_noise: float = 0
    
    def setup(self):
        self.input_projection = nn.Dense(self.hidden_size, use_bias=True)
        self.hidden_projection = nn.Dense(self.hidden_size, use_bias=True)
        
    def __call__(self, x, hidden):
        """
        Args:
            x: input (batch_size, input_dim)
            hidden: previous hidden state (batch_size, hidden_size)
        Returns:
            new_hidden: (batch_size, hidden_size)
        """
        input_contrib = self.input_projection(x)
        hidden_contrib = self.hidden_projection(hidden)
        
        # Add noise like in PyTorch version
        if self.var_noise > 0:
            noise = random.normal(
                self.make_rng('noise'), 
                hidden_contrib.shape
            ) * jnp.sqrt(self.var_noise)
            hidden_contrib = hidden_contrib + noise
            
        new_hidden = input_contrib + hidden_contrib
        return new_hidden


def create_train_state(
    rng_key: chex.PRNGKey,
    obs_size: int,
    action_size: int, 
    hidden_size: int,
    num_envs: int,
    learning_rate: float,
    rnn_type: str = 'GRU',
    var_noise: float = 1e-4,
) -> TrainState:
    """Initialize training state"""
    
    # Network input size: obs + prev_obs + prev_action + prev_reward
    input_size = obs_size + action_size + 1
    
    # Initialize networktrain_state
    network = A2CRNNFlax(
        action_size=action_size,
        hidden_size=hidden_size, 
        rnn_type=rnn_type,
        var_noise=var_noise
    )
    
    # Initialize parameters
    rng_key, param_key, hidden_key = random.split(rng_key, 3)
    dummy_input = jnp.zeros((1, input_size))
    dummy_hidden = jnp.zeros((1, hidden_size))
    
    params = network.init(
        param_key, 
        dummy_input, 
        dummy_hidden, 
        dummy_hidden
    )
    
    # Initialize optimizer
    optimizer = optax.rmsprop(learning_rate)
    opt_state = optimizer.init(params)
    
    # Initialize hidden states for all environments
    actor_hidden = jnp.zeros((num_envs, hidden_size))
    critic_hidden = jnp.zeros((num_envs, hidden_size))
    
    # Initialize previous step info
    prev_obs = jnp.zeros((num_envs, obs_size))
    prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((num_envs,))
    
    return TrainState(
        params=params,
        opt_state=opt_state,
        rng_key=rng_key,
        actor_hidden=actor_hidden,
        critic_hidden=critic_hidden,
        prev_obs=prev_obs,
        prev_action=prev_action,
        prev_reward=prev_reward,
        learning_rate=learning_rate,
        grads=None,
    )


@struct.dataclass
class TrajectoryData:
    """Data collected during trajectory rollout"""
    observations: jnp.ndarray  # (num_envs, n_steps, obs_size)
    actions: jnp.ndarray       # (num_envs, n_steps)
    rewards: jnp.ndarray       # (num_envs, n_steps)
    logits: jnp.ndarray       # (num_envs, n_steps, action_size)
    values: jnp.ndarray       # (num_envs, n_steps)
    dones: jnp.ndarray        # (num_envs, n_steps)


def collect_trajectory(
    train_state: TrainState,
    env_states: TreadmillEnvState, 
    env_params: TreadmillEnvParams,
    input_noise_std: float,
) -> Tuple[TrajectoryData, TrainState, TreadmillEnvState]:
    """Collect trajectory using lax.scan over time steps"""
    
    network = A2CRNNFlax(
        action_size=2,  # Fixed ACTION_SIZE
        hidden_size=128,  # This should come from config
        var_noise=1e-4,   # This should come from config  
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
        
        # Forward pass through network
        logits, values, new_actor_hidden, new_critic_hidden = network.apply(
            train_state.params,
            network_input,
            train_state.actor_hidden,
            train_state.critic_hidden,
            rngs={'noise': rng_key} if train_state.params  else {}  # Simple check
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
        }
        
        return (new_train_state, new_env_states), step_data
    
    # Run scan over time steps using compile-time constant
    (final_train_state, final_env_states), trajectory_data = lax.scan(
        scan_step,
        (train_state, env_states),
        None,
        length=N_STEPS_PER_UPDATE  # Now a compile-time constant
    )
    
    # Reshape trajectory data from (n_steps, num_envs, ...) to (num_envs, n_steps, ...)
    trajectory_data = jax.tree.map(
        lambda x: jnp.swapaxes(x, 0, 1), trajectory_data
    )
    
    trajectory = TrajectoryData(**trajectory_data)
    
    return trajectory, final_train_state, final_env_states, 


def compute_a2c_loss(
    params: Any,
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    gamma: float,
    critic_weight: float,
    entropy_weight: float,
    input_noise_std: float,
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
    )
    
    # Now compute loss using the collected trajectory
    returns = jax.vmap(compute_n_step_returns, (0, None,))(trajectory.rewards, gamma)
    advantages = returns - trajectory.values
    
    # Actor loss (policy gradient)
    probs = jax.nn.softmax(trajectory.logits)
    log_probs = jnp.log(probs + 1e-8)
    chosen_log_probs = jnp.take_along_axis(
        log_probs,
        trajectory.actions[..., None],
        axis=-1
    ).squeeze(-1)
    
    actor_loss = -jnp.mean(chosen_log_probs * advantages)
    
    # Critic loss
    critic_loss = jnp.mean(advantages ** 2)
    
    # Entropy loss
    entropy = -jnp.sum(probs * jnp.log(probs + 1e-8), axis=-1)
    entropy_loss = -jnp.mean(entropy)
    
    # Total loss
    total_loss = actor_loss + critic_weight * critic_loss + entropy_weight * entropy_loss
    
    metrics = {
        'total_loss': total_loss,
        'actor_loss': actor_loss,
        'critic_loss': critic_loss,
        'entropy_loss': entropy_loss,
        'mean_reward': jnp.mean(trajectory.rewards),
        'final_train_state': final_train_state,
        'final_env_states': final_env_states,
    }
    
    return total_loss, metrics


def compute_n_step_returns(rewards, gamma):
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
        rolling_sum += jnp.power(gamma, i) * reward
        return (i+1, rolling_sum), rolling_sum
    
    _, returns = lax.scan(
        compute_return,
        (0, 0),
        rewards,
        reverse=True,
    )
    return returns


@partial(jax.jit, static_argnames=['rnn_type'])
def train_step(
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    gamma: float,
    critic_weight: float,
    entropy_weight: float,
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
        input_noise_std,
    )
    
    # Apply updates
    optimizer = optax.rmsprop(train_state.learning_rate)
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)
    
    # Get updated states from metrics
    final_train_state = metrics['final_train_state']
    final_env_states = metrics['final_env_states']
    
    # Update training state
    new_train_state = final_train_state.replace(
        params=new_params,
        opt_state=new_opt_state,
    )
    
    return new_train_state, final_env_states, metrics


# Compile-time constants for JAX JIT compatibility
N_UPDATES_PER_SESSION = 100
N_STEPS_PER_UPDATE = 200

# Configuration matching your original hyperparameters
@struct.dataclass
class TrainingConfig:
    # Environment params
    num_envs: int = 64
    patch_types_per_env: int = 3
    obs_size: int = 4  # patch_types_per_env + 1
    action_size: int = 2
    dwell_time_for_reward: int = 6
    reward_site_len: int = 3
    input_noise_std: float = 0.05
    
    # Agent params  
    hidden_size: int = 128
    critic_weight: float = 0.0785
    entropy_weight: float = 1.016e-06
    gamma: float = 0.987
    learning_rate: float = 2.5e-5
    var_noise: float = 1e-4
    rnn_type: str = 'GRU'
    
    # Training params (runtime configurable)
    n_sessions: int = 5000
    
    # Logging
    output_state_save_rate: int = 100



def train_a2c_jax(config: TrainingConfig = None, load_path: str = None, train: bool = False):
    """Main training function that matches your existing structure"""
    
    if config is None:
        config = TrainingConfig()
    
    print("Starting JAX A2C Training...")
    print(f"Num envs: {config.num_envs}")
    print(f"Sessions: {config.n_sessions}")
    print(f"Updates per session: {N_UPDATES_PER_SESSION}")
    print(f"Steps per update: {N_STEPS_PER_UPDATE}")
    
    # Initialize everything
    rng_key = random.PRNGKey(0)
    env_params = treadmill_session_default_params()
    
    # Create training state
    train_state = create_train_state(
        rng_key=rng_key,
        obs_size=config.obs_size,
        action_size=config.action_size,
        hidden_size=config.hidden_size,
        num_envs=config.num_envs,
        learning_rate=config.learning_rate,
        rnn_type=config.rnn_type,
        var_noise=config.var_noise,
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
    
    _, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)
    print(f"Initialized {config.num_envs} environments")
    
    # Storage for logging (matching your original structure)
    all_session_rewards = []
    
    # Training loop (outer loop stays in Python for logging)
    for session_num in trange(config.n_sessions, desc='Sessions'):
        
        avg_rewards_per_update = np.empty((config.num_envs, N_UPDATES_PER_SESSION))
        all_info = []
        
        for update_num in range(N_UPDATES_PER_SESSION):
            
            # JITted training step
            if train:
                train_state, env_states, metrics = train_step(
                    train_state=train_state,
                    env_states=env_states, 
                    env_params=env_params,
                    gamma=config.gamma,
                    critic_weight=config.critic_weight,
                    entropy_weight=config.entropy_weight,
                    input_noise_std=config.input_noise_std,
                    action_size=config.action_size,
                    hidden_size=config.hidden_size,
                    var_noise=config.var_noise,
                    rnn_type=config.rnn_type,
                )
                # Extract metrics for logging (numpy conversion happens here)
                avg_rewards_per_update[:, update_num] = np.array(metrics['mean_reward'])
            else:
                # if train is False, surface entire 'trajectory' object to be saved
                trajectory, train_state, env_states = collect_trajectory(
                    train_state, env_states, env_params, config.input_noise_std
                )
            
            # Could add more detailed info logging here if needed
            # all_info.append(some_info_from_metrics)
            
        # Session-level logging
        session_mean_reward = np.mean(avg_rewards_per_update)
        all_session_rewards.append(session_mean_reward)
        
        print(f'Session {session_num}: Avg reward = {session_mean_reward:.4f}')
        
        # Periodic saving (matching your original save rate)
        if session_num % config.output_state_save_rate == 0:
            print(f"Would save checkpoint at session {session_num}")
            save_dir = Path("checkpoints").resolve()  # makes it absolute
            checkpoints.save_checkpoint(
                ckpt_dir=str(save_dir),
                target={"params": train_state.params},  # or just train_state
                step=session_num,
                overwrite=False, # allows overwriting instead of accumulating
            )
    
    print("Training completed!")
    return train_state, all_session_rewards


def main():
    """Entry point for training"""
    
    # You can customize the config here
    config = TrainingConfig(
        n_sessions=2500,  # Shorter for testing
        num_envs=60,  # Smaller for testing
        learning_rate=1e-4,
        hidden_size=128,
        output_state_save_rate=20,
    )

    print(config)
    
    print("Starting JAX A2C training with test configuration...")
    
    final_train_state, rewards = train_a2c_jax(config)
    
    print("\nTraining Summary:")
    print(f"Final average reward: {np.mean(rewards[-10:]):.4f}")
    print(f"Best average reward: {np.max(rewards):.4f}")
        

if __name__ == "__main__":
    main()