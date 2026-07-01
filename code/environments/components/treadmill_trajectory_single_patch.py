"""Trajectory collector for the single never-ending-patch environment.

Copy of treadmill_trajectory.collect_trajectory that drives the single-patch
env (treadmill_env_single_patch_jax) instead of the full foraging env, so the
same A2C update math can be reused for the single-patch curriculum stage.
"""

import jax
import jax.numpy as jnp
from jax import random, lax
from typing import Tuple
from functools import partial

from environments.treadmill_env_jax import TreadmillEnvParams, TreadmillEnvState
from environments.treadmill_env_single_patch_jax import TreadmillSinglePatchEnvironment
from environments.components.train_state import TrainState
from environments.components.treadmill_trajectory import TrajectoryData
from agents.a2c_rnn_flax import A2CRNNFlax


@partial(jax.jit, static_argnames=['rnn_type', 'hidden_size', 'n_steps', 'obs_size', 'patch_type'])
def collect_trajectory_single_patch(
    train_state: TrainState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    input_noise_std: float,
    unit_noise_std: float,
    rnn_type: str,
    hidden_size: int,
    obs_size: int,
    n_steps: int,
    patch_type: int = 0,
) -> Tuple[TrajectoryData, TrainState, TreadmillEnvState]:
    """Collect a trajectory in the single never-ending patch via lax.scan."""

    network = A2CRNNFlax(
        action_size=2,
        hidden_size=hidden_size,
        unit_noise_std=unit_noise_std,
        rnn_type=rnn_type,
        obs_size=obs_size,
    )

    reset_fn, step_fn, get_obs_fn = TreadmillSinglePatchEnvironment(patch_type)

    step_num = jnp.zeros_like(env_states.exp_filtered_reward_rate)

    def scan_step(carry, _):
        train_state, env_states, step_num = carry
        rng_key = train_state.rng_key

        prev_action_one_hot = jax.nn.one_hot(train_state.prev_action, num_classes=2)
        network_input = jnp.concatenate([
            train_state.prev_obs,
            prev_action_one_hot,
            train_state.prev_reward[..., None],
        ], axis=-1)

        rng_key, noise_key = random.split(rng_key)
        obs_noise = random.normal(noise_key, network_input.shape) * input_noise_std
        network_input = network_input + obs_noise

        rng_key, network_noise_key = random.split(rng_key)

        logits, values, new_actor_hidden, new_critic_hidden, pred_env_quality, pred_obs, pred_reward_rate = network.apply(
            train_state.params,
            jax.lax.stop_gradient(network_input),
            train_state.actor_hidden,
            train_state.critic_hidden,
            rngs={'noise': network_noise_key} if train_state.params else {}
        )

        rng_key, action_key = random.split(rng_key)
        action_keys = random.split(action_key, logits.shape[0])
        actions = jax.vmap(
            lambda key, logit: random.categorical(key, logit)
        )(action_keys, logits)

        rng_key, step_key = random.split(rng_key)
        step_keys = random.split(step_key, actions.shape[0])
        step_results = jax.vmap(
            lambda key, state, action: step_fn(key, state, action, env_params)
        )(step_keys, env_states, actions)

        new_obs, new_env_states, rewards, dones, infos = step_results

        new_step_num = step_num + 1
        new_reward_rate = (new_env_states.exp_filtered_reward_rate
                           + (rewards - new_env_states.exp_filtered_reward_rate) / new_step_num)
        new_env_states = new_env_states.replace(
            exp_filtered_reward_rate=new_reward_rate,
        )

        new_train_state = train_state.replace(
            rng_key=rng_key,
            actor_hidden=new_actor_hidden,
            critic_hidden=new_critic_hidden,
            prev_obs=new_obs,
            prev_action=actions,
            prev_reward=rewards,
        )

        step_data = {
            'observations': network_input,
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

        return (new_train_state, new_env_states, new_step_num), step_data

    (final_train_state, final_env_states, _), trajectory_data = lax.scan(
        scan_step,
        (train_state, env_states, step_num),
        None,
        length=n_steps,
    )

    trajectory_data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory_data)
    trajectory = TrajectoryData(**trajectory_data)
    return trajectory, final_train_state, final_env_states
