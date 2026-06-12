import sys
from pathlib import Path
import dataclasses
from dataclasses import dataclass, field
from typing import List, Optional

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import jax
import jax.numpy as jnp

if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = jax.Array

from jax import random
from flax import serialization
from flax.training import checkpoints
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import trange
import argparse
import pickle
import json

from aux_funcs import zero_pad
from agents.a2c_rnn_flax import init_network_and_params
from environments.components.train_state import create_train_state
from environments.components.treadmill_trajectory import collect_trajectory
from environments.treadmill_env_jax import (
    TreadmillEnvironment,
    treadmill_session_default_params,
)
from train_treadmill_agent_jax import (
    N_UPDATES_PER_SESSION,
    N_STEPS_PER_UPDATE,
    run_session_updates_with_metrics,
    reward_param_style_str_to_int,
    reward_func_type_str_to_int,
)


@dataclass
class CurriculumStep:
    n_sessions: int
    output_save_start: int
    output_save_step: int
    output_save_end: Optional[int] = None
    dwell_time_for_reward: int = 3
    reward_site_len: int = 3
    fixed_patches: List = field(default_factory=lambda: [0, 0, 0])
    reward_param_style: str = 'fixed'
    reward_func_type: str = 'exp'
    reward_decay_consts: List = field(default_factory=lambda: [0.0, 10.0, 30.0])
    reward_prob_prefactors: List = field(default_factory=lambda: [0.8, 0.8, 0.8])
    reward_decay_range: List = field(default_factory=lambda: [0.0, 40.0])
    reward_prob_range: List = field(default_factory=lambda: [0.05, 0.95])
    interreward_len_bounds: List = field(default_factory=lambda: [1.0, 6.0])
    interreward_len_decay_rate: float = 0.8
    interpatch_len_bounds: List = field(default_factory=lambda: [1.0, 12.0])
    interpatch_len_decay_rate: float = 0.1
    global_reward_weight: float = 1.0
    learning_rate: float = 0.0001


@dataclass
class CurriculumConfig:
    exp_name: str = ''
    num_envs: int = 128
    patch_types_per_env: int = 3
    obs_size: int = 4
    action_size: int = 2
    hidden_size: int = 64
    critic_weight: float = 0.05
    entropy_weight: float = 0.0025
    env_prediction_weight: float = 0.0
    activity_norm_weight: float = 0.001
    pred_obs_weight: float = 0.0
    gamma: float = 0.999
    input_noise_std: float = 0.01
    unit_noise_std: float = 0.01
    rnn_type: str = 'GRU'
    init_scale: float = 1.0
    seed: int = 0
    curriculum: List[CurriculumStep] = field(default_factory=list)


def load_curriculum_config(filepath: str) -> CurriculumConfig:
    with open(filepath, 'r') as f:
        d = json.load(f)
    steps = [CurriculumStep(**s) for s in d.pop('curriculum')]
    return CurriculumConfig(curriculum=steps, **d)


def build_env_params(step: CurriculumStep):
    base = treadmill_session_default_params()
    return base.replace(
        reward_param_style=reward_param_style_str_to_int(step.reward_param_style),
        reward_func_type=reward_func_type_str_to_int(step.reward_func_type),
        fixed_patches=jnp.array(step.fixed_patches),
        reward_decay_consts=jnp.array(step.reward_decay_consts),
        reward_prob_prefactors=jnp.array(step.reward_prob_prefactors),
        reward_decay_range=jnp.array(step.reward_decay_range),
        reward_prob_range=jnp.array(step.reward_prob_range),
        interreward_len_bounds=jnp.array(step.interreward_len_bounds),
        interreward_len_decay_rate=step.interreward_len_decay_rate,
        interpatch_len_bounds=jnp.array(step.interpatch_len_bounds),
        interpatch_len_decay_rate=step.interpatch_len_decay_rate,
        dwell_time_for_reward=step.dwell_time_for_reward,
        reward_site_len=step.reward_site_len,
    )


def should_save(session_in_step: int, n_sessions: int, save_start: int,
                save_step: int, save_end: Optional[int]) -> bool:
    if save_start == -1:
        return session_in_step == n_sessions - 1
    end = save_end if save_end is not None else n_sessions
    return (save_start <= session_in_step < end and
            (session_in_step - save_start) % save_step == 0)


def train_curriculum(config: CurriculumConfig):
    total_sessions = sum(s.n_sessions for s in config.curriculum)
    print(f"Starting curriculum training: {config.exp_name}")
    print(f"  {len(config.curriculum)} steps, {total_sessions} total sessions")
    print(f"  Num envs: {config.num_envs}, hidden: {config.hidden_size}")
    print(f"  Updates/session: {N_UPDATES_PER_SESSION}, steps/update: {N_STEPS_PER_UPDATE}")

    rng_key = random.key(config.seed)
    net_init_key, rng_key = random.split(rng_key)

    _, params = init_network_and_params(
        hidden_size=config.hidden_size,
        action_size=config.action_size,
        obs_size=config.obs_size,
        rnn_type=config.rnn_type,
        unit_noise_std=config.unit_noise_std,
        rng_key=net_init_key,
        init_scale=config.init_scale,
    )

    train_state = create_train_state(
        rng_key=rng_key,
        obs_size=config.obs_size,
        hidden_size=config.hidden_size,
        num_envs=config.num_envs,
        learning_rate=config.curriculum[0].learning_rate,
        params=params,
    )

    save_dir = Path(f"checkpoints/{config.exp_name}").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    rewards_dir = save_dir / '_reward_rates'
    rewards_dir.mkdir(parents=True, exist_ok=True)

    results_dir = Path('results') / config.exp_name
    results_dir.mkdir(parents=True, exist_ok=True)

    reset_fn, _, _ = TreadmillEnvironment()

    all_session_rewards = []
    session_abs = 0

    for step_idx, step in enumerate(config.curriculum):
        print(f"\n{'='*60}")
        print(f"Curriculum step {step_idx}: {step.n_sessions} sessions")
        print(f"  reward_param_style={step.reward_param_style}, "
              f"reward_func_type={step.reward_func_type}")
        print(f"  save_start={step.output_save_start}, "
              f"save_step={step.output_save_step}, "
              f"save_end={step.output_save_end}")

        env_params = build_env_params(step)
        train_state = train_state.replace(learning_rate=step.learning_rate)

        step_traj_dir = results_dir / f'step_{step_idx:02d}'
        step_traj_dir.mkdir(parents=True, exist_ok=True)

        step_weights_dir = save_dir / f'step_{step_idx:02d}'
        step_weights_dir.mkdir(parents=True, exist_ok=True)

        step_rewards_dir = rewards_dir / f'step_{step_idx:02d}'
        step_rewards_dir.mkdir(parents=True, exist_ok=True)

        for session_in_step in trange(step.n_sessions, desc=f'Step {step_idx}'):

            rng_key, reset_key = random.split(train_state.rng_key)
            reset_keys = random.split(reset_key, config.num_envs)
            obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)

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
                critic_weight=config.critic_weight,
                entropy_weight=config.entropy_weight,
                env_prediction_weight=config.env_prediction_weight,
                global_reward_weight=step.global_reward_weight,
                activity_norm_weight=config.activity_norm_weight,
                pred_obs_weight=config.pred_obs_weight,
                input_noise_std=config.input_noise_std,
                action_size=config.action_size,
                hidden_size=config.hidden_size,
                unit_noise_std=config.unit_noise_std,
                rnn_type=config.rnn_type,
                obs_size=config.obs_size,
            )

            avg_rewards = all_metrics['mean_reward']
            grad_norms = all_metrics['grad_norm']
            session_mean_reward = float(jnp.mean(avg_rewards))
            all_session_rewards.append(session_mean_reward)

            print(f'Step {step_idx} sess {session_in_step} (abs {session_abs}): '
                  f'reward={session_mean_reward:.4f}, '
                  f'grad_norm={float(jnp.mean(grad_norms)):.4f}')

            sn = zero_pad(session_in_step, 6)
            with open(step_rewards_dir / sn, 'ab') as f:
                np.save(f, np.array(avg_rewards))

            if should_save(session_in_step, step.n_sessions, step.output_save_start,
                           step.output_save_step, step.output_save_end):
                rng_key, save_reset_key = random.split(train_state.rng_key)
                train_state = train_state.replace(rng_key=rng_key)
                save_obs, save_env_states = jax.vmap(reset_fn, in_axes=(0, None))(
                    save_reset_key[None], env_params
                )
                save_train_state = train_state.replace(
                    actor_hidden=jnp.zeros((1, config.hidden_size)),
                    critic_hidden=jnp.zeros((1, config.hidden_size)),
                    prev_action=jnp.zeros((1,), dtype=jnp.int32),
                    prev_reward=jnp.zeros((1,)),
                    prev_obs=save_obs,
                )
                trajectory, _, _ = collect_trajectory(
                    train_state=save_train_state,
                    env_states=save_env_states,
                    env_params=env_params,
                    input_noise_std=config.input_noise_std,
                    unit_noise_std=config.unit_noise_std,
                    rnn_type=config.rnn_type,
                    hidden_size=config.hidden_size,
                    obs_size=config.obs_size,
                    n_steps=N_UPDATES_PER_SESSION * N_STEPS_PER_UPDATE,
                )
                traj_no_batch = jax.tree_util.tree_map(lambda x: x[0], trajectory)
                traj_dict = serialization.to_state_dict(traj_no_batch)
                traj_path = step_traj_dir / f'traj_{session_abs:06d}.pkl'
                with open(traj_path, 'wb') as f:
                    pickle.dump([traj_dict], f)
                print(f'  -> Saved trajectory to {traj_path}')

                checkpoints.save_checkpoint(
                    ckpt_dir=str(step_weights_dir),
                    target={'params': train_state.params},
                    step=session_abs,
                    overwrite=False,
                    keep=float('inf'),
                )
                print(f'  -> Saved weights to {step_weights_dir}/checkpoint_{session_abs}')

            session_abs += 1

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(all_session_rewards)
        for k in range(step_idx + 1):
            boundary = sum(s.n_sessions for s in config.curriculum[:k])
            ax.axvline(boundary, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Session (global)')
        ax.set_ylabel('Mean reward rate')
        ax.set_title(f'{config.exp_name} — after step {step_idx}')
        fig.tight_layout()
        fig.savefig(results_dir / 'reward_rate.png', dpi=100)
        plt.close(fig)

    print("\nCurriculum training complete!")
    return train_state, all_session_rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to curriculum JSON config file')
    parser.add_argument('--n_networks', type=int, default=1,
                        help='Number of networks to train sequentially (default: 1)')
    args = parser.parse_args()

    config = load_curriculum_config(args.config)

    for network_idx in range(args.n_networks):
        network_seed = config.seed + network_idx
        network_exp_name = (
            f"{config.exp_name}_net{network_idx}" if args.n_networks > 1
            else config.exp_name
        )
        network_config = dataclasses.replace(
            config, seed=network_seed, exp_name=network_exp_name
        )
        train_curriculum(network_config)


if __name__ == '__main__':
    main()
