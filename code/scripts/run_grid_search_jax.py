import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from functools import partial

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import jax
import jax.numpy as jnp

if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = jax.Array

from jax import random, lax
from flax import serialization
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import trange
import argparse
import pickle
import json

from aux_funcs import zero_pad
from environments.treadmill_env_jax import (
    TreadmillEnvironment,
    TreadmillEnvParams,
    treadmill_session_default_params,
)
from agents.grid_search_agent_jax import (
    GridSearchAgentState,
    init_agent_state,
    grid_search_step,
    advance_policy,
)
from train_treadmill_agent_jax import (
    reward_param_style_str_to_int,
    reward_func_type_str_to_int,
)


# Steps per session: matches N_UPDATES_PER_SESSION * N_STEPS_PER_UPDATE from the RNN trainer
N_STEPS_PER_SESSION = 20000


@dataclass
class GridSearchConfig:
    exp_name: str = ''
    num_envs: int = 128
    patch_types_per_env: int = 3
    obs_size: int = 4
    seed: int = 0

    # Grid search params
    n_steps_per_session: int = N_STEPS_PER_SESSION
    n_sessions_per_policy: int = 1
    n_eval_sessions: int = 30
    strategy: str = 'reward_count'
    wait_time_for_reward: int = 3
    stop_ranges_per_patch: List = field(default_factory=lambda: [[0, 8], [0, 8], [0, 8]])

    # Observation structure (fixed by env; patch cue at 0, odors at 1-3)
    patch_cue_idx: int = 0
    odor_cues_start: int = 1
    odor_cues_end: int = 4

    # Environment params
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


def load_config(filepath: str) -> GridSearchConfig:
    with open(filepath, 'r') as f:
        d = json.load(f)
    return GridSearchConfig(**d)


def build_env_params(config: GridSearchConfig) -> TreadmillEnvParams:
    base = treadmill_session_default_params()
    return base.replace(
        reward_param_style=reward_param_style_str_to_int(config.reward_param_style),
        reward_func_type=reward_func_type_str_to_int(config.reward_func_type),
        fixed_patches=jnp.array(config.fixed_patches),
        reward_decay_consts=jnp.array(config.reward_decay_consts),
        reward_prob_prefactors=jnp.array(config.reward_prob_prefactors),
        reward_decay_range=jnp.array(config.reward_decay_range),
        reward_prob_range=jnp.array(config.reward_prob_range),
        interreward_len_bounds=jnp.array(config.interreward_len_bounds),
        interreward_len_decay_rate=config.interreward_len_decay_rate,
        interpatch_len_bounds=jnp.array(config.interpatch_len_bounds),
        interpatch_len_decay_rate=config.interpatch_len_decay_rate,
        dwell_time_for_reward=config.dwell_time_for_reward,
        reward_site_len=config.reward_site_len,
    )


@partial(jax.jit, static_argnames=['strategy', 'odor_cues_start', 'odor_cues_end',
                                    'patch_cue_idx', 'wait_time_for_reward', 'n_steps'])
def run_grid_search_session(
    agent_state: GridSearchAgentState,
    env_states,
    initial_obs: jnp.ndarray,       # (num_envs, obs_size)
    env_params: TreadmillEnvParams,
    rng_key: jnp.ndarray,
    n_stops_for_patch: jnp.ndarray,  # (n_patches,) int32
    strategy: str,
    odor_cues_start: int,
    odor_cues_end: int,
    patch_cue_idx: int,
    wait_time_for_reward: int,
    n_steps: int,
):
    """Run one session via lax.scan. Carries current_obs so the agent sees
    the obs produced by each env step before acting on it next step."""
    _, step_fn, _ = TreadmillEnvironment()
    num_envs = initial_obs.shape[0]

    def scan_step(carry, _):
        agent_state, env_states, rng_key, current_obs = carry

        action, new_agent_state = grid_search_step(
            agent_state, current_obs, n_stops_for_patch,
            strategy, odor_cues_start, odor_cues_end, patch_cue_idx, wait_time_for_reward,
        )

        rng_key, step_key = random.split(rng_key)
        step_keys = random.split(step_key, num_envs)

        new_obs, new_env_states, rewards, _, _ = jax.vmap(
            lambda k, s, a: step_fn(k, s, a, env_params)
        )(step_keys, env_states, action)

        new_agent_state = new_agent_state.replace(
            rewards_in_patch=(new_agent_state.rewards_in_patch + rewards).astype(jnp.int32),
        )

        return (new_agent_state, new_env_states, rng_key, new_obs), rewards

    (final_agent_state, final_env_states, final_rng, _), all_rewards = lax.scan(
        scan_step,
        (agent_state, env_states, rng_key, initial_obs),
        None,
        length=n_steps,
    )

    return final_agent_state, final_env_states, final_rng, all_rewards  # all_rewards: (n_steps, num_envs)


def run_grid_search(config: GridSearchConfig):
    stop_ranges = np.array(config.stop_ranges_per_patch)
    n_patches = len(stop_ranges)
    n_policies = int(np.prod(stop_ranges[:, 1] - stop_ranges[:, 0] + 1))

    print(f"Starting grid search: {config.exp_name}")
    print(f"  {n_patches} patch types, {n_policies} total policies")
    print(f"  {config.n_sessions_per_policy} session(s) per policy, "
          f"{config.n_steps_per_session} steps each")
    print(f"  Strategy: {config.strategy}")

    rng_key = random.key(config.seed)
    env_params = build_env_params(config)
    reset_fn, _, _ = TreadmillEnvironment()

    save_dir = Path(f"checkpoints/{config.exp_name}").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(f"results/{config.exp_name}").resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    reward_rates_for_policies = np.zeros((config.num_envs, n_policies))
    n_stops_for_policies = np.zeros((n_patches, n_policies), dtype=int)
    n_stops_for_patch = stop_ranges[:, 0].copy()

    all_mean_rewards = []

    for policy_idx in trange(n_policies, desc='Grid search'):
        n_stops_for_policies[:, policy_idx] = n_stops_for_patch.copy()
        policy_rewards = np.zeros((config.num_envs,))

        for _ in range(config.n_sessions_per_policy):
            rng_key, reset_key = random.split(rng_key)
            reset_keys = random.split(reset_key, config.num_envs)
            initial_obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(
                reset_keys, env_params
            )

            agent_state = init_agent_state(config.num_envs, config.obs_size)

            rng_key, session_key = random.split(rng_key)
            _, _, _, all_rewards = run_grid_search_session(
                agent_state=agent_state,
                env_states=env_states,
                initial_obs=initial_obs,
                env_params=env_params,
                rng_key=session_key,
                n_stops_for_patch=jnp.array(n_stops_for_patch, dtype=jnp.int32),
                strategy=config.strategy,
                odor_cues_start=config.odor_cues_start,
                odor_cues_end=config.odor_cues_end,
                patch_cue_idx=config.patch_cue_idx,
                wait_time_for_reward=config.wait_time_for_reward,
                n_steps=config.n_steps_per_session,
            )
            policy_rewards += np.array(all_rewards.mean(axis=0))

        policy_rewards /= config.n_sessions_per_policy
        reward_rates_for_policies[:, policy_idx] = policy_rewards
        mean_r = float(policy_rewards.mean())
        all_mean_rewards.append(mean_r)

        print(f'Policy {policy_idx}: stops={n_stops_for_patch}, '
              f'mean_reward={mean_r:.4f}')

        n_stops_for_patch, search_finished = advance_policy(n_stops_for_patch, stop_ranges)
        if search_finished:
            break

    # Optimal policy per env
    optimal_policy_idx = reward_rates_for_policies.argmax(axis=1)  # (num_envs,)
    optimal_n_stops = n_stops_for_policies[:, optimal_policy_idx]   # (n_patches, num_envs)

    print("\nOptimal policies per env:")
    for k in range(config.num_envs):
        print(f"  Env {k:3d}: stops={n_stops_for_policies[:, optimal_policy_idx[k]]}, "
              f"reward={reward_rates_for_policies[k, optimal_policy_idx[k]]:.4f}")

    # Plot reward rates across grid
    fig, ax = plt.subplots(figsize=(max(8, n_policies // 10), 4))
    ax.plot(all_mean_rewards)
    ax.set_xlabel('Policy index')
    ax.set_ylabel('Mean reward rate')
    ax.set_title(config.exp_name)
    fig.tight_layout()
    fig.savefig(results_dir / 'grid_search_rewards.png', dpi=100)
    plt.close(fig)

    # Save grid search results
    search_results = {
        'reward_rates_for_policies': reward_rates_for_policies,
        'n_stops_for_policies': n_stops_for_policies,
        'optimal_policy_idx': optimal_policy_idx,
        'optimal_n_stops': optimal_n_stops,
    }
    with open(results_dir / 'grid_search_results.pkl', 'wb') as f:
        pickle.dump(search_results, f)
    print(f"\nSearch results saved to {results_dir / 'grid_search_results.pkl'}")

    # Evaluation: use mode of optimal stops across envs as a single shared policy
    modal_stops = np.array([
        np.bincount(optimal_n_stops[i]).argmax() for i in range(n_patches)
    ])
    print(f"\nEvaluating modal optimal policy: stops={modal_stops}")

    eval_trajectories = []
    for episode in trange(config.n_eval_sessions, desc='Evaluation'):
        rng_key, reset_key = random.split(rng_key)
        reset_keys = random.split(reset_key, config.num_envs)
        initial_obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(
            reset_keys, env_params
        )

        agent_state = init_agent_state(config.num_envs, config.obs_size)

        rng_key, session_key = random.split(rng_key)
        _, _, _, all_rewards = run_grid_search_session(
            agent_state=agent_state,
            env_states=env_states,
            initial_obs=initial_obs,
            env_params=env_params,
            rng_key=session_key,
            n_stops_for_patch=jnp.array(modal_stops, dtype=jnp.int32),
            strategy=config.strategy,
            odor_cues_start=config.odor_cues_start,
            odor_cues_end=config.odor_cues_end,
            patch_cue_idx=config.patch_cue_idx,
            wait_time_for_reward=config.wait_time_for_reward,
            n_steps=config.n_steps_per_session,
        )

        episode_reward = float(jnp.mean(all_rewards))
        eval_trajectories.append({'mean_reward': episode_reward})

    mean_eval_reward = np.mean([t['mean_reward'] for t in eval_trajectories])
    print(f"Eval mean reward rate: {mean_eval_reward:.4f}")

    eval_results = {
        'eval_trajectories': eval_trajectories,
        'mean_eval_reward': mean_eval_reward,
        'modal_stops': modal_stops,
        'search_results': search_results,
    }
    with open(results_dir / 'eval_results.pkl', 'wb') as f:
        pickle.dump(eval_results, f)
    print(f"Eval results saved to {results_dir / 'eval_results.pkl'}")

    return eval_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file')
    args = parser.parse_args()

    config = load_config(args.config)
    run_grid_search(config)


if __name__ == '__main__':
    main()
