import sys
import os
from pathlib import Path

if __name__ == '__main__':
    curr_file_path = Path(__file__)
    sys.path.append(str(curr_file_path.parent.parent))

import jax
import jax.numpy as jnp
if not hasattr(jnp, 'DeviceArray'):
    jnp.DeviceArray = jax.Array

from jax import random, lax
from flax import struct, serialization
import numpy as np
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import trange
import argparse
import pickle
import json
from datetime import datetime
from typing import Tuple, Dict
from functools import partial

from enum import IntEnum
from aux_funcs import zero_pad


class RewardParamStyle(IntEnum):
    FIXED = 0
    INDEP = 1
    COUPLED = 2


class RewardFuncType(IntEnum):
    EXP = 0
    BLOCK = 1
    MARKOV = 2


def reward_param_style_str_to_int(style):
    if isinstance(style, int):
        return style
    try:
        return RewardParamStyle[style.upper()].value
    except KeyError:
        raise ValueError(f"Unknown reward param style: {style}")


def reward_func_type_str_to_int(func_type):
    if isinstance(func_type, int):
        return func_type
    try:
        return RewardFuncType[func_type.upper()].value
    except KeyError:
        raise ValueError(f"Unknown reward func type: {func_type}")
from environments.treadmill_env_jax import (
    TreadmillEnvironment,
    TreadmillEnvParams,
    TreadmillEnvState,
    treadmill_session_default_params,
)


N_STEPS_PER_SESSION = 20000


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

@struct.dataclass
class SyntheticAgentState:
    prev_obs: jnp.ndarray                # (num_envs, obs_size)
    failure_count: jnp.ndarray           # (num_envs,)  int32
    odor_stop_count: jnp.ndarray         # (num_envs,)  int32
    in_odor_stop: jnp.ndarray            # (num_envs,)  bool
    got_reward_this_site: jnp.ndarray    # (num_envs,)  bool
    opt_out_flag: jnp.ndarray            # (num_envs,)  bool
    prev_in_odor: jnp.ndarray            # (num_envs,)  bool
    acc_rewards_in_patch: jnp.ndarray    # (num_envs,)  int32  cumulative rewards this patch
    rng_key: jnp.ndarray


# ---------------------------------------------------------------------------
# Trajectory data  (mirrors TrajectoryData, omitting network-specific fields)
# ---------------------------------------------------------------------------

@struct.dataclass
class SyntheticTrajectoryData:
    # agent-specific
    observations: jnp.ndarray               # (num_envs, n_steps, obs_size)
    actions: jnp.ndarray                    # (num_envs, n_steps)
    rewards: jnp.ndarray                    # (num_envs, n_steps)
    dones: jnp.ndarray                      # (num_envs, n_steps)
    failure_count: jnp.ndarray              # (num_envs, n_steps)  state before update
    opt_out_flag: jnp.ndarray               # (num_envs, n_steps)  state before update
    exp_filtered_reward_rate: jnp.ndarray   # (num_envs, n_steps)  cumulative mean reward rate
    # env infos
    action: jnp.ndarray
    current_patch_num: jnp.ndarray
    current_position: jnp.ndarray
    current_patch_start: jnp.ndarray
    agent_in_patch: jnp.ndarray
    reward_bounds: jnp.ndarray
    reward_site_idx: jnp.ndarray
    current_reward_site_attempted: jnp.ndarray
    patch_reward_param: jnp.ndarray
    patch_reward_prob_prefactor: jnp.ndarray
    reward: jnp.ndarray
    environment_quality: jnp.ndarray


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@struct.dataclass
class SyntheticAgentConfig:
    exp_name: str = ''
    num_envs: int = 64
    n_sessions: int = 30
    n_failures: int = 3
    dwell_steps: int = 3
    seed: int = 0

    # env params  (mirrors TrainingConfig)
    reward_param_style: str = 'fixed'
    reward_func_type: str = 'exp'
    reward_decay_consts: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([0.0, 10.0, 30.0]))
    reward_prob_prefactors: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([0.8, 0.8, 0.8]))
    reward_decay_range: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([0.0, 40.0]))
    reward_prob_range: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([0.0, 1.0]))
    patch_active_transition_prob_range: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([0.9, 0.9]))
    interreward_len_bounds: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([1.0, 6.0]))
    interreward_len_decay_rate: float = 0.8
    interpatch_len_bounds: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.array([1.0, 12.0]))
    interpatch_len_decay_rate: float = 0.1
    dwell_time_for_reward: int = 3
    reward_site_len: int = 3
    gr_offset: float = 0.05
    gr_scale: float = 0.0
    acc_reward_scale: float = 0.0


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=['n_failures', 'dwell_steps', 'n_steps'])
def collect_synthetic_trajectory(
    agent_state: SyntheticAgentState,
    env_states: TreadmillEnvState,
    env_params: TreadmillEnvParams,
    n_failures: int,
    dwell_steps: int,
    n_steps: int,
    gr_offset: float = 0.05,
    gr_scale: float = 0.0,
    acc_reward_scale: float = 0.0,
) -> Tuple[SyntheticTrajectoryData, SyntheticAgentState, TreadmillEnvState]:
    """Collect a trajectory with the synthetic count-failures agent via lax.scan."""

    _, step_fn, _ = TreadmillEnvironment()
    num_envs = env_states.current_position.shape[0]
    step_num = jnp.zeros(num_envs)

    def scan_step(carry, _):
        agent_state, env_states, step_num = carry
        rng_key = agent_state.rng_key

        obs = agent_state.prev_obs                           # (num_envs, obs_size)
        in_patch  = obs[..., 0] > 0.5                       # (num_envs,)
        in_odor   = jnp.any(obs[..., 1:4] > 0.5, axis=-1)  # (num_envs,)
        new_odor_onset = (~agent_state.prev_in_odor) & in_odor & in_patch

        # ---- action --------------------------------------------------------
        # Stop if: starting a new odor site  OR  still mid-stop (< dwell_steps)
        stopping = ~agent_state.opt_out_flag & (
            new_odor_onset |
            (agent_state.in_odor_stop & (agent_state.odor_stop_count < dwell_steps))
        )
        actions = jnp.where(stopping, 0, 1).astype(jnp.int32)

        # ---- step environment ----------------------------------------------
        rng_key, step_key = random.split(rng_key)
        step_keys = random.split(step_key, num_envs)
        new_obs, new_env_states, rewards, dones, infos = jax.vmap(
            lambda k, s, a: step_fn(k, s, a, env_params)
        )(step_keys, env_states, actions)

        # ---- cumulative mean reward rate -----------------------------------
        new_step_num = step_num + 1
        new_reward_rate = (new_env_states.exp_filtered_reward_rate
                           + (rewards - new_env_states.exp_filtered_reward_rate) / new_step_num)
        new_env_states = new_env_states.replace(exp_filtered_reward_rate=new_reward_rate)

        # ---- update agent state --------------------------------------------
        finishing_stop = agent_state.in_odor_stop & (agent_state.odor_stop_count >= dwell_steps)
        starting_stop  = new_odor_onset & ~agent_state.opt_out_flag

        # Odor-stop bookkeeping
        new_in_odor_stop = jnp.where(
            finishing_stop, False,
            jnp.where(starting_stop, True, agent_state.in_odor_stop)
        )
        new_odor_stop_count = jnp.where(
            finishing_stop, 0,
            jnp.where(starting_stop, 1,
            jnp.where(agent_state.in_odor_stop,
                      agent_state.odor_stop_count + 1,
                      agent_state.odor_stop_count))
        )

        # Reward tracking: accumulate any reward received while odor is on
        new_got_reward = jnp.where(
            finishing_stop | ~in_patch, False,
            jnp.where(in_odor & in_patch,
                      agent_state.got_reward_this_site | (rewards > 0),
                      agent_state.got_reward_this_site)
        )

        # Failure count: increment on unrewarded stop, reset on rewarded stop or patch exit
        final_got_reward = agent_state.got_reward_this_site | (rewards > 0)
        new_failure_count = jnp.where(
            ~in_patch, 0,
            jnp.where(
                finishing_stop,
                jnp.where(final_got_reward, 0, agent_state.failure_count + 1),
                agent_state.failure_count,
            )
        )

        # Accumulated rewards in patch: increment on reward, reset on patch exit.
        new_acc_rewards_in_patch = jnp.where(
            ~in_patch, 0,
            agent_state.acc_rewards_in_patch + (rewards > 0).astype(jnp.int32),
        )

        # Opt-out flag: threshold scales with global reward rate (MVT-like) and
        # increases with accumulated rewards collected in the current patch.
        effective_n_failures = jnp.maximum(
            n_failures
            + (gr_offset - new_env_states.exp_filtered_reward_rate) / gr_offset * gr_scale
            + acc_reward_scale * new_acc_rewards_in_patch,
            1.,
        )
        # jax.debug.print('Effective n_failures: {x}', x=effective_n_failures)
        new_opt_out_flag = jnp.where(
            ~in_patch, False,
            agent_state.opt_out_flag | (new_failure_count >= effective_n_failures)
        )

        new_agent_state = agent_state.replace(
            prev_obs=new_obs,
            failure_count=new_failure_count,
            odor_stop_count=new_odor_stop_count,
            in_odor_stop=new_in_odor_stop,
            got_reward_this_site=new_got_reward,
            opt_out_flag=new_opt_out_flag,
            prev_in_odor=in_odor & in_patch,
            acc_rewards_in_patch=new_acc_rewards_in_patch,
            rng_key=rng_key,
        )

        step_data = {
            'observations': obs,
            'actions': actions,
            'rewards': rewards,
            'dones': dones,
            'failure_count': agent_state.failure_count,   # log state before update
            'opt_out_flag': agent_state.opt_out_flag,
            'exp_filtered_reward_rate': new_env_states.exp_filtered_reward_rate,
        } | infos

        return (new_agent_state, new_env_states, new_step_num), step_data

    (final_agent_state, final_env_states, _), trajectory_data = lax.scan(
        scan_step,
        (agent_state, env_states, step_num),
        None,
        length=n_steps,
    )

    # Reshape (n_steps, num_envs, ...) → (num_envs, n_steps, ...)
    trajectory_data = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), trajectory_data)
    trajectory = SyntheticTrajectoryData(**trajectory_data)

    return trajectory, final_agent_state, final_env_states


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_synthetic_agent(config: SyntheticAgentConfig = None, save_trajectories: bool = True,
                        save_outputs: bool = True):
    """Run the synthetic agent.

    save_outputs : if False, nothing is written to disk (no checkpoints dir,
        per-session reward files, figures, or results/trajectory pickles); the
        results dict is still returned in memory. Useful for plotting in a
        notebook without consuming disk.
    """
    if config is None:
        config = SyntheticAgentConfig()

    print('Running synthetic count-failures foraging agent')
    print(f'  n_failures : {config.n_failures}')
    print(f'  dwell_steps: {config.dwell_steps}')
    print(f'  num_envs   : {config.num_envs}')
    print(f'  n_sessions : {config.n_sessions}')

    rng_key = random.key(config.seed)

    env_params = treadmill_session_default_params()
    env_params = env_params.replace(
        reward_param_style=reward_param_style_str_to_int(config.reward_param_style),
        reward_func_type=reward_func_type_str_to_int(config.reward_func_type),
        reward_decay_consts=config.reward_decay_consts,
        reward_prob_prefactors=config.reward_prob_prefactors,
        reward_decay_range=config.reward_decay_range,
        reward_prob_range=config.reward_prob_range,
        patch_active_transition_prob_range=config.patch_active_transition_prob_range,
        interreward_len_bounds=config.interreward_len_bounds,
        interreward_len_decay_rate=config.interreward_len_decay_rate,
        interpatch_len_bounds=config.interpatch_len_bounds,
        interpatch_len_decay_rate=config.interpatch_len_decay_rate,
        dwell_time_for_reward=config.dwell_time_for_reward,
    )

    reset_fn, _, _ = TreadmillEnvironment()

    if save_outputs:
        save_dir = Path(f'checkpoints/{config.exp_name}').resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir_rewards = save_dir / '_reward_rates'
        save_dir_rewards.mkdir(parents=True, exist_ok=True)
        results_dir = Path('results') / config.exp_name
        results_dir.mkdir(parents=True, exist_ok=True)

    all_session_rewards = []
    all_trajectories = [] if save_trajectories else None

    for session_num in trange(config.n_sessions, desc='Sessions'):
        rng_key, reset_key, agent_key = random.split(rng_key, 3)
        reset_keys = random.split(reset_key, config.num_envs)
        obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)

        in_patch_init = obs[..., 0] > 0.5
        in_odor_init  = jnp.any(obs[..., 1:4] > 0.5, axis=-1) & in_patch_init

        agent_state = SyntheticAgentState(
            prev_obs=obs,
            failure_count=jnp.zeros(config.num_envs, dtype=jnp.int32),
            odor_stop_count=jnp.zeros(config.num_envs, dtype=jnp.int32),
            in_odor_stop=jnp.zeros(config.num_envs, dtype=bool),
            got_reward_this_site=jnp.zeros(config.num_envs, dtype=bool),
            opt_out_flag=jnp.zeros(config.num_envs, dtype=bool),
            prev_in_odor=in_odor_init,
            acc_rewards_in_patch=jnp.zeros(config.num_envs, dtype=jnp.int32),
            rng_key=agent_key,
        )

        trajectory, final_agent_state, final_env_states = collect_synthetic_trajectory(
            agent_state=agent_state,
            env_states=env_states,
            env_params=env_params,
            n_failures=config.n_failures,
            dwell_steps=config.dwell_steps,
            n_steps=N_STEPS_PER_SESSION,
            gr_offset=config.gr_offset,
            gr_scale=config.gr_scale,
            acc_reward_scale=config.acc_reward_scale,
        )

        session_mean_reward = float(jnp.mean(trajectory.rewards))
        all_session_rewards.append(session_mean_reward)
        print(f'Session {session_num}: mean reward = {session_mean_reward:.4f}')

        if session_num == 0 and save_outputs:
            T = 200
            obs   = np.array(trajectory.observations[0, :T])   # (T, obs_size)
            acts  = np.array(trajectory.actions[0, :T])        # (T,) 0=stop 1=move
            lacts = np.array(trajectory.action[0, :T])         # (T,) actual movement
            rews  = np.array(trajectory.rewards[0, :T])        # (T,)

            fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
            t = np.arange(T)

            for i in range(obs.shape[1]):
                axes[0].plot(t, obs[:, i], label=f'obs[{i}]')
            axes[0].set_ylabel('Observations')
            axes[0].legend(fontsize=7, ncol=obs.shape[1])

            axes[1].step(t, acts, where='post')
            axes[1].set_ylabel('Action\n(0=stop, 1=move)')
            axes[1].set_yticks([0, 1])

            axes[2].step(t, lacts, where='post')
            axes[2].set_ylabel('Movement\n(last action)')

            axes[3].vlines(t[rews > 0], 0, rews[rews > 0], colors='tab:green')
            axes[3].set_ylabel('Reward')
            axes[3].set_xlabel('Step')

            fig.suptitle(f'{config.exp_name} — first 200 steps (env 0)')
            fig.tight_layout()
            fig.savefig(results_dir / 'first_200_steps.png', dpi=100)
            plt.close(fig)

        if save_trajectories:
            all_trajectories.append(serialization.to_state_dict(trajectory))

        if save_outputs:
            sn = zero_pad(session_num, 6)
            with open(save_dir_rewards / f'{sn}', 'ab') as f:
                np.save(f, np.array(trajectory.rewards))

    # Reward curve
    if save_outputs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(all_session_rewards)
        ax.set_xlabel('Session')
        ax.set_ylabel('Mean reward rate')
        ax.set_title(config.exp_name)
        fig.tight_layout()
        fig.savefig(results_dir / 'reward_rate.png', dpi=100)
        plt.close(fig)

    results = {
        'session_rewards': all_session_rewards,
        'mean_reward_rate': float(np.mean(all_session_rewards)),
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }

    if save_outputs and save_trajectories and all_trajectories:
        traj_file = results_dir / f'trajectories_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        with open(traj_file, 'wb') as f:
            pickle.dump(all_trajectories, f)
        print(f'Trajectories saved to {traj_file}')

    if save_outputs:
        results_file = results_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

    print(f'\nDone. Mean reward rate: {results["mean_reward_rate"]:.4f}')
    return results, all_trajectories


# ---------------------------------------------------------------------------
# Config helpers  (mirrors train_treadmill_agent_jax)
# ---------------------------------------------------------------------------

def load_config_from_json(filepath: str) -> SyntheticAgentConfig:
    with open(filepath) as f:
        d = json.load(f)
    for key in ('reward_decay_consts', 'reward_prob_prefactors',
                'reward_decay_range', 'reward_prob_range',
                'patch_active_transition_prob_range',
                'interreward_len_bounds', 'interpatch_len_bounds'):
        if key in d:
            d[key] = jnp.array(d[key])
    if 'reward_param_style' in d:
        d['reward_param_style'] = reward_param_style_str_to_int(d['reward_param_style'])
    if 'reward_func_type' in d:
        d['reward_func_type'] = reward_func_type_str_to_int(d['reward_func_type'])
    config = SyntheticAgentConfig()
    return config.replace(**{k: v for k, v in d.items() if hasattr(config, k)})


def save_config_to_json(config: SyntheticAgentConfig, filepath: str) -> None:
    d = serialization.to_state_dict(config)
    with open(filepath, 'w') as f:
        json.dump(d, f, indent=2)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run synthetic count-failures foraging agent')
    parser.add_argument('--config',            type=str,   default=None)
    parser.add_argument('--exp_title',         type=str,   default='synthetic_agent')
    parser.add_argument('--n_failures',        type=int,   default=3)
    parser.add_argument('--dwell_steps',       type=int,   default=3)
    parser.add_argument('--num_envs',          type=int,   default=64)
    parser.add_argument('--n_sessions',        type=int,   default=30)
    parser.add_argument('--seed',              type=int,   default=0)
    parser.add_argument('--save_trajectories', action='store_true')
    args = parser.parse_args()

    time_stamp = str(datetime.now()).replace(' ', '_')

    if args.config:
        config = load_config_from_json(args.config)
        if not config.exp_name:
            config = config.replace(exp_name=f'synthetic_{time_stamp}')
    else:
        exp_name = f'{args.exp_title}_nfail{args.n_failures}_{time_stamp}'
        config = SyntheticAgentConfig(
            exp_name=exp_name,
            n_failures=args.n_failures,
            dwell_steps=args.dwell_steps,
            num_envs=args.num_envs,
            n_sessions=args.n_sessions,
            seed=args.seed,
        )

    print(config)
    run_synthetic_agent(config, save_trajectories=args.save_trajectories)


if __name__ == '__main__':
    main()
