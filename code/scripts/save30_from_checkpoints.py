"""Load already-trained checkpoints and save a 30-session (n_save_envs=30) FROZEN rollout each,
WITHOUT retraining.

Mirrors the snapshot-save block of train_treadmill_agent_jax_curriculum.py, but restores the params
from the existing checkpoint on disk instead of training. For a config whose exp_name ends in
"_save30", the source checkpoint is the same run WITHOUT that suffix:

    checkpoints/<exp_without_save30>_net<idx>/step_00/checkpoint_*        <- weights loaded here
    results/<exp_with_save30>_net<idx>/step_00/traj_000999.pkl           <- 30-env rollout saved here

Usage (mirrors the trainer CLI):
    python code/scripts/save30_from_checkpoints.py --config <save30_config.json> --n_networks 3
Optional --rollout_steps overrides the rollout length (default n_updates_per_session*n_steps_per_update).
"""
import sys
import os
from pathlib import Path

if __name__ == '__main__':
    sys.path.append(str(Path(__file__).parent.parent))   # code/

import argparse
import glob
import pickle

import jax
import jax.numpy as jnp
from jax import random
from flax import serialization
from flax.training import checkpoints

from agents.a2c_rnn_flax import init_network_and_params
from environments.components.train_state import create_train_state
from environments.components.treadmill_trajectory import collect_trajectory
from environments.treadmill_env_jax import TreadmillEnvironment
from train_treadmill_agent_jax_curriculum import load_curriculum_config, build_env_params

N_SAVE_ENVS = 30
SAVE_SUFFIX = '_save30'


def save_one(config, source_exp, output_exp, seed, rollout_steps=None):
    reset_fn, _, _ = TreadmillEnvironment()

    # network structure -> restore trained params
    _, params0 = init_network_and_params(
        hidden_size=config.hidden_size, action_size=config.action_size, obs_size=config.obs_size,
        rnn_type=config.rnn_type, unit_noise_std=config.unit_noise_std,
        rng_key=random.key(seed), init_scale=config.init_scale,
    )
    ckpt_dir = str(Path(f'checkpoints/{source_exp}/step_00').resolve())
    hits = sorted(glob.glob(os.path.join(ckpt_dir, 'checkpoint_*')))
    if not hits:
        raise FileNotFoundError(f'no checkpoint under {ckpt_dir} (has the run finished?)')
    restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target={'params': params0})
    params = restored['params']
    print(f'  loaded weights from {os.path.relpath(hits[-1])}')

    step = config.curriculum[0]
    env_params = build_env_params(step)

    # 30-env frozen rollout of one full session length, with the trained weights
    ts = create_train_state(
        rng_key=random.key(seed), obs_size=config.obs_size, hidden_size=config.hidden_size,
        num_envs=N_SAVE_ENVS, learning_rate=step.learning_rate, params=params,
    )
    key, reset_key = random.split(random.key(seed))
    reset_keys = random.split(reset_key, N_SAVE_ENVS)
    obs, env_states = jax.vmap(reset_fn, in_axes=(0, None))(reset_keys, env_params)
    ts = ts.replace(
        rng_key=key,
        actor_hidden=jnp.zeros((N_SAVE_ENVS, config.hidden_size)),
        critic_hidden=jnp.zeros((N_SAVE_ENVS, config.hidden_size)),
        prev_action=jnp.zeros((N_SAVE_ENVS,), dtype=jnp.int32),
        prev_reward=jnp.zeros((N_SAVE_ENVS,)),
        prev_obs=obs,
    )
    n_steps = rollout_steps or (config.n_updates_per_session * config.n_steps_per_update)
    trajectory, _, _ = collect_trajectory(
        train_state=ts, env_states=env_states, env_params=env_params,
        input_noise_std=config.input_noise_std, unit_noise_std=config.unit_noise_std,
        rnn_type=config.rnn_type, hidden_size=config.hidden_size, obs_size=config.obs_size,
        n_steps=n_steps,
    )
    traj_dicts = [
        serialization.to_state_dict(jax.tree_util.tree_map(lambda x: x[i], trajectory))
        for i in range(N_SAVE_ENVS)
    ]
    out_dir = Path(f'results/{output_exp}/step_00')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'traj_000999.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(traj_dicts, f)
    print(f'  saved {N_SAVE_ENVS} envs x {n_steps} steps -> {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='save30 curriculum config (exp_name should end in "_save30")')
    parser.add_argument('--n_networks', type=int, default=1)
    parser.add_argument('--rollout_steps', type=int, default=None,
                        help='override rollout length (default n_updates_per_session*n_steps_per_update)')
    args = parser.parse_args()

    config = load_curriculum_config(args.config)
    save_exp = config.exp_name
    source_base = save_exp[:-len(SAVE_SUFFIX)] if save_exp.endswith(SAVE_SUFFIX) else save_exp

    for idx in range(args.n_networks):
        seed = config.seed + idx
        if args.n_networks > 1:
            source_exp, output_exp = f'{source_base}_net{idx}', f'{save_exp}_net{idx}'
        else:
            source_exp, output_exp = source_base, save_exp
        print(f'>>> {output_exp} (from {source_exp}, seed {seed})')
        save_one(config, source_exp, output_exp, seed, rollout_steps=args.rollout_steps)


if __name__ == '__main__':
    main()
