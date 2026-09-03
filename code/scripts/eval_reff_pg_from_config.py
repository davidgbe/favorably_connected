"""Evaluate a trained reff_pg network from a config JSON that carries its own checkpoint.

The JSON is a normal training config plus ONE extra key:
    "checkpoint_path": "checkpoints/<exp_name>/snapshots"   # a dir -> the LATEST checkpoint in it is
                                                            #   loaded (flax restore_checkpoint), or a
                                                            #   specific checkpoint_<step> path.
Everything else is passed straight through to TrainingConfig. Evaluation forces a single-env,
noise-free rollout of --n_episodes episodes and (optionally) saves trajectories to
results/<exp_name>/ (where reff_behavioral_traces.ipynb reads them).

Usage:
    python3 code/scripts/eval_reff_pg_from_config.py \
        training_configs/reff_pg_exp_gru_initial_prob_offset_no_belief_clean_ms0p5_eval.json \
        --n_episodes 30 --save_trajectories
"""
import sys
import os
import json
import argparse
from pathlib import Path

# make `scripts` / `agents` / `environments` importable (same as the training script)
sys.path.append(str(Path(__file__).resolve().parent.parent))

import jax.numpy as jnp
from scripts.train_treadmill_agent_jax_reff_pg import TrainingConfig, evaluate_a2c_jax

# config fields that must become jnp arrays (mirrors load_config_from_json)
_ARRAY_FIELDS = [
    'reward_decay_consts', 'reward_prob_prefactors', 'reward_decay_range', 'reward_prob_range',
    'patch_active_transition_prob_range', 'interreward_len_bounds', 'interpatch_len_bounds', 'fixed_patches',
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('config', help='config JSON (a training config + a "checkpoint_path" key)')
    ap.add_argument('--n_episodes', type=int, default=30, help='number of evaluation episodes (default: 30)')
    ap.add_argument('--save_trajectories', action='store_true', help='save trajectories to results/<exp_name>/')
    args = ap.parse_args()

    with open(args.config) as f:
        raw = json.load(f)

    checkpoint_path = raw.pop('checkpoint_path', None)
    if not checkpoint_path:
        sys.exit('config JSON must contain a "checkpoint_path" key (a snapshots dir or a checkpoint_<step> path)')
    raw.pop('test_sessions', None)  # tolerate an eval-only key if present

    for k in _ARRAY_FIELDS:
        if k in raw:
            raw[k] = jnp.array(raw[k])

    config = TrainingConfig().replace(**raw)
    # eval overrides: deterministic SINGLE-env rollout of n_episodes episodes. num_envs MUST be 1 --
    # evaluate_a2c_jax builds the train state with a single env, so a config num_envs>1 (from the
    # training JSON) makes prev_obs (num_envs,4) mismatch prev_action/prev_reward (1,*) in collect.
    config = config.replace(n_sessions=args.n_episodes, num_envs=1, unit_noise_std=0.0, input_noise_std=0.0)

    ckpt = checkpoint_path if os.path.isabs(checkpoint_path) else os.path.join(os.getcwd(), checkpoint_path)
    print(f"Evaluating {config.exp_name}: {args.n_episodes} episodes, checkpoint dir {ckpt}")

    results, _ = evaluate_a2c_jax(
        config=config,
        checkpoint_path=ckpt,
        save_trajectories=args.save_trajectories,
    )
    print(f"\nMean reward rate: {results['mean_reward_rate']:.4f} ± {results['std_reward_rate']:.4f}  "
          f"(exp_name={config.exp_name})")


if __name__ == '__main__':
    main()
