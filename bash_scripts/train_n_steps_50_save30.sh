#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# Save 30 sessions (n_save_envs=30) for the n_steps_50 networks WITHOUT retraining:
# loads the already-generated checkpoints and does a single frozen 30-env rollout per network.
# Weights are the trained end-of-training params (training itself changed weights across its sessions).
# Output: results/<exp>_save30_net<idx>/step_00/traj_000999.pkl  (originals untouched).
#
# Run from the repo root:  bash bash_scripts/train_n_steps_50_save30.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# cd to repo root (the dir containing this script's parent), so paths land correctly.
cd "$(dirname "$0")/.."

SCRIPT="code/scripts/save30_from_checkpoints.py"
CONFIG_DIR="training_configs/fixed_exp_gru_initial_prob_offset_n_steps_sweep_save30"
N_NETWORKS=3

python3 "${SCRIPT}" \
    --config "${CONFIG_DIR}/n_steps_50.json" \
    --n_networks "${N_NETWORKS}"

echo "n_steps_50 save30-from-checkpoint rollout complete."
