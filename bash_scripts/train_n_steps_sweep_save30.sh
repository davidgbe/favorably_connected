#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
# Save 30 sessions (n_save_envs=30) for the n_steps sweep {1,5,10,20} x 3 networks WITHOUT retraining:
# loads the already-generated checkpoints and does a single frozen 30-env rollout per network.
# Weights are the trained end-of-training params (training itself changed weights across its sessions).
# Output: results/<exp>_save30_net<idx>/step_00/traj_000999.pkl  (originals untouched).
#
# Run from the repo root:  bash bash_scripts/train_n_steps_sweep_save30.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# cd to repo root (the dir containing this script's parent), so paths land correctly.
cd "$(dirname "$0")/.."

SCRIPT="code/scripts/save30_from_checkpoints.py"
CONFIG_DIR="training_configs/fixed_exp_gru_initial_prob_offset_n_steps_sweep_save30"
N_NETWORKS=3

CONFIGS=(
    "n_steps_1.json"
    "n_steps_5.json"
    "n_steps_10.json"
    "n_steps_20.json"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=================================================================="
    echo ">>> Saving ${N_NETWORKS} x 30-env rollouts from checkpoints for ${cfg}"
    echo "=================================================================="
    python3 "${SCRIPT}" \
        --config "${CONFIG_DIR}/${cfg}" \
        --n_networks "${N_NETWORKS}"
done

echo "All save30-from-checkpoint rollouts complete."
