#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# n_steps_per_update sweep: train 3 networks apiece for each n_steps_per_update in {1, 5, 10, 20, 50},
# all with gamma = 0.999 and learning_rate = 1e-4 (same seed 6000 as the gamma sweep).
# Configs live in training_configs/fixed_exp_gru_initial_prob_offset_n_steps_sweep/ (copies of
# fixed_exp_gru_initial_prob_offset_base.json with only n_steps_per_update + exp_name changed).
#
# Run from the repo root:  bash bash_scripts/train_n_steps_sweep.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# cd to repo root (the dir containing this script's parent), so output paths land correctly.
cd "$(dirname "$0")/.."

SCRIPT="code/scripts/train_treadmill_agent_jax_curriculum.py"
CONFIG_DIR="training_configs/fixed_exp_gru_initial_prob_offset_n_steps_sweep"
N_NETWORKS=3

CONFIGS=(
    "n_steps_1.json"
    "n_steps_5.json"
    "n_steps_10.json"
    "n_steps_20.json"
    "n_steps_50.json"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=================================================================="
    echo ">>> Training ${N_NETWORKS} networks for ${cfg}"
    echo "=================================================================="
    python3 "${SCRIPT}" \
        --config "${CONFIG_DIR}/${cfg}" \
        --n_networks "${N_NETWORKS}"
done

echo "All n_steps-sweep runs complete."
