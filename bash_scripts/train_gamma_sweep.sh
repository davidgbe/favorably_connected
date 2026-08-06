#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
# Gamma sweep: train 3 networks apiece for each gamma in {0.5, 0.8, 0.95, 0.99, 0.999}.
# Configs live in training_configs/fixed_exp_gru_initial_prob_offset_gamma_sweep/ (copies of
# fixed_exp_gru_initial_prob_offset_base.json with only gamma + exp_name changed).
#
# Run from the repo root:  bash bash_scripts/train_gamma_sweep.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# cd to repo root (the dir containing this script's parent), so output paths land correctly.
cd "$(dirname "$0")/.."

SCRIPT="code/scripts/train_treadmill_agent_jax_curriculum.py"
CONFIG_DIR="training_configs/fixed_exp_gru_initial_prob_offset_gamma_sweep"
N_NETWORKS=3

CONFIGS=(
    "gamma_0p5.json"
    "gamma_0p8.json"
    "gamma_0p95.json"
    "gamma_0p99.json"
    "gamma_0p999.json"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=================================================================="
    echo ">>> Training ${N_NETWORKS} networks for ${cfg}"
    echo "=================================================================="
    python3 "${SCRIPT}" \
        --config "${CONFIG_DIR}/${cfg}" \
        --n_networks "${N_NETWORKS}"
done

echo "All gamma-sweep runs complete."
