#!/bin/bash
# Train the base reward-modulated policy-gradient (r^eff) agent on a MARKOV / persistence-gated
# environment (reward_func_type=markov, reward_param_style=indep), matching the environment used in
# training_configs/persistence_gated_exp_gru_markov.json but with reff_pg's agent hyperparameters.
# See code/scripts/train_treadmill_agent_jax_reff_pg.py.
#
# Run from the repo root:  bash bash_scripts/train_reff_pg_markov_persistence.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# cd to repo root (parent of this script's dir) so the relative paths resolve.
cd "$(dirname "$0")/.."

python3 code/scripts/train_treadmill_agent_jax_reff_pg.py \
    --config training_configs/reff_pg_markov_persistence.json \
    --n_networks 1
