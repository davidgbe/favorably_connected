#!/bin/bash
# Train a network with the reward-modulated policy-gradient (r^eff) loss.
#   loss: per-step effective reward (luck-modulated reward at reward steps, -E[r] baseline on empty
#   steps), credited back over each inter-reward interval and averaged over the chunk. See
#   code/scripts/train_treadmill_agent_jax_reff_pg.py.
#
# Run from the repo root:  bash bash_scripts/train_reff_pg.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# cd to repo root (parent of this script's dir) so the relative paths resolve.
cd "$(dirname "$0")/.."

python3 code/scripts/train_treadmill_agent_jax_reff_pg.py \
    --config training_configs/reff_pg_exp_gru_initial_prob_offset_belief_cleaned_up.json \
    --n_networks 1
