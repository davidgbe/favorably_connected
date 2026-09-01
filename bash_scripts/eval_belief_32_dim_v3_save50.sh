#!/bin/bash
# Load the last numbered checkpoint of reff_pg_exp_gru_initial_prob_offset_belief_32_dim_v3 and run 50
# sessions in test/eval mode, saving the trajectories (for behavioral analysis).
#   --checkpoint_path points at the snapshots/ dir; flax.restore_checkpoint picks the LATEST step
#   (currently checkpoint_3000). --test_sessions 50 sets the number of sessions saved.
# Trajectories land in results/<exp_name>/trajectories_<timestamp>.pkl (exp_name from the config).
#
# Run from anywhere:  bash bash_scripts/eval_belief_32_dim_v3_save50.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}

# cd to repo root (parent of this script's dir) so relative paths resolve.
cd "$(dirname "$0")/.."

python3 code/scripts/train_treadmill_agent_jax_reff_pg.py \
    --config training_configs/reff_pg_belief_32_dim_v3_eval50.json \
    --test \
    --checkpoint_path checkpoints/reff_pg_exp_gru_initial_prob_offset_belief_32_dim_v3/snapshots \
    --test_sessions 50 \
    --save_trajectories
