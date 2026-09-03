#!/bin/bash
# Load the LAST checkpoint of the clean_ms0p5 (use_belief=False, m_scale=0.5) run and evaluate it.
# The config JSON is an exact copy of the ms0p5 training config plus a "checkpoint_path" key pointing
# at the run's snapshots/ dir (flax restore_checkpoint loads the latest checkpoint there).
# Trajectories are saved to results/<exp_name>/ (where reff_behavioral_traces.ipynb reads them).
#
# Run from the repo root:  bash bash_scripts/eval_reff_pg_no_belief_clean_ms0p5.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
cd "$(dirname "$0")/.."

python3 code/scripts/eval_reff_pg_from_config.py \
    training_configs/reff_pg_exp_gru_initial_prob_offset_no_belief_clean_ms0p5_eval.json \
    --n_episodes 30 \
    --save_trajectories
