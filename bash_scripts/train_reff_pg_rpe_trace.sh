#!/bin/bash
# Train a network with the belief-weighted RPE-trace policy-gradient variant.
#   advantage: A_t = sum_{s=t}^{t+K} gamma^(s-t) r_s + <D_t, b_t> + gamma^K V_{t+K} - V_t,
#   where D_t = sum_{l<t} lambda^(t-l) (r_l - E[r_l|s_l]) b_l is a leaky vector accumulator of
#   belief-weighted reward-prediction errors (the k-step return is NO LONGER M-gated; the trace is
#   added). See code/scripts/train_treadmill_agent_jax_reff_pg_rpe_trace.py.
#
# Run from the repo root:  bash bash_scripts/train_reff_pg_rpe_trace.sh
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0

# cd to repo root (parent of this script's dir) so the relative paths resolve.
cd "$(dirname "$0")/.."

python3 code/scripts/train_treadmill_agent_jax_reff_pg_rpe_trace.py \
    --config training_configs/reff_pg_rpe_trace_exp_gru_initial_prob_offset.json \
    --n_networks 1
