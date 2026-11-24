#!/bin/bash
# Train a GRU on the Markov environment (reward_func_type=markov): one odor per patch type, shared
# emission (reward_prob_prefactors) and transition (patch_active_transition_prob_range) params.
# Config: training_configs/markov_pretrain.json
#
# Run from the repo root:  bash bash_scripts/train_markov_pretrain.sh
set -euo pipefail

# macOS has no CUDA, so run on CPU. On a GPU box: JAX_PLATFORMS=cuda bash bash_scripts/train_markov_pretrain.sh
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# cd to repo root (parent of this script's dir) so the relative paths resolve.
cd "$(dirname "$0")/.."

uv run python code/scripts/train_treadmill_agent_jax_curriculum.py \
  --config training_configs/markov_pretrain.json
