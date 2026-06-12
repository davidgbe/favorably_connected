#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/two_stage_exp_gru_initial_prob_offset.json

python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
  --config training_configs/two_stage_exp_gru_initial_prob_offset_coupled_to_fixed_to_equal.json