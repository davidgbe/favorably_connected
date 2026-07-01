#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/two_stage_exp_gru_initial_prob_offset.json

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/two_stage_exp_gru_initial_prob_offset_coupled_to_fixed_to_equal.json

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/structure_evo_exp_gru_initial_prob_offset_first_250.json

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/indep_exp_gru_offset_and_decay_b.json

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/indep_exp_gru_offset_and_decay_d_test.json

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/fixed_exp_gru_initial_prob_offset_high_becomes_low.json

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#   --config training_configs/fixed_exp_gru_initial_prob_offset_low_becomes_high_slow.json

python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
  --config training_configs/indep_exp_gru_offset_and_decay_markov_test.json