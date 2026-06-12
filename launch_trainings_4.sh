#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python3 code/scripts/train_treadmill_agent_jax.py \
#     --config training_configs/fixed_exp_gru_initial_prob_offset_v4.json \
#     --n_networks 5

# python3 code/scripts/train_treadmill_agent_jax.py \
#     --config training_configs/indep_exp_gru_initial_prob_offset_global_reward_1.json \
#     --n_networks 1

# python3 code/scripts/train_treadmill_agent_jax.py \
#     --config training_configs/indep_exp_gru_initial_prob_offset_global_reward_1e-2.json \
#     --n_networks 1

python3 code/scripts/train_treadmill_agent_jax.py \
    --config training_configs/coupled_exp_gru_initial_prob_offset_global_reward.json \
    --n_networks 1