#!/bin/bash

# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_all_tau_0_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999

export CUDA_VISIBLE_DEVICES=0  # Uses the second GPU (index 1)
# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_global_adam_reward_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999
# python code/scripts/train_treadmill_agent_jax_multihead_value.py --exp_title gru_exp_indep_actor_non_geo_500_gammas_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999

# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_all_tau_0.001_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --env_prediction_weight 0.001

python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_global_reward_100_alpha_0.9997_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --global_reward_weight 100
# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_global_reward_1_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --global_reward_weight 1