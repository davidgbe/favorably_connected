#!/bin/bash

# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_all_tau_0_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999

export CUDA_VISIBLE_DEVICES=1  # Uses the second GPU (index 1)
# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_gae_lam_0.96_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999
python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_global_reward_10_alpha_0.9997_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --global_reward_weight 10

# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_global_reward_100_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --global_reward_weight 100
# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_global_reward_25_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --global_reward_weight 25

# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_all_tau_0.001_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --env_prediction_weight 0.001