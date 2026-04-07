#!/bin/bash

python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_all_tau_0.002_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --env_prediction_weight 0.002
# python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_pred_all_tau_0.1_gamma_0.999_seed_8002 --rnn_type GRU --seed 8002 --reward_func exp --curr_style indep --gamma 0.999 --env_prediction_weight 0.1