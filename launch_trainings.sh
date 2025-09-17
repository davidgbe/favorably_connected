#!/bin/bash

python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_exp_fixed_seed_5000 --rnn_type VANILLA --seed 5000 --reward_func exp --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_exp_fixed_seed_5001 --rnn_type VANILLA --seed 5001 --reward_func exp --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_block_fixed_seed_5002 --rnn_type VANILLA --seed 5002 --reward_func block --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_block_fixed_seed_5003 --rnn_type VANILLA --seed 5003 --reward_func block --curr_style fixed

python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_exp_fixed_seed_5004 --rnn_type VANILLA --seed 5004 --reward_func exp --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_exp_fixed_seed_5005 --rnn_type VANILLA --seed 5005 --reward_func exp --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_block_fixed_seed_5006 --rnn_type VANILLA --seed 5006 --reward_func block --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_block_fixed_seed_5007 --rnn_type VANILLA --seed 5007 --reward_func block --curr_style indep

python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_fixed_seed_5000 --rnn_type GRU --seed 5000 --reward_func exp --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_fixed_seed_5001 --rnn_type GRU --seed 5001 --reward_func exp --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title gru_block_fixed_seed_5002 --rnn_type GRU --seed 5002 --reward_func block --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title gru_block_fixed_seed_5003 --rnn_type GRU --seed 5003 --reward_func block --curr_style fixed

python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_seed_5004 --rnn_type GRU --seed 5004 --reward_func exp --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title gru_exp_indep_seed_5005 --rnn_type GRU --seed 5005 --reward_func exp --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title gru_block_indep_seed_5006 --rnn_type GRU --seed 5006 --reward_func block --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title gru_block_indep_seed_5007 --rnn_type GRU --seed 5007 --reward_func block --curr_style indep