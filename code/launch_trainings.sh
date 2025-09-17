#!/bin/bash

python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_exp_fixed_seed_3000 --rnn_type VANILLA --seed 3000 --reward_func exp --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_exp_fixed_seed_3001 --rnn_type VANILLA --seed 3001 --reward_func exp --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_block_indep_seed_3002 --rnn_type VANILLA --seed 3002 --reward_func block --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_block_indep_seed_3003 --rnn_type VANILLA --seed 3003 --reward_func block --curr_style indep

python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_block_fixed_seed_3004 --rnn_type VANILLA --seed 3004 --reward_func block --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_block_fixed_seed_3005 --rnn_type VANILLA --seed 3005 --reward_func block --curr_style fixed
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_exp_indep_seed_3006 --rnn_type VANILLA --seed 3006 --reward_func exp --curr_style indep
python code/scripts/train_treadmill_agent_jax.py --exp_title vanilla_fixed_ortho_init_v2_indep_exp_indep_seed_3007 --rnn_type VANILLA --seed 3007 --reward_func exp --curr_style indep