#!/bin/bash

# python code/scripts/train_treadmill_agent_jax.py \
#     --exp_title vanilla_fixed_ortho_init_v2_indep_exp_fixed_seed_3001_test_2000 \
#     --rnn_type VANILLA \
#     --seed 2000 \
#     --reward_func exp \
#     --curr_style fixed \
#     --test \
#     --save_trajectories \
#     --checkpoint_path checkpoints/vanilla_fixed_ortho_init_v2_indep_exp_fixed_seed_3001/checkpoint_1050

# python code/scripts/train_treadmill_agent_jax.py \
#     --exp_title vanilla_fixed_ortho_init_v2_indep_block_indep_seed_3002_test_2000 \
#     --rnn_type VANILLA \
#     --seed 2000 \
#     --reward_func block \
#     --curr_style indep \
#     --test \
#     --save_trajectories \
#     --checkpoint_path checkpoints/vanilla_fixed_ortho_init_v2_indep_block_indep_seed_3002/checkpoint_1200
    
python code/scripts/train_treadmill_agent_jax.py \
    --exp_title vanilla_fixed_ortho_init_v2_indep_block_indep_seed_3003_test_2000_block_fix_test_discard \
    --rnn_type VANILLA \
    --seed 2000 \
    --reward_func block \
    --curr_style indep \
    --test \
    --save_trajectories \
    --checkpoint_path checkpoints/vanilla_fixed_ortho_init_v2_indep_block_indep_seed_3003/checkpoint_1200