#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python3 code/scripts/train_treadmill_agent_jax.py --config training_configs/fixed/_exp_gru_reward_decay.json --n_networks 5
python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont.json --n_networks 5

# python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py \
#  --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued.json \
#  --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_net0/checkpoint_0 \
#  --test --save_trajectories