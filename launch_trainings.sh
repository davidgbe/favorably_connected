#!/bin/bash
export CUDA_VISIBLE_DEVICES=1

# python3 code/scripts/train_treadmill_agent_jax.py --config training_configs/fixed/_exp_gru_reward_decay.json --n_networks 5
python3 code/scripts/train_treadmill_agent_jax.py --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1.json --n_networks 10
