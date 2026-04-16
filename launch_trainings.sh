#!/bin/bash

python3 code/scripts/train_treadmill_agent_jax.py --config training_configs/fixed_exp_gru_reward_decay.json --n_networks 5
python3 code/scripts/train_treadmill_agent_jax.py --config training_configs/fixed_exp_gru_initial_prob_offset.json --n_networks 5