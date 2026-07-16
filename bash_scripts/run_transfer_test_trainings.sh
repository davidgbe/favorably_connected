#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#     --config training_configs/transfer_test_gru_markov_randomized.json \
#     --n_networks 2

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#     --config training_configs/transfer_test_gru_exp_fixed.json \
#     --n_networks 2

# python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
#     --config training_configs/transfer_test_gru_exp_indep.json \
#     --n_networks 2

python3 code/scripts/train_treadmill_agent_jax_curriculum.py \
    --config training_configs/transfer_test_gru_exp_indep_tau_capped_40.json \
    --n_networks 2