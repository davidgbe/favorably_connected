#!/bin/bash

python ./scripts/train_treadmill_agent_cont.py \
    --exp_title cont_vanilla_low_noise_01 \
    --env CODE_OCEAN \
    --curr_style MIXED \
    --noise_var 0.0001 \
    --activity_reg 1 \
    --load_path structural_priors_in_rl_low_noise_01_2024-12-02/rl_agent_outputs/run_test_2024-11-23_00_31_10_741317_var_noise_0.0001_activity_weight_1.0/rnn_weights/09950.h5 \
