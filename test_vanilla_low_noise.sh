#!/bin/bash

python ./code/scripts/train_treadmill_agent_cont.py \
    --exp_title structural_priors_in_rl_low_noise_optim_01_test \
    --curr_style MIXED \
    --noise_var 0.0001 \
    --activity_reg 1 \
    --load_path rl_agent_outputs/structural_priors_in_rl_low_noise_optim_01_2024-12-04_22_52_55_172250_var_noise_0.0001_activity_weight_1.0/rnn_weights/19950.h5 \

python ./code/scripts/load_and_run_treadmill_agent_v2.py --exp_title structural_priors_in_rl_low_noise_optim_01_test --curr_style MIXED --noise_var 0.0001 --activity_reg 1 --load_path rl_agent_outputs/structural_priors_in_rl_low_noise_optim_01_2024-12-04_22_52_55_172250_var_noise_0.0001_activity_weight_1.0/rnn_weights/19950.pth \