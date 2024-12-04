#!/bin/bash

python ./scripts/train_treadmill_agent.py \
    --exp_title structural_priors_in_rl_low_noise_optim \
    --env CODE_OCEAN \
    --curr_style MIXED \
    --noise_var 0.0001 \
    --activity_reg 1 \

# python ./code/scripts/train_treadmill_agent.py --exp_title run_test --curr_style MIXED --noise_var 0.0001 --activity_reg 1