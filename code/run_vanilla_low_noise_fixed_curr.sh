#!/bin/bash

python ./scripts/train_treadmill_agent.py \
    --exp_title structural_priors_in_rl_fixed_curr_optim \
    --env CODE_OCEAN \
    --curr_style FIXED \
    --noise_var 0.0001 \
    --activity_reg 1 \