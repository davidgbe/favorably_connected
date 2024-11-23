#!/bin/bash

python ./scripts/train_treadmill_agent.py \
    --exp_title run_test \
    --env CODE_OCEAN \
    --curr_style MIXED \
    --noise_var 0.001 \
    --activity_reg 1 \

# python ./code/scripts/train_treadmill_agent.py --exp_title run_test --env CODE_OCEAN --curr_style MIXED --noise_var 0.001 --activity_reg 1