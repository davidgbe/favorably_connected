#!/bin/bash

python ./scripts/train_treadmill_agent.py \
    --exp_title run_test \
    --env CODE_OCEAN \
    --curr_style MIXED \
    --noise_var 0.0001 \
    --activity_reg 1 \