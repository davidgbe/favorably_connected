#!/bin/bash

python .\code\scripts\load_and_run_treadmill_agent_inter.py --exp_title test_intervention_noisy_minus_1p0 --scale_factor '-1' --load_path rl_agent_outputs/he_init_with_noise_std_all_0p1_2025-04-19_23_17_25_922161_var_noise_0.0001_activity_weight_1/rnn_weights/01800.pth --noise_var 1e-4