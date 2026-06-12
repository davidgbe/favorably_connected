#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py \
#  --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued.json \
#  --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_net0/checkpoint_0 \
#  --test --save_trajectories

#  python3 code/scripts/train_treadmill_agent_jax.py \
#  --config training_configs/fixed_exp_gru_reward_decay_intersite_intervention.json \
#  --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_net2/checkpoint_450 \
#  --test --save_trajectories \
#  --intervention_points saved_states/fixed_exp_gru_reward_decay_net2/fp_1_0_0_0_0_1_0.pkl

#   python3 code/scripts/train_treadmill_agent_jax.py \
#  --config training_configs/coupled_exp_gru_initial_prob_offset_global_reward_test_one_patch_constant.json \
#  --checkpoint_path checkpoints/coupled_exp_gru_initial_prob_offset_global_reward/checkpoint_0 \
#  --test --save_trajectories \

#    python3 code/scripts/train_treadmill_agent_jax.py \
#  --config training_configs/fixed_exp_gru_initial_prob_offset_v4_trained_on_coupled.json \
#  --checkpoint_path checkpoints/coupled_exp_gru_initial_prob_offset_global_reward/checkpoint_0 \
#  --test --save_trajectories \

    python3 code/scripts/train_treadmill_agent_jax.py \
 --config training_configs/coupled_exp_gru_initial_prob_offset_global_reward_provided_one_patch_constant.json \
 --checkpoint_path checkpoints/coupled_exp_gru_initial_prob_offset_global_reward_provided/checkpoint_0 \
 --test --save_trajectories \