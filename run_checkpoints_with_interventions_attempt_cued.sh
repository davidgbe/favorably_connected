#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

  python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py \
 --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_intersite_intervention.json \
 --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net0/checkpoint_0 \
 --test --save_trajectories \
 --intervention_points saved_states/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net0/fp_1_0_0_0_0_0_1_0.pkl

  python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py \
 --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_intersite_intervention.json \
 --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net1/checkpoint_0 \
 --test --save_trajectories \
 --intervention_points saved_states/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net1/fp_1_0_0_0_0_0_1_0.pkl

   python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py \
 --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_intersite_intervention.json \
 --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net2/checkpoint_0 \
 --test --save_trajectories \
 --intervention_points saved_states/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net2/fp_1_0_0_0_0_0_1_0.pkl

   python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py \
 --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_intersite_intervention.json \
 --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net3/checkpoint_0 \
 --test --save_trajectories \
 --intervention_points saved_states/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net3/fp_1_0_0_0_0_0_1_0.pkl

   python3 code/scripts/train_treadmill_agent_jax_attempt_cued.py \
 --config training_configs/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_intersite_intervention.json \
 --checkpoint_path checkpoints/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net4/checkpoint_0 \
 --test --save_trajectories \
 --intervention_points saved_states/fixed_exp_gru_reward_decay_init_scale_0p1_attempt_cued_cont_net4/fp_1_0_0_0_0_0_1_0.pkl
