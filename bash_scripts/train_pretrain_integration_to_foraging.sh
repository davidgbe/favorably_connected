#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

python3 code/scripts/train_treadmill_agent_jax_pretrain_curriculum.py \
  --config training_configs/pretrain_curriculum_integration_to_foraging.json

python3 code/scripts/train_treadmill_agent_jax_pretrain_curriculum.py \
  --config training_configs/pretrain_curriculum_foraging.json
