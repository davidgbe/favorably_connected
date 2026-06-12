#!/bin/bash
cd "$(dirname "$0")"
python code/scripts/train_treadmill_agent_jax.py --config training_configs/default.json
