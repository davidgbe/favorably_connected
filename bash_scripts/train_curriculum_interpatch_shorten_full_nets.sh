#!/bin/bash
# Run the "interpatch_shorten" curriculum starting from each structure_evo_..._full_net{0..4} network
# (using each net's FINAL checkpoint, checkpoint_2975). Stage 2 keeps the stage-1 reward prefactors
# [0, 0.6, 0.8] and instead raises interpatch_len_decay_rate 0.1 -> 0.4, shortening interpatch runs.
# One config per net -- identical curriculum, only exp_name + init_checkpoint differ -- so each run
# writes to its own checkpoints/results dir.
#
# Configs are round-robined into one lane per GPU; lanes run concurrently, each config sequentially
# within its lane. Override the GPU set with e.g.
#   GPUS="0 1 2" bash bash_scripts/train_curriculum_interpatch_shorten_full_nets.sh
#
# Run from the repo root:  bash bash_scripts/train_curriculum_interpatch_shorten_full_nets.sh
set -euo pipefail

cd "$(dirname "$0")/.."

read -r -a GPUS <<< "${GPUS:-0 1}"
NG=${#GPUS[@]}

CONFIGS=(
    training_configs/interpatch_shorten/interpatch_shorten_full_net0.json
    training_configs/interpatch_shorten/interpatch_shorten_full_net1.json
    training_configs/interpatch_shorten/interpatch_shorten_full_net2.json
    training_configs/interpatch_shorten/interpatch_shorten_full_net3.json
    training_configs/interpatch_shorten/interpatch_shorten_full_net4.json
)

LOG_DIR="logs/interpatch_shorten_full_nets_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs -> ${LOG_DIR}"

run_lane() {
    local slot="$1"
    local gpu="${GPUS[$slot]}"
    local i cfg base
    for i in "${!CONFIGS[@]}"; do
        (( i % NG == slot )) || continue
        cfg="${CONFIGS[$i]}"
        base="$(basename "${cfg}" .json)"
        echo ">>> [GPU ${gpu}] START ${base}"
        CUDA_VISIBLE_DEVICES="${gpu}" uv run python code/scripts/train_treadmill_agent_jax_curriculum.py \
            --config "${cfg}" \
            > "${LOG_DIR}/${base}.log" 2>&1
        echo "<<< [GPU ${gpu}] DONE  ${base}  (log: ${LOG_DIR}/${base}.log)"
    done
}

pids=()
for slot in "${!GPUS[@]}"; do
    run_lane "${slot}" &
    pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
    wait "${pid}" || rc=1
done

if (( rc == 0 )); then
    echo "All ${#CONFIGS[@]} curricula trained (${NG} GPUs)."
else
    echo "One or more lanes FAILED -- check ${LOG_DIR}/*.log" >&2
fi
exit "${rc}"
