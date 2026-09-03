#!/bin/bash
# Train one network for each of the no_belief_cleaned_up JSON configs (an m_scale sweep:
# 0.5 / 0.25 / 0.1, all use_belief=False), IN PARALLEL across GPUs. Each config has a distinct
# exp_name, so checkpoints/results land in separate directories.
#
# Configs are round-robined into one lane per GPU; each lane runs its configs sequentially, and the
# lanes run concurrently. With 2 GPUs and 3 configs: GPU0 runs configs 0 & 2, GPU1 runs config 1.
# Override the GPU set with e.g.  GPUS="0 1 2" bash bash_scripts/train_reff_pg_no_belief_cleaned_up_sweep.sh
#
# Run from the repo root:  bash bash_scripts/train_reff_pg_no_belief_cleaned_up_sweep.sh
set -euo pipefail

# cd to repo root (parent of this script's dir) so the relative paths resolve.
cd "$(dirname "$0")/.."

# GPUs to spread the runs over (space-separated). Default: 0 1.
read -r -a GPUS <<< "${GPUS:-0 1}"
NG=${#GPUS[@]}

CONFIGS=(
    training_configs/reff_pg_exp_gru_initial_prob_offset_no_belief_cleaned_up.json
    training_configs/reff_pg_exp_gru_initial_prob_offset_no_belief_cleaned_up_m_scale_0p25.json
    training_configs/reff_pg_exp_gru_initial_prob_offset_no_belief_cleaned_up_m_scale_0p1.json
)

LOG_DIR="logs/no_belief_cleaned_up_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
echo "Logs -> ${LOG_DIR}"

# One lane per GPU slot: run the configs assigned to this slot (round-robin) sequentially.
run_lane() {
    local slot="$1"
    local gpu="${GPUS[$slot]}"
    local i cfg base
    for i in "${!CONFIGS[@]}"; do
        (( i % NG == slot )) || continue
        cfg="${CONFIGS[$i]}"
        base="$(basename "${cfg}" .json)"
        echo ">>> [GPU ${gpu}] START ${base}"
        CUDA_VISIBLE_DEVICES="${gpu}" python3 code/scripts/train_treadmill_agent_jax_reff_pg.py \
            --config "${cfg}" \
            --n_networks 1 \
            > "${LOG_DIR}/${base}.log" 2>&1
        echo "<<< [GPU ${gpu}] DONE  ${base}  (log: ${LOG_DIR}/${base}.log)"
    done
}

# Launch one lane per GPU in parallel; fail the script if any lane fails.
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
    echo "All ${#CONFIGS[@]} networks trained (${NG} GPUs)."
else
    echo "One or more lanes FAILED — check ${LOG_DIR}/*.log" >&2
fi
exit "${rc}"
