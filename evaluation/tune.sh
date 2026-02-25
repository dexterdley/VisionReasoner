#!/bin/bash
# Tune VORD noise_step (diffusion timestep) hyperparameter
# Usage: bash evaluation/tune.sh

set -e

mkdir -p ./evaluation/logs

NOISE_STEPS=(250 500 750 1000)
SEED=42

echo "=== VORD Noise Step Tuning ==="
echo "Steps: ${NOISE_STEPS[@]}"
echo "Seed: $SEED"
echo ""

for NS in "${NOISE_STEPS[@]}"; do
    echo ">>> Running noise_step=$NS ..."
    VISUAL_ALPHA=1.5 MODE="VORD" SEED=$SEED NOISE_STEP=$NS \
        bash evaluation/eval_segmentation.sh ./ReasonSeg_val/ \
        > ./evaluation/logs/tune_vord_ns_${NS}_seed_${SEED}.txt 2>&1
    echo "    Done. Log: ./evaluation/logs/tune_vord_ns_${NS}_seed_${SEED}.txt"
    grep -E "gIoU|cIoU|bbox_AP" "./evaluation/logs/tune_vord_ns_${NS}_seed_${SEED}.txt" 2>/dev/null | sed 's/^/    /'
    echo ""
done

echo ""
echo "=== Results Summary ==="
for NS in "${NOISE_STEPS[@]}"; do
    LOG="./evaluation/logs/tune_vord_ns_${NS}_seed_${SEED}.txt"
    echo "--- noise_step=$NS ---"
    grep -E "gIoU|cIoU|bbox_AP" "$LOG" 2>/dev/null || echo "  (no results found)"
done
