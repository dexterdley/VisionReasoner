#!/bin/bash

mkdir -p ./evaluation/logs

for SEED in 42 55 69
do
	VISUAL_ALPHA=1.5 MODE="VORD" SEED=$SEED bash evaluation/eval_segmentation.sh ./ReasonSeg_test/ > ./evaluation/logs/eval_segmentation_vord_seed_$SEED.txt 2>&1
	VISUAL_ALPHA=1.5 MODE="VCD" SEED=$SEED bash evaluation/eval_segmentation.sh ./ReasonSeg_test/ > ./evaluation/logs/eval_segmentation_vcd_seed_$SEED.txt 2>&1
	VISUAL_ALPHA=0 MODE="REGULAR" SEED=$SEED bash evaluation/eval_segmentation.sh ./ReasonSeg_test/ > ./evaluation/logs/eval_segmentation_regular_seed_$SEED.txt 2>&1
done