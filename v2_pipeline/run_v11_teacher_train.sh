#!/usr/bin/env bash
set -euo pipefail

# Train candidate model from v1 base + teacher merged manifests.
# Safety: medium (uses GPU/CPU), non-destructive.

GPU_INDEX="${GPU_INDEX:-0}"
NAME="${NAME:-uzbek_v11_google_teacher}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-5e-5}"
BS="${BS:-16}"
ACC="${ACC:-4}"

CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=8 nice -n 10 \
python3 -u /root/stt/finetune_nemo.py \
  --model-path /root/stt/nemo_experiment/uzbek_fastconformer_finetune/final_model.nemo \
  --train-manifest /root/stt/v2_pipeline/manifests/combined_train_teacher_v11.jsonl \
  --val-manifest /root/stt/v2_pipeline/manifests/combined_val_teacher_v11.jsonl \
  --test-manifest /root/stt/v2_pipeline/manifests/test_private_v11.jsonl \
  --train-mode ctc-only \
  --batch-size "$BS" \
  --accumulate-grad-batches "$ACC" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --freeze-encoder 1 \
  --name "$NAME" \
  --output-dir /root/stt/v2_pipeline/models

