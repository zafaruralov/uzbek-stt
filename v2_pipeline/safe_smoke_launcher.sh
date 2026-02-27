#!/usr/bin/env bash
set -euo pipefail

# Wait until GPU1 memory usage is below threshold, then run a small smoke train.
# This avoids disrupting other services.

GPU_INDEX="${GPU_INDEX:-1}"
MAX_USED_MIB="${MAX_USED_MIB:-8000}"
SLEEP_SEC="${SLEEP_SEC:-60}"

LOG_DIR="/root/stt/v2_pipeline/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/smoke_run_$(date +%Y%m%d_%H%M%S).log"

echo "[safe-smoke] waiting for GPU${GPU_INDEX} used_mem <= ${MAX_USED_MIB} MiB"
echo "[safe-smoke] log: $LOG_FILE"

while true; do
  USED="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_INDEX+1))p" | tr -d ' ')"
  if [[ -z "${USED:-}" ]]; then
    echo "[safe-smoke] could not read GPU memory; retrying in ${SLEEP_SEC}s"
    sleep "$SLEEP_SEC"
    continue
  fi
  TS="$(date '+%F %T')"
  echo "[safe-smoke] $TS gpu${GPU_INDEX}_used=${USED}MiB"
  if (( USED <= MAX_USED_MIB )); then
    echo "[safe-smoke] threshold met; starting smoke training"
    break
  fi
  sleep "$SLEEP_SEC"
done

CUDA_VISIBLE_DEVICES="$GPU_INDEX" OMP_NUM_THREADS=8 nice -n 10 \
python3 /root/stt/finetune_nemo.py \
  --train-manifest /root/stt/v2_pipeline/manifests/open_train_norm_clean_v1.jsonl \
  --val-manifest /root/stt/v2_pipeline/manifests/open_val_norm_clean_v1.jsonl \
  --test-manifest /root/stt/v2_pipeline/manifests/open_test_norm_clean_v1.jsonl \
  --train-mode ctc-only \
  --batch-size 8 \
  --accumulate-grad-batches 2 \
  --epochs 1 \
  --max-steps 50 \
  --name uzbek_v2_smoke \
  --output-dir /root/stt/v2_pipeline/models \
  2>&1 | tee "$LOG_FILE"

echo "[safe-smoke] done"
