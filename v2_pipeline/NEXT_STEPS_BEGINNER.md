# Uzbek STT v2 - Beginner Next Steps

## What is already done
- Datasets downloaded and prepared.
- Audio exported to stable `.wav` paths.
- Clean manifests created (`clean_v1`).

## Your next action (safe smoke test, small run)
This checks everything works before full training.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 nice -n 10 \
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
  --output-dir /root/stt/v2_pipeline/models
```

## If smoke test is good, start full training (later/off-hours)
```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 nice -n 10 \
python3 /root/stt/finetune_nemo.py \
  --train-manifest /root/stt/v2_pipeline/manifests/open_train_norm_clean_v1.jsonl \
  --val-manifest /root/stt/v2_pipeline/manifests/open_val_norm_clean_v1.jsonl \
  --test-manifest /root/stt/v2_pipeline/manifests/open_test_norm_clean_v1.jsonl \
  --train-mode ctc-only \
  --batch-size 16 \
  --accumulate-grad-batches 4 \
  --epochs 12 \
  --lr 1e-4 \
  --freeze-encoder 2 \
  --name uzbek_v2_ctc_clean_v1 \
  --output-dir /root/stt/v2_pipeline/models
```

## Monitor
```bash
nvidia-smi
```

## Auto-safe option (wait for free GPU1, then smoke test)
```bash
/root/stt/v2_pipeline/safe_smoke_launcher.sh
```
Default waits until GPU1 memory usage is `<= 8000 MiB`, then starts the small smoke run.
