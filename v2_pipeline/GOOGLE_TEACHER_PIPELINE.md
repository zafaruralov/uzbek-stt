# Google Teacher Pipeline (v1 -> v1.1)

This keeps your `v1` safe and trains a new candidate model.

## Safety first
- `Low`: preparing manifests and filtering JSONL files.
- `Medium`: sending audio to Google STT (privacy + cost).
- `Medium`: training (GPU load), but no destructive file deletes.

## 0) Prepare auth
You need one of these:

1. `GOOGLE_API_KEY`
2. `GOOGLE_ACCESS_TOKEN`

Example:
```bash
export GOOGLE_API_KEY="YOUR_KEY"
```

## 1) Build unlabeled manifest from your audiobook folder (Low)
```bash
python3 /root/stt/v2_pipeline/prepare_audio_manifest.py \
  --audio-dir /root/stt/raw_audio \
  --output-manifest /root/stt/v2_pipeline/manifests/user_unlabeled_audiobooks.jsonl \
  --source-label user_audiobook \
  --recursive
```

## 2) Teacher transcription with Google STT v2 Chirp (Medium)
```bash
python3 -u /root/stt/v2_pipeline/google_teacher_transcribe.py \
  --input-manifest /root/stt/v2_pipeline/manifests/user_unlabeled_audiobooks.jsonl \
  --output-jsonl /root/stt/v2_pipeline/manifests/teacher_google_raw.jsonl \
  --errors-jsonl /root/stt/v2_pipeline/manifests/teacher_google_errors.jsonl \
  --project-id YOUR_GCP_PROJECT_ID \
  --location global \
  --recognizer _ \
  --language-code uz-UZ \
  --model chirp_3
```

## 3) Filter teacher labels (Low)
```bash
python3 /root/stt/v2_pipeline/filter_teacher_manifest.py \
  --input-jsonl /root/stt/v2_pipeline/manifests/teacher_google_raw.jsonl \
  --output-jsonl /root/stt/v2_pipeline/manifests/teacher_google_clean.jsonl \
  --rejected-jsonl /root/stt/v2_pipeline/manifests/teacher_google_rejected.jsonl \
  --report-json /root/stt/v2_pipeline/reports/teacher_google_filter_report.json \
  --source-label google_teacher \
  --license-label user-provided
```

## 4) Merge private v1 + teacher (Low)
```bash
python3 /root/stt/v2_pipeline/merge_private_with_teacher.py \
  --teacher-manifest /root/stt/v2_pipeline/manifests/teacher_google_clean.jsonl \
  --out-dir /root/stt/v2_pipeline/manifests \
  --report-json /root/stt/v2_pipeline/reports/merge_teacher_v11_report.json \
  --teacher-val-ratio 0.05 \
  --private-train-repeat 2
```

Outputs:
- `/root/stt/v2_pipeline/manifests/combined_train_teacher_v11.jsonl`
- `/root/stt/v2_pipeline/manifests/combined_val_teacher_v11.jsonl`
- `/root/stt/v2_pipeline/manifests/test_private_v11.jsonl`

## 5) Train v1.1 candidate from v1 model (Medium)
```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 nice -n 10 python3 -u /root/stt/finetune_nemo.py \
  --model-path /root/stt/nemo_experiment/uzbek_fastconformer_finetune/final_model.nemo \
  --train-manifest /root/stt/v2_pipeline/manifests/combined_train_teacher_v11.jsonl \
  --val-manifest /root/stt/v2_pipeline/manifests/combined_val_teacher_v11.jsonl \
  --test-manifest /root/stt/v2_pipeline/manifests/test_private_v11.jsonl \
  --train-mode ctc-only \
  --batch-size 16 \
  --accumulate-grad-batches 4 \
  --epochs 10 \
  --lr 5e-5 \
  --freeze-encoder 1 \
  --name uzbek_v11_google_teacher \
  --output-dir /root/stt/v2_pipeline/models
```

## 6) Compare with v1 on same private test (Low)
```bash
python3 /root/stt/v2_pipeline/eval_model_v21.py \
  --model-path /root/stt/v2_pipeline/models/uzbek_v11_google_teacher/final_model.nemo \
  --manifest /root/stt/v2_pipeline/manifests/test_private_v11.jsonl \
  --output-json /root/stt/v2_pipeline/reports/eval_v11_teacher_private_test.json \
  --device cuda:0 \
  --batch-size 8
```

Only replace v1 if new model is better.

