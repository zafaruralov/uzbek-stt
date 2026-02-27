#!/usr/bin/env python3
"""Quick test: load the fine-tuned NeMo model and transcribe a sample."""

import nemo.collections.asr as nemo_asr
import sys

MODEL_PATH = "/root/stt/nemo_experiment/uzbek_fastconformer_finetune/final_model.nemo"

print("Loading model...")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH)
model.eval()
print("Model loaded!\n")

# Transcribe files passed as arguments, or use a default sample
audio_files = sys.argv[1:] or [
    "/root/stt/nemo_data/segments/Audio kitob ｜ Jinoyat va jazo 10-trek ｜ Fyodor Dostoyevskiy_0000.wav"
]

for f in audio_files:
    print(f"File: {f}")
    result = model.transcribe([f])
    # result can be a list of strings or a Hypothesis object
    if isinstance(result, list):
        text = result[0] if isinstance(result[0], str) else result[0].text
    else:
        text = str(result)
    print(f"Text: {text}\n")
