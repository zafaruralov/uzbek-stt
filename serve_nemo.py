#!/usr/bin/env python3
"""Simple Gradio server for the fine-tuned NeMo Uzbek ASR model."""

import tempfile
import time

import gradio as gr
import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np

MODEL_PATH = "/root/stt/nemo_experiment/uzbek_fastconformer_finetune/final_model.nemo"

print("Loading NeMo model...")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH)
model.eval()
print("Model loaded!")


def transcribe(audio_path):
    if audio_path is None:
        return "Audio fayl yuklang yoki mikrofon orqali yozing."

    start = time.time()

    # Ensure 16kHz mono WAV for NeMo
    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
        audio = audio[indices]
        sr = 16000

    # Write temp WAV
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr, subtype="PCM_16")
        tmp_path = tmp.name

    result = model.transcribe([tmp_path])
    elapsed = time.time() - start

    if isinstance(result, list):
        text = result[0] if isinstance(result[0], str) else result[0].text
    else:
        text = str(result)

    duration = len(audio) / sr
    return f"{text}\n\n---\nDuration: {duration:.1f}s | Inference: {elapsed:.2f}s | RTF: {elapsed/duration:.2f}x"


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Audio"),
    outputs=gr.Textbox(label="Transcript", lines=5),
    title="Uzbek ASR — Fine-tuned FastConformer",
    description="NeMo FastConformer-Hybrid model fine-tuned on 48h Uzbek audiobook data. WER: 8.18%",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7865, share=False)
