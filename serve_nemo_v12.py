#!/usr/bin/env python3
"""Gradio server for Uzbek ASR v1.2 (Saodat Asri audiobook) model."""

import tempfile
import time

import gradio as gr
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf

MODEL_PATH = "/root/stt/v2_pipeline/models/uzbek_v12_saodat/uzbek_v12_saodat/final_model.nemo"
PORT = 7868

print("Loading NeMo v1.2 model...")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH)
model.eval()
print("Model loaded!")


def transcribe(audio_path):
    if audio_path is None:
        return "Audio fayl yuklang yoki mikrofon orqali yozing."

    start = time.time()
    audio, sr = sf.read(audio_path)

    if audio.ndim > 1:
        audio = audio[:, 0]

    if sr != 16000:
        ratio = 16000 / sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
        audio = audio[indices]
        sr = 16000

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, sr, subtype="PCM_16")
        tmp_path = tmp.name

    result = model.transcribe([tmp_path], batch_size=1, verbose=False)
    elapsed = time.time() - start

    if isinstance(result, list):
        text = result[0] if isinstance(result[0], str) else result[0].text
    else:
        text = str(result)

    duration = len(audio) / sr
    return (
        f"{text}\n\n---\n"
        f"Duration: {duration:.1f}s | Inference: {elapsed:.2f}s | RTF: {elapsed/max(duration, 1e-6):.2f}x"
    )


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath", label="Audio"),
    outputs=gr.Textbox(label="Transcript", lines=6),
    title="Uzbek ASR v1.2 (Saodat Asri)",
    description=(
        "Model: v1.2 — v1.1 best + 19h Saodat Asri audiobook aligned data. "
        "104h total training data. Test WER: 8.31%"
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
