#!/usr/bin/env python3
"""Gradio server for the v2 open-data NeMo Uzbek ASR model."""

import tempfile
import time

import gradio as gr
import nemo.collections.asr as nemo_asr
import soundfile as sf
import numpy as np

MODEL_PATH = "/root/stt/v2_pipeline/models/uzbek_v2_open/final_model.nemo"

print("Loading NeMo v2 model...")
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
    title="Uzbek ASR v2 — Open Data FastConformer",
    description="114M param FastConformer model fine-tuned on 170K open-source Uzbek samples. Test WER: 9.69%",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866, share=False)
