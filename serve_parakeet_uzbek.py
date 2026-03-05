#!/usr/bin/env python3
"""Gradio server for Uzbek ASR — Parakeet TDT 0.6B (fine-tuned).

Serves the fine-tuned Parakeet TDT model with audio preprocessing
for call center / telephone audio.
"""

import argparse
import tempfile
import time

import gradio as gr
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from scipy.signal import butter, sosfilt

DEFAULT_MODEL_PATH = "/root/stt/v2_pipeline/models/parakeet_tdt_uzbek/parakeet_tdt_uzbek/final_model.nemo"


# ── Audio Preprocessing (same as serve_nemo_v14.py) ──

def bandpass_filter(audio, sr, lowcut=110, highcut=3500, order=5):
    nyq = sr / 2
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def adaptive_noise_gate(audio, sr, frame_ms=20, threshold_factor=1.5):
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames < 5:
        return audio
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)
    sorted_rms = np.sort(rms)
    noise_floor = np.mean(sorted_rms[:max(1, n_frames // 10)])
    threshold = noise_floor * threshold_factor
    result = audio.copy()
    for i in range(n_frames):
        start = i * frame_len
        end = start + frame_len
        if rms[i] < threshold:
            gain = (rms[i] / threshold) ** 2
            result[start:end] *= gain
    return result


def spectral_denoise(audio, sr, noise_frames=5, reduction_factor=2.0):
    n_fft = 1024
    hop = n_fft // 2
    n_frames_total = 1 + (len(audio) - n_fft) // hop
    if n_frames_total < noise_frames + 2:
        return audio
    window = np.hanning(n_fft).astype(np.float32)
    stft = np.zeros((n_frames_total, n_fft // 2 + 1), dtype=np.complex64)
    for i in range(n_frames_total):
        start = i * hop
        frame = audio[start:start + n_fft] * window
        stft[i] = np.fft.rfft(frame)
    noise_spectrum = np.mean(np.abs(stft[:noise_frames]) ** 2, axis=0)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    power = magnitude ** 2
    clean_power = np.maximum(power - reduction_factor * noise_spectrum, 0.01 * power)
    clean_magnitude = np.sqrt(clean_power)
    clean_stft = clean_magnitude * np.exp(1j * phase)
    result = np.zeros(len(audio), dtype=np.float32)
    window_sum = np.zeros(len(audio), dtype=np.float32)
    for i in range(n_frames_total):
        start = i * hop
        frame = np.fft.irfft(clean_stft[i]).astype(np.float32) * window
        result[start:start + n_fft] += frame
        window_sum[start:start + n_fft] += window ** 2
    mask = window_sum > 1e-8
    result[mask] /= window_sum[mask]
    return result


def rms_normalize(audio, target_db=-20):
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    target_rms = 10 ** (target_db / 20)
    gain = min(target_rms / rms, 50.0)
    return (audio * gain).astype(np.float32)


def preprocess_audio(audio, sr):
    audio = bandpass_filter(audio, sr)
    audio = spectral_denoise(audio, sr)
    audio = adaptive_noise_gate(audio, sr)
    audio = rms_normalize(audio)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--port", type=int, default=7872)
    parser.add_argument(
        "--decoder", type=str, default="tdt",
        choices=["tdt", "ctc"],
        help="Decoder to use for inference (tdt = TDT greedy, ctc = CTC greedy)",
    )
    args = parser.parse_args()

    print(f"Loading Parakeet TDT Uzbek model from {args.model_path}...")
    model = nemo_asr.models.ASRModel.restore_from(args.model_path)
    model.eval()

    if args.decoder == "ctc":
        model.change_decoding_strategy(
            decoder_type="ctc",
            decoding_cfg=OmegaConf.create({"strategy": "greedy_batch"}),
        )
        print("Using CTC decoder")
    else:
        print("Using TDT decoder (default)")

    def transcribe(audio_path, enable_preprocessing):
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

        if enable_preprocessing:
            audio = preprocess_audio(audio, sr)

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
        preproc_label = "ON" if enable_preprocessing else "OFF"
        return (
            f"{text}\n\n---\n"
            f"Duration: {duration:.1f}s | Inference: {elapsed:.2f}s | RTF: {elapsed/max(duration, 1e-6):.2f}x | "
            f"Decoder: {args.decoder} | Preprocessing: {preproc_label}"
        )

    demo = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(type="filepath", label="Audio"),
            gr.Checkbox(value=True, label="Audio Preprocessing (bandpass + denoise + noise gate + normalize)"),
        ],
        outputs=gr.Textbox(label="Transcript", lines=6),
        title="Uzbek ASR — Parakeet TDT 0.6B (Fine-tuned)",
        description=(
            "Model: NVIDIA Parakeet TDT 0.6B v3 fine-tuned for Uzbek. 600M params.\n"
            "Architecture: FastConformer encoder + TDT (Token-and-Duration Transducer).\n"
            "License: CC-BY-4.0. First open-source Parakeet-based Uzbek STT.\n\n"
            "Audio preprocessing: FFT bandpass, spectral denoising, noise gate, RMS normalization."
        ),
    )

    demo.launch(server_name="0.0.0.0", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
