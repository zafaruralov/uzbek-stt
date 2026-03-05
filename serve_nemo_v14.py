#!/usr/bin/env python3
"""Gradio server for Uzbek ASR v1.4 (CTC + Strong Augmentation + Audio Preprocessing).

Includes bekzod123-style audio preprocessing for call center / telephone audio:
- FFT bandpass filter (110-3500 Hz) for narrowband audio
- Adaptive noise gate
- Spectral denoising (spectral subtraction)
- Dynamic RMS normalization
"""

import tempfile
import time

import gradio as gr
import nemo.collections.asr as nemo_asr
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf
from scipy.signal import butter, sosfilt

MODEL_PATH = "/root/stt/v2_pipeline/models/uzbek_v15_call_center/uzbek_v15_call_center/final_model.nemo"
PORT = 7871

# ── Audio Preprocessing Functions ──


def bandpass_filter(audio: np.ndarray, sr: int, lowcut: float = 110, highcut: float = 3500, order: int = 5) -> np.ndarray:
    """FFT bandpass filter for narrowband/telephone audio.

    Removes low-frequency rumble (<110Hz) and high-frequency noise (>3500Hz)
    typical in telephone/call center recordings.
    """
    nyq = sr / 2
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    sos = butter(order, [low, high], btype="band", output="sos")
    return sosfilt(sos, audio).astype(np.float32)


def adaptive_noise_gate(audio: np.ndarray, sr: int, frame_ms: float = 20, threshold_factor: float = 1.5) -> np.ndarray:
    """Adaptive noise gate — suppresses frames below a dynamic threshold.

    Estimates noise floor from the quietest 10% of frames, then gates
    frames that are below threshold_factor * noise_floor.
    """
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames < 5:
        return audio

    # Compute RMS energy per frame
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-10)

    # Estimate noise floor from quietest 10% of frames
    sorted_rms = np.sort(rms)
    noise_floor = np.mean(sorted_rms[:max(1, n_frames // 10)])
    threshold = noise_floor * threshold_factor

    # Apply soft gate (smooth attenuation, not hard cutoff)
    result = audio.copy()
    for i in range(n_frames):
        start = i * frame_len
        end = start + frame_len
        if rms[i] < threshold:
            # Soft attenuation based on how far below threshold
            gain = (rms[i] / threshold) ** 2
            result[start:end] *= gain

    return result


def spectral_denoise(audio: np.ndarray, sr: int, noise_frames: int = 5, reduction_factor: float = 2.0) -> np.ndarray:
    """Spectral subtraction denoising.

    Estimates noise spectrum from the first N frames, then subtracts
    a scaled version of the noise spectrum from all frames.
    """
    n_fft = 1024
    hop = n_fft // 2

    # STFT
    n_frames_total = 1 + (len(audio) - n_fft) // hop
    if n_frames_total < noise_frames + 2:
        return audio

    window = np.hanning(n_fft).astype(np.float32)
    stft = np.zeros((n_frames_total, n_fft // 2 + 1), dtype=np.complex64)

    for i in range(n_frames_total):
        start = i * hop
        frame = audio[start:start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        stft[i] = spectrum

    # Estimate noise spectrum from first N frames
    noise_spectrum = np.mean(np.abs(stft[:noise_frames]) ** 2, axis=0)

    # Spectral subtraction
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    power = magnitude ** 2
    clean_power = np.maximum(power - reduction_factor * noise_spectrum, 0.01 * power)
    clean_magnitude = np.sqrt(clean_power)

    # Reconstruct
    clean_stft = clean_magnitude * np.exp(1j * phase)
    result = np.zeros(len(audio), dtype=np.float32)
    window_sum = np.zeros(len(audio), dtype=np.float32)

    for i in range(n_frames_total):
        start = i * hop
        frame = np.fft.irfft(clean_stft[i]).astype(np.float32) * window
        result[start:start + n_fft] += frame
        window_sum[start:start + n_fft] += window ** 2

    # Normalize by window overlap
    mask = window_sum > 1e-8
    result[mask] /= window_sum[mask]

    return result


def rms_normalize(audio: np.ndarray, target_db: float = -20) -> np.ndarray:
    """Dynamic RMS normalization to target dB level."""
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    target_rms = 10 ** (target_db / 20)
    gain = target_rms / rms
    # Limit gain to prevent amplifying silence
    gain = min(gain, 50.0)
    return (audio * gain).astype(np.float32)


def preprocess_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Full preprocessing pipeline for call center / telephone audio.

    Order matters:
    1. Bandpass — remove out-of-band noise first
    2. Spectral denoise — remove stationary noise (hum, static)
    3. Noise gate — suppress remaining quiet noise between speech
    4. RMS normalize — bring to consistent level for the model
    """
    audio = bandpass_filter(audio, sr)
    audio = spectral_denoise(audio, sr)
    audio = adaptive_noise_gate(audio, sr)
    audio = rms_normalize(audio)
    return audio


# ── Model Loading ──

print("Loading NeMo v1.4 model...")
model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(MODEL_PATH)
model.eval()
model.change_decoding_strategy(
    decoder_type="ctc",
    decoding_cfg=OmegaConf.create({"strategy": "greedy_batch"}),
)
print("Model loaded (CTC decoder)!")


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

    # Apply preprocessing if enabled
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

    # CTC decoder outputs ⁇ (U+2047) with spaces for apostrophes not in vocab
    text = text.replace(" \u2047 ", "'").replace("\u2047", "'")

    duration = len(audio) / sr
    preproc_label = "ON" if enable_preprocessing else "OFF"
    return (
        f"{text}\n\n---\n"
        f"Duration: {duration:.1f}s | Inference: {elapsed:.2f}s | RTF: {elapsed/max(duration, 1e-6):.2f}x | "
        f"Preprocessing: {preproc_label}"
    )


demo = gr.Interface(
    fn=transcribe,
    inputs=[
        gr.Audio(type="filepath", label="Audio"),
        gr.Checkbox(value=True, label="Audio Preprocessing (bandpass + denoise + noise gate + normalize)"),
    ],
    outputs=gr.Textbox(label="Transcript", lines=6),
    title="Uzbek ASR v1.4 (CTC + Strong Augmentation + Preprocessing)",
    description=(
        "Model: v1.4 — CTC fine-tuned with speed perturbation + noise injection (MUSAN) "
        "+ telephone band-pass + strong SpecAugment. 114M params.\n\n"
        "Audio preprocessing: FFT bandpass (110-3500Hz), spectral denoising, "
        "adaptive noise gate, RMS normalization — optimized for call center audio."
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
