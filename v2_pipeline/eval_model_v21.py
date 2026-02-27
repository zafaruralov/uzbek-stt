#!/usr/bin/env python3
"""Evaluate a NeMo ASR model on a JSONL manifest and report WER/CER."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import nemo.collections.asr as nemo_asr
import torch


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def levenshtein(seq_a: list[str], seq_b: list[str]) -> int:
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    # seq_a is longer
    prev = list(range(len(seq_b) + 1))
    for i, a in enumerate(seq_a, start=1):
        cur = [i]
        for j, b in enumerate(seq_b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if a == b else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate .nemo model WER/CER on a manifest.")
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-json", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=0, help="0 means full manifest.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {args.manifest}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    refs: list[str] = []
    audio_paths: list[str] = []
    with args.manifest.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            refs.append(normalize_text(str(rec["text"])))
            audio_paths.append(str(rec["audio_filepath"]))
            if args.max_samples > 0 and len(audio_paths) >= args.max_samples:
                break

    print(f"[eval] loading model: {args.model_path}")
    model = nemo_asr.models.ASRModel.restore_from(str(args.model_path), map_location="cpu")
    model = model.to(device)
    model.eval()

    hyps: list[str] = []
    print(f"[eval] transcribing {len(audio_paths)} samples on {device} ...")
    for i in range(0, len(audio_paths), args.batch_size):
        batch = audio_paths[i : i + args.batch_size]
        preds = model.transcribe(batch, batch_size=len(batch), verbose=False)
        for pred in preds:
            if isinstance(pred, str):
                text = pred
            else:
                # fallback for hypothesis-like objects
                text = getattr(pred, "text", str(pred))
            hyps.append(normalize_text(text))
        if (i // args.batch_size + 1) % 50 == 0:
            print(f"[eval] processed {min(i + args.batch_size, len(audio_paths))}/{len(audio_paths)}")

    if len(hyps) != len(refs):
        raise RuntimeError(f"Prediction count mismatch: refs={len(refs)} hyps={len(hyps)}")

    total_word_err = 0
    total_ref_words = 0
    total_char_err = 0
    total_ref_chars = 0
    worst_examples: list[dict] = []

    for idx, (r, h) in enumerate(zip(refs, hyps)):
        r_words = r.split()
        h_words = h.split()
        we = levenshtein(r_words, h_words)
        total_word_err += we
        total_ref_words += max(1, len(r_words))

        r_chars = list(r)
        h_chars = list(h)
        ce = levenshtein(r_chars, h_chars)
        total_char_err += ce
        total_ref_chars += max(1, len(r_chars))

        sample_wer = we / max(1, len(r_words))
        if len(worst_examples) < 20:
            worst_examples.append(
                {
                    "idx": idx,
                    "sample_wer": round(sample_wer, 4),
                    "ref": r[:250],
                    "hyp": h[:250],
                }
            )
            worst_examples.sort(key=lambda x: x["sample_wer"], reverse=True)
        elif sample_wer > worst_examples[-1]["sample_wer"]:
            worst_examples[-1] = {
                "idx": idx,
                "sample_wer": round(sample_wer, 4),
                "ref": r[:250],
                "hyp": h[:250],
            }
            worst_examples.sort(key=lambda x: x["sample_wer"], reverse=True)

    wer = total_word_err / max(1, total_ref_words)
    cer = total_char_err / max(1, total_ref_chars)

    report = {
        "model_path": str(args.model_path.resolve()),
        "manifest": str(args.manifest.resolve()),
        "device_used": device,
        "samples": len(refs),
        "wer": round(wer, 6),
        "cer": round(cer, 6),
        "total_word_errors": int(total_word_err),
        "total_ref_words": int(total_ref_words),
        "total_char_errors": int(total_char_err),
        "total_ref_chars": int(total_ref_chars),
        "worst_examples_top20": worst_examples,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
