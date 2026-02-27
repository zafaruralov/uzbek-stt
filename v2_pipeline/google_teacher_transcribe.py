#!/usr/bin/env python3
"""Transcribe a manifest with Google Speech-to-Text v2 (Chirp) as teacher labels.

This script uses direct REST calls and supports either:
- API key via --api-key / GOOGLE_API_KEY
- OAuth bearer token via --access-token / GOOGLE_ACCESS_TOKEN

Input manifest must contain at least:
- audio_filepath
- duration (optional but recommended)
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


DEFAULT_MAX_INLINE_BYTES = 9_000_000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Teacher transcription via Google Speech v2 REST.")
    p.add_argument("--input-manifest", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--errors-jsonl", type=Path, required=True)
    p.add_argument("--project-id", type=str, required=True)
    p.add_argument("--location", type=str, default="global")
    p.add_argument("--recognizer", type=str, default="_", help="Usually '_' for implicit recognizer.")
    p.add_argument("--language-code", type=str, default="uz-UZ")
    p.add_argument("--model", type=str, default="chirp_3")
    p.add_argument("--api-key", type=str, default=os.getenv("GOOGLE_API_KEY", ""))
    p.add_argument("--access-token", type=str, default=os.getenv("GOOGLE_ACCESS_TOKEN", ""))
    p.add_argument("--timeout-sec", type=float, default=180.0)
    p.add_argument("--sleep-sec", type=float, default=0.15)
    p.add_argument("--max-inline-bytes", type=int, default=DEFAULT_MAX_INLINE_BYTES)
    p.add_argument("--max-samples", type=int, default=0, help="0 = all.")
    p.add_argument("--overwrite", action="store_true", help="Start from scratch; truncate output/errors.")
    return p.parse_args()


def load_done_paths(paths: list[Path]) -> set[str]:
    done: set[str] = set()
    for p in paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ap = rec.get("audio_filepath")
                if isinstance(ap, str) and ap:
                    done.add(ap)
    return done


def make_endpoint(project_id: str, location: str, recognizer: str) -> str:
    return (
        f"https://speech.googleapis.com/v2/projects/{project_id}"
        f"/locations/{location}/recognizers/{recognizer}:recognize"
    )


def extract_transcript(resp_json: dict[str, Any]) -> tuple[str, float | None]:
    parts: list[str] = []
    confs: list[float] = []
    for item in resp_json.get("results", []):
        alts = item.get("alternatives", [])
        if not alts:
            continue
        top = alts[0]
        t = str(top.get("transcript", "")).strip()
        if t:
            parts.append(t)
        c = top.get("confidence")
        if isinstance(c, (int, float)):
            confs.append(float(c))
    transcript = " ".join(parts).strip()
    if confs:
        return transcript, sum(confs) / len(confs)
    return transcript, None


def post_recognize(
    endpoint: str,
    audio_bytes: bytes,
    language_code: str,
    model: str,
    timeout_sec: float,
    api_key: str,
    access_token: str,
) -> requests.Response:
    body = {
        "config": {
            "autoDecodingConfig": {},
            "languageCodes": [language_code],
            "model": model,
        },
        "content": base64.b64encode(audio_bytes).decode("ascii"),
    }
    headers = {"Content-Type": "application/json"}
    params: dict[str, str] = {}
    if api_key:
        params["key"] = api_key
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"
    return requests.post(endpoint, headers=headers, params=params, json=body, timeout=timeout_sec)


def main() -> None:
    args = parse_args()
    if not args.input_manifest.exists():
        raise FileNotFoundError(f"input manifest not found: {args.input_manifest}")
    if not args.api_key and not args.access_token:
        raise ValueError(
            "No auth provided. Set GOOGLE_API_KEY or GOOGLE_ACCESS_TOKEN "
            "or pass --api-key/--access-token."
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.errors_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"

    done = set()
    if not args.overwrite:
        done = load_done_paths([args.output_jsonl, args.errors_jsonl])

    endpoint = make_endpoint(args.project_id, args.location, args.recognizer)
    print(f"[teacher] endpoint={endpoint}")
    print(f"[teacher] resume_done={len(done)}")

    processed = 0
    success = 0
    failed = 0
    skipped_done = 0
    skipped_size = 0

    with args.input_manifest.open("r", encoding="utf-8") as inp, \
            args.output_jsonl.open(mode, encoding="utf-8") as out_ok, \
            args.errors_jsonl.open(mode, encoding="utf-8") as out_err:
        for line in inp:
            if args.max_samples > 0 and processed >= args.max_samples:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            audio_filepath = str(rec.get("audio_filepath", "")).strip()
            if not audio_filepath:
                continue
            if audio_filepath in done:
                skipped_done += 1
                continue

            processed += 1
            ap = Path(audio_filepath)
            if not ap.exists():
                failed += 1
                err = {
                    "audio_filepath": audio_filepath,
                    "error_type": "missing_file",
                    "message": "audio file does not exist",
                }
                out_err.write(json.dumps(err, ensure_ascii=False) + "\n")
                continue

            audio_bytes = ap.read_bytes()
            if len(audio_bytes) > args.max_inline_bytes:
                skipped_size += 1
                err = {
                    "audio_filepath": audio_filepath,
                    "error_type": "inline_size_limit",
                    "bytes": len(audio_bytes),
                    "max_inline_bytes": args.max_inline_bytes,
                }
                out_err.write(json.dumps(err, ensure_ascii=False) + "\n")
                continue

            try:
                resp = post_recognize(
                    endpoint=endpoint,
                    audio_bytes=audio_bytes,
                    language_code=args.language_code,
                    model=args.model,
                    timeout_sec=args.timeout_sec,
                    api_key=args.api_key,
                    access_token=args.access_token,
                )
                if resp.status_code >= 400:
                    failed += 1
                    err = {
                        "audio_filepath": audio_filepath,
                        "error_type": "http_error",
                        "status_code": resp.status_code,
                        "response": resp.text[:2000],
                    }
                    out_err.write(json.dumps(err, ensure_ascii=False) + "\n")
                    continue

                payload = resp.json()
                transcript, conf = extract_transcript(payload)
                if not transcript:
                    failed += 1
                    err = {
                        "audio_filepath": audio_filepath,
                        "error_type": "empty_transcript",
                        "response": payload,
                    }
                    out_err.write(json.dumps(err, ensure_ascii=False) + "\n")
                    continue

                out_rec = {
                    "audio_filepath": audio_filepath,
                    "duration": float(rec.get("duration", 0.0) or 0.0),
                    "text": transcript,
                    "teacher_confidence": conf,
                    "teacher": "google_speech_v2",
                    "teacher_model": args.model,
                    "language_code": args.language_code,
                    "project_id": args.project_id,
                    "location": args.location,
                    "recognizer": args.recognizer,
                    "source": str(rec.get("source", "teacher_input")),
                }
                original_text = str(rec.get("text", "")).strip()
                if original_text:
                    out_rec["original_text"] = original_text
                out_ok.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                success += 1
            except Exception as e:
                failed += 1
                err = {
                    "audio_filepath": audio_filepath,
                    "error_type": "exception",
                    "message": str(e),
                }
                out_err.write(json.dumps(err, ensure_ascii=False) + "\n")

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

            if processed % 25 == 0:
                print(
                    f"[teacher] processed={processed} ok={success} fail={failed} "
                    f"skip_done={skipped_done} skip_size={skipped_size}"
                )

    summary = {
        "input_manifest": str(args.input_manifest.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "errors_jsonl": str(args.errors_jsonl.resolve()),
        "processed": processed,
        "success": success,
        "failed": failed,
        "skipped_done": skipped_done,
        "skipped_size": skipped_size,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

