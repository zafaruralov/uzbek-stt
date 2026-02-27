#!/usr/bin/env python3
"""Create an unlabeled audio manifest from a directory of audio files.

Output JSONL records:
{
  "audio_filepath": "/abs/path/file.wav",
  "duration": 12.34,
  "text": "",
  "source": "user_audiobook"
}
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import soundfile as sf


SUPPORTED_EXTS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".ogg",
    ".opus",
    ".aac",
    ".wma",
    ".webm",
}


def ffprobe_duration(path: Path) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


def soundfile_duration(path: Path) -> float | None:
    try:
        info = sf.info(str(path))
        return float(info.duration)
    except Exception:
        return None


def get_duration(path: Path) -> float | None:
    d = ffprobe_duration(path)
    if d is not None and d > 0:
        return d
    d = soundfile_duration(path)
    if d is not None and d > 0:
        return d
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build unlabeled manifest from audio directory.")
    p.add_argument("--audio-dir", type=Path, required=True, help="Directory containing audio files.")
    p.add_argument("--output-manifest", type=Path, required=True, help="Output JSONL path.")
    p.add_argument("--source-label", type=str, default="user_audiobook")
    p.add_argument("--recursive", action="store_true", help="Scan subdirectories recursively.")
    p.add_argument("--min-duration", type=float, default=0.2)
    p.add_argument("--max-duration", type=float, default=7200.0, help="Max duration in seconds.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_dir.exists():
        raise FileNotFoundError(f"audio dir not found: {args.audio_dir}")

    if args.recursive:
        files = sorted(p for p in args.audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    else:
        files = sorted(p for p in args.audio_dir.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped = 0
    total_hours = 0.0
    with args.output_manifest.open("w", encoding="utf-8") as out:
        for p in files:
            dur = get_duration(p)
            if dur is None or dur < args.min_duration or dur > args.max_duration:
                dropped += 1
                continue
            rec = {
                "audio_filepath": str(p.resolve()),
                "duration": float(dur),
                "text": "",
                "source": args.source_label,
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            total_hours += dur / 3600.0

    report = {
        "input_dir": str(args.audio_dir.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "total_found": len(files),
        "kept": kept,
        "dropped": dropped,
        "hours": round(total_hours, 3),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

