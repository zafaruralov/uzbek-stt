#!/usr/bin/env python3
"""Re-transcribe audio files via the Kotib STT API and output NeMo manifests."""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ASYNC_THRESHOLD = 1 * 1024 * 1024 * 1024  # 1 GB
POLL_INTERVAL = 15  # seconds between async status checks
MAX_RETRIES = 3
RETRY_BACKOFF = 10  # seconds, doubles each retry
SYNC_TIMEOUT = 7200  # 2 hours for sync requests


def discover_audio_files(input_path: Path) -> list[Path]:
    """Return sorted list of WAV files from a file or directory."""
    if input_path.is_file():
        return [input_path]
    exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm", ".opus"}
    files = sorted(
        f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in exts
    )
    return files


def transcribe_sync(filepath: Path, api_url: str) -> dict:
    """Send file to sync endpoint and return the response JSON."""
    url = f"{api_url}/api/v1/transcribe"
    params = {"word_timestamps": "true", "diarization_mode": "none"}
    size_mb = filepath.stat().st_size / (1024 * 1024)
    log.info("Uploading %s (%.0f MB) to sync endpoint...", filepath.name, size_mb)
    with open(filepath, "rb") as f:
        resp = requests.post(
            url,
            params=params,
            files={"audio_file": (filepath.name, f, "audio/wav")},
            timeout=SYNC_TIMEOUT,
        )
    resp.raise_for_status()
    return resp.json()


def transcribe_async(filepath: Path, api_url: str) -> dict:
    """Submit file to async endpoint, poll until done, return the result."""
    submit_url = f"{api_url}/api/v1/jobs/submit"
    size_mb = filepath.stat().st_size / (1024 * 1024)
    log.info("Uploading %s (%.0f MB) to async endpoint...", filepath.name, size_mb)

    with open(filepath, "rb") as f:
        resp = requests.post(
            submit_url,
            files={"file": (filepath.name, f, "audio/wav")},
            data={"word_timestamps": "true", "diarization_mode": "none"},
            timeout=600,
        )
    resp.raise_for_status()
    job = resp.json()
    job_id = job["job_id"]
    log.info("Job submitted: %s", job_id)

    # Poll for completion
    status_url = f"{api_url}/api/v1/jobs/{job_id}/status"
    result_url = f"{api_url}/api/v1/jobs/{job_id}/result"

    while True:
        time.sleep(POLL_INTERVAL)
        status_resp = requests.get(status_url, timeout=30)
        status_resp.raise_for_status()
        status = status_resp.json()["status"]

        if status == "completed":
            log.info("Job %s completed", job_id)
            result_resp = requests.get(result_url, timeout=60)
            result_resp.raise_for_status()
            return result_resp.json().get("result", result_resp.json())

        if status == "failed":
            error = status_resp.json()
            raise RuntimeError(f"Job {job_id} failed: {error}")

        log.info("Job %s status: %s — polling in %ds...", job_id, status, POLL_INTERVAL)


def transcribe_file(filepath: Path, api_url: str) -> dict:
    """Transcribe a file, choosing sync or async based on size. Retries on failure."""
    use_async = filepath.stat().st_size >= ASYNC_THRESHOLD
    last_err = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if use_async:
                return transcribe_async(filepath, api_url)
            else:
                return transcribe_sync(filepath, api_url)
        except requests.exceptions.Timeout:
            if not use_async:
                log.warning("Sync timed out for %s, falling back to async", filepath.name)
                use_async = True
                continue
            last_err = f"Timeout on attempt {attempt}"
        except requests.exceptions.ConnectionError as e:
            last_err = str(e)
        except requests.exceptions.HTTPError as e:
            last_err = str(e)
        except RuntimeError as e:
            last_err = str(e)
            break  # async job failed server-side, don't retry

        backoff = RETRY_BACKOFF * (2 ** (attempt - 1))
        log.warning("Attempt %d/%d failed (%s). Retrying in %ds...", attempt, MAX_RETRIES, last_err, backoff)
        time.sleep(backoff)

    log.error("All attempts failed for %s: %s", filepath.name, last_err)
    return None


def write_srt(segments: list[dict], output_path: Path) -> None:
    """Write SRT subtitle file from segments with word timestamps."""
    with open(output_path, "w", encoding="utf-8") as f:
        idx = 1
        for seg in segments:
            words = seg.get("words")
            if words:
                # Group words into subtitle lines (~10 words each)
                for i in range(0, len(words), 10):
                    chunk = words[i : i + 10]
                    start = chunk[0].get("start", seg["start"])
                    end = chunk[-1].get("end", seg["end"])
                    text = " ".join(w.get("word", w.get("text", "")) for w in chunk).strip()
                    if not text:
                        continue
                    f.write(f"{idx}\n")
                    f.write(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}\n")
                    f.write(f"{text}\n\n")
                    idx += 1
            else:
                text = seg.get("text", "").strip()
                if not text:
                    continue
                f.write(f"{idx}\n")
                f.write(f"{_fmt_srt_time(seg['start'])} --> {_fmt_srt_time(seg['end'])}\n")
                f.write(f"{text}\n\n")
                idx += 1


def _fmt_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_manifest(results: list[dict], output_path: Path) -> None:
    """Write NeMo-compatible JSONL manifest."""
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log.info("Manifest written: %s (%d entries)", output_path, len(results))


def load_completed(manifest_path: Path) -> set[str]:
    """Load already-completed audio filepaths from an existing manifest for resume."""
    done = set()
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    done.add(entry["audio_filepath"])
    return done


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio files via Kotib STT API and produce NeMo manifests."
    )
    parser.add_argument(
        "input", type=Path, help="Audio file or directory of audio files"
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=None,
        help="Output directory (default: kotib_output/ next to input)"
    )
    parser.add_argument(
        "--api-url", default="http://localhost:8080",
        help="Kotib STT API base URL (default: http://localhost:8080)"
    )
    parser.add_argument(
        "--no-srt", action="store_true", help="Skip SRT subtitle generation"
    )
    args = parser.parse_args()

    # Resolve paths
    input_path = args.input.resolve()
    if args.output_dir:
        output_dir = args.output_dir.resolve()
    else:
        parent = input_path.parent if input_path.is_file() else input_path
        output_dir = parent.parent / "kotib_output"

    # Create output directories
    texts_dir = output_dir / "texts"
    srt_dir = output_dir / "srt"
    raw_dir = output_dir / "raw_responses"
    for d in [output_dir, texts_dir, srt_dir, raw_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"

    # Discover files
    files = discover_audio_files(input_path)
    if not files:
        log.error("No audio files found in %s", input_path)
        sys.exit(1)
    log.info("Found %d audio files", len(files))

    # Check API health
    try:
        health = requests.get(f"{args.api_url}/api/v1/health", timeout=10).json()
        log.info("API health: %s", health)
    except Exception as e:
        log.error("API not reachable at %s: %s", args.api_url, e)
        sys.exit(1)

    # Resume support — load already completed files
    completed = load_completed(manifest_path)
    if completed:
        log.info("Resuming: %d files already in manifest, skipping them", len(completed))

    # Load existing manifest entries for resume
    existing_entries = []
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_entries.append(json.loads(line))

    results = list(existing_entries)
    total = len(files)
    skipped = 0
    failed = 0

    for i, filepath in enumerate(files, 1):
        abs_path = str(filepath.resolve())

        if abs_path in completed:
            log.info("[%d/%d] Skipping (already done): %s", i, total, filepath.name)
            skipped += 1
            continue

        log.info("[%d/%d] Transcribing: %s", i, total, filepath.name)
        start_time = time.time()

        response = transcribe_file(filepath, args.api_url)

        if response is None:
            log.error("[%d/%d] FAILED: %s", i, total, filepath.name)
            failed += 1
            continue

        elapsed = time.time() - start_time
        text = response.get("text", "")
        duration = response.get("duration", 0)
        segments = response.get("segments", [])

        if not text:
            log.warning("[%d/%d] Empty transcript for %s", i, total, filepath.name)

        log.info(
            "[%d/%d] Done in %.0fs — duration: %.1fs, chars: %d, segments: %d",
            i, total, elapsed, duration, len(text), len(segments),
        )

        # Save raw API response
        raw_path = raw_dir / (filepath.stem + ".json")
        with open(raw_path, "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        # Save plain text
        txt_path = texts_dir / (filepath.stem + ".txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)

        # Save SRT
        if not args.no_srt and segments:
            srt_path = srt_dir / (filepath.stem + ".srt")
            write_srt(segments, srt_path)

        # Append to manifest
        entry = {
            "audio_filepath": abs_path,
            "text": text,
            "duration": round(duration, 2),
        }
        results.append(entry)

        # Write manifest after each file (incremental save for resume)
        write_manifest(results, manifest_path)

    # Summary
    done = len(results)
    total_duration = sum(r["duration"] for r in results)
    hours = total_duration / 3600
    log.info("=" * 60)
    log.info("Complete: %d/%d files transcribed (%.1f hours)", done, total, hours)
    if skipped:
        log.info("Skipped (resumed): %d", skipped)
    if failed:
        log.info("Failed: %d", failed)
    log.info("Manifest: %s", manifest_path)


if __name__ == "__main__":
    main()
