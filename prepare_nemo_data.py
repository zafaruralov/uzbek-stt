#!/usr/bin/env python3
"""Segment long audio into 10-30s clips for NeMo ASR fine-tuning.

Reads word-level timestamps from Kotib raw JSON responses, groups words
into segments at natural pauses, extracts audio slices, and creates
train/val/test manifest splits (split by source file, not segment).
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import soundfile as sf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MIN_DUR = 10.0  # seconds
MAX_DUR = 30.0  # seconds
PAUSE_GAP = 0.5  # seconds — gap between words to consider a pause
SAMPLE_RATE = 16000


def load_word_timestamps(raw_json_path: Path) -> list[dict]:
    """Load word-level timestamps from a Kotib raw API response JSON."""
    data = json.loads(raw_json_path.read_text(encoding="utf-8"))
    words = []
    for seg in data.get("segments", []):
        for w in seg.get("words", []):
            if "start" in w and "end" in w and w.get("word", "").strip():
                words.append(w)
    return words


def group_into_segments(
    words: list[dict], min_dur: float = MIN_DUR, max_dur: float = MAX_DUR
) -> list[dict]:
    """Group words into segments of min_dur-max_dur seconds.

    Splits at natural pauses (gap > PAUSE_GAP between words), preferring
    splits after sentence-ending punctuation (. ! ?).
    """
    if not words:
        return []

    segments = []
    seg_words = []
    seg_start = words[0]["start"]

    for i, w in enumerate(words):
        seg_words.append(w)
        seg_end = w["end"]
        seg_dur = seg_end - seg_start

        # Check if we should split here
        if seg_dur >= min_dur and i < len(words) - 1:
            next_w = words[i + 1]
            gap = next_w["start"] - w["end"]
            ends_sentence = w["word"].rstrip().endswith((".", "!", "?"))

            # Force split if approaching max duration
            if seg_dur >= max_dur:
                segments.append(_make_segment(seg_words, seg_start, seg_end))
                seg_words = []
                seg_start = next_w["start"]
            # Prefer splitting at sentence boundaries with pauses
            elif ends_sentence and gap > PAUSE_GAP:
                segments.append(_make_segment(seg_words, seg_start, seg_end))
                seg_words = []
                seg_start = next_w["start"]
            # Split at any natural pause if we're past min duration
            elif gap > PAUSE_GAP and seg_dur >= min_dur + 2:
                segments.append(_make_segment(seg_words, seg_start, seg_end))
                seg_words = []
                seg_start = next_w["start"]

    # Remaining words
    if seg_words:
        seg_end = seg_words[-1]["end"]
        seg_dur = seg_end - seg_start
        if seg_dur >= 1.0:  # keep if at least 1 second
            if segments and seg_dur < min_dur / 2:
                # Merge short tail into last segment if it won't exceed max
                last = segments[-1]
                merged_dur = seg_end - last["start"]
                if merged_dur <= max_dur * 1.2:
                    last["end"] = seg_end
                    last["text"] += " " + " ".join(
                        w["word"] for w in seg_words
                    )
                    last["text"] = last["text"].strip()
                else:
                    segments.append(
                        _make_segment(seg_words, seg_start, seg_end)
                    )
            else:
                segments.append(_make_segment(seg_words, seg_start, seg_end))

    return segments


def _make_segment(words: list[dict], start: float, end: float) -> dict:
    text = " ".join(w["word"] for w in words).strip()
    return {"start": start, "end": end, "text": text}


def extract_audio_segment(
    src_path: Path, start: float, end: float, dst_path: Path
) -> float:
    """Extract an audio slice and write as 16kHz mono int16 WAV.

    Returns the actual duration of the written segment.
    """
    info = sf.info(src_path)
    sr = info.samplerate
    start_frame = int(start * sr)
    end_frame = int(end * sr)
    # Clamp to file length
    end_frame = min(end_frame, info.frames)
    n_frames = end_frame - start_frame
    if n_frames <= 0:
        return 0.0

    audio, file_sr = sf.read(
        src_path, start=start_frame, frames=n_frames, dtype="int16"
    )

    # Convert to mono if stereo
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Resample if needed (unlikely — our files are 16kHz)
    if file_sr != SAMPLE_RATE:
        ratio = SAMPLE_RATE / file_sr
        new_len = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
        audio = audio[indices]

    sf.write(dst_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return len(audio) / SAMPLE_RATE


def process_file(
    raw_json_path: Path,
    audio_path: Path,
    output_dir: Path,
    source_name: str,
    min_dur: float = MIN_DUR,
    max_dur: float = MAX_DUR,
) -> list[dict]:
    """Process one file: segment words, extract audio, return manifest entries."""
    words = load_word_timestamps(raw_json_path)
    if not words:
        log.warning("No words found in %s", raw_json_path.name)
        return []

    segments = group_into_segments(words, min_dur=min_dur, max_dur=max_dur)
    if not segments:
        log.warning("No segments created from %s", raw_json_path.name)
        return []

    entries = []
    for idx, seg in enumerate(segments):
        seg_name = f"{source_name}_{idx:04d}"
        seg_wav = output_dir / f"{seg_name}.wav"

        duration = extract_audio_segment(
            audio_path, seg["start"], seg["end"], seg_wav
        )
        if duration < 1.0:
            seg_wav.unlink(missing_ok=True)
            continue

        text = seg["text"].strip()
        if not text:
            seg_wav.unlink(missing_ok=True)
            continue

        entries.append(
            {
                "audio_filepath": str(seg_wav.resolve()),
                "text": text,
                "duration": round(duration, 2),
                "source": source_name,
            }
        )

    return entries


def create_splits(
    entries: list[dict],
    train_ratio: float = 0.9,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries by source file to avoid data leakage.

    All segments from one source go into the same split.
    """
    # Group by source
    sources: dict[str, list[dict]] = {}
    for e in entries:
        src = e["source"]
        sources.setdefault(src, []).append(e)

    # Sort sources by total duration (descending) for balanced packing
    source_list = sorted(
        sources.keys(),
        key=lambda s: sum(e["duration"] for e in sources[s]),
        reverse=True,
    )

    rng = random.Random(seed)

    total_dur = sum(e["duration"] for e in entries)
    val_target = total_dur * val_ratio
    test_target = total_dur * test_ratio

    # First, assign sources to val and test using bin-packing
    # Pick smaller sources for val/test to get closer to target durations
    val_sources, test_sources, train_sources = [], [], []
    val_dur, test_dur = 0.0, 0.0

    # Shuffle smaller sources to randomize which go to val vs test
    small_sources = [s for s in source_list if sum(e["duration"] for e in sources[s]) < val_target * 2]
    large_sources = [s for s in source_list if s not in set(small_sources)]
    rng.shuffle(small_sources)

    for src in small_sources:
        src_dur = sum(e["duration"] for e in sources[src])
        if val_dur < val_target:
            val_sources.append(src)
            val_dur += src_dur
        elif test_dur < test_target:
            test_sources.append(src)
            test_dur += src_dur
        else:
            train_sources.append(src)

    # All large sources go to train
    train_sources.extend(large_sources)

    # Ensure val and test aren't empty
    if not val_sources and len(train_sources) > 2:
        # Pick smallest train source for val
        train_sources.sort(key=lambda s: sum(e["duration"] for e in sources[s]))
        val_sources.append(train_sources.pop(0))
    if not test_sources and len(train_sources) > 2:
        train_sources.sort(key=lambda s: sum(e["duration"] for e in sources[s]))
        test_sources.append(train_sources.pop(0))

    train = [e for s in train_sources for e in sources[s]]
    val = [e for s in val_sources for e in sources[s]]
    test = [e for s in test_sources for e in sources[s]]

    log.info(
        "Split: train=%d segs (%.1fh, %d sources), val=%d segs (%.1fh, %d sources), test=%d segs (%.1fh, %d sources)",
        len(train), sum(e["duration"] for e in train) / 3600, len(train_sources),
        len(val), sum(e["duration"] for e in val) / 3600, len(val_sources),
        len(test), sum(e["duration"] for e in test) / 3600, len(test_sources),
    )

    return train, val, test


def write_manifest(entries: list[dict], path: Path) -> None:
    """Write NeMo-compatible JSONL manifest (without the extra 'source' field)."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            row = {
                "audio_filepath": e["audio_filepath"],
                "text": e["text"],
                "duration": e["duration"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %s (%d entries)", path.name, len(entries))


def main():
    parser = argparse.ArgumentParser(
        description="Segment audio files into 10-30s clips for NeMo fine-tuning."
    )
    parser.add_argument(
        "--kotib-output",
        type=Path,
        default=Path("/root/stt/kotib_output"),
        help="Kotib output directory containing raw_responses/ and manifest.jsonl",
    )
    parser.add_argument(
        "--saodat-raw",
        type=Path,
        default=Path("/root/stt/kotib_output/saodat_raw"),
        help="Directory with re-transcribed Saodat chunk raw_responses/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/stt/nemo_data"),
        help="Output directory for segments and manifests",
    )
    parser.add_argument(
        "--min-dur", type=float, default=MIN_DUR, help="Min segment duration (s)"
    )
    parser.add_argument(
        "--max-dur", type=float, default=MAX_DUR, help="Max segment duration (s)"
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    min_dur = args.min_dur
    max_dur = args.max_dur

    all_entries = []

    # --- Process regular files (21 non-Saodat files) ---
    raw_dir = args.kotib_output / "raw_responses"
    manifest_path = args.kotib_output / "manifest.jsonl"

    # Build audio path lookup from manifest
    audio_lookup = {}
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                stem = Path(entry["audio_filepath"]).stem
                audio_lookup[stem] = Path(entry["audio_filepath"])

    raw_files = sorted(raw_dir.glob("*.json"))
    log.info("Found %d raw response files in %s", len(raw_files), raw_dir)

    for rj in raw_files:
        stem = rj.stem
        audio_path = audio_lookup.get(stem)
        if not audio_path or not audio_path.exists():
            log.warning("Audio not found for %s, skipping", stem)
            continue

        log.info("Processing: %s", stem)
        entries = process_file(rj, audio_path, segments_dir, source_name=stem, min_dur=min_dur, max_dur=max_dur)
        log.info("  -> %d segments", len(entries))
        all_entries.extend(entries)

    # --- Process Saodat chunks ---
    saodat_raw_dir = args.saodat_raw / "raw_responses"
    saodat_chunks_dir = args.kotib_output / "saodat_chunks"

    if saodat_raw_dir.exists():
        saodat_jsons = sorted(saodat_raw_dir.glob("*.json"))
        log.info("Found %d Saodat chunk raw responses", len(saodat_jsons))

        for rj in saodat_jsons:
            chunk_name = rj.stem  # e.g. "chunk_000"
            audio_path = saodat_chunks_dir / f"{chunk_name}.wav"
            if not audio_path.exists():
                log.warning("Saodat chunk audio not found: %s", audio_path)
                continue

            source_name = f"saodat_{chunk_name}"
            log.info("Processing Saodat: %s", chunk_name)
            entries = process_file(
                rj, audio_path, segments_dir, source_name=source_name,
                min_dur=min_dur, max_dur=max_dur,
            )
            log.info("  -> %d segments", len(entries))
            all_entries.extend(entries)
    else:
        log.warning(
            "Saodat raw responses not found at %s — skipping Saodat",
            saodat_raw_dir,
        )

    if not all_entries:
        log.error("No segments produced. Check inputs.")
        return

    total_dur_h = sum(e["duration"] for e in all_entries) / 3600
    log.info(
        "Total: %d segments, %.1f hours from %d sources",
        len(all_entries),
        total_dur_h,
        len({e["source"] for e in all_entries}),
    )

    # Write combined manifest
    manifest_all = output_dir / "manifest_all.jsonl"
    write_manifest(all_entries, manifest_all)

    # Create splits
    train, val, test = create_splits(all_entries)
    write_manifest(train, output_dir / "manifest_train.jsonl")
    write_manifest(val, output_dir / "manifest_val.jsonl")
    write_manifest(test, output_dir / "manifest_test.jsonl")

    # Print duration distribution summary
    durations = [e["duration"] for e in all_entries]
    log.info("Duration stats: min=%.1fs, max=%.1fs, mean=%.1fs, median=%.1fs",
             min(durations), max(durations),
             sum(durations) / len(durations),
             sorted(durations)[len(durations) // 2])

    # Histogram
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, float("inf")]
    labels = ["<5s", "5-10s", "10-15s", "15-20s", "20-25s", "25-30s", "30-35s", "35-40s", ">40s"]
    for i in range(len(bins) - 1):
        count = sum(1 for d in durations if bins[i] <= d < bins[i + 1])
        if count:
            bar = "#" * min(count // 5, 40)
            log.info("  %6s: %4d %s", labels[i], count, bar)


if __name__ == "__main__":
    main()
