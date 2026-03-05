#!/usr/bin/env python3
"""Prepare call center audio for NeMo ASR fine-tuning (v1.5).

Takes corrected transcriptions from kotib_transcribe.py output,
segments audio into 10-30s clips, creates NeMo manifests, and
merges with existing 104h training data for v1.5 training.

Usage:
    # After transcription + manual correction:
    python3 prepare_call_center_data.py

    # Custom paths:
    python3 prepare_call_center_data.py \
        --kotib-output /root/stt/call_center_output \
        --audio-dir /root/stt/call_center \
        --output-dir /root/stt/call_center_nemo \
        --merge-train /root/stt/v2_pipeline/manifests/combined_train_norm_v21.jsonl \
        --merge-val /root/stt/v2_pipeline/manifests/combined_val_norm_v21.jsonl
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

MIN_DUR = 10.0
MAX_DUR = 30.0
PAUSE_GAP = 0.5
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


def load_corrected_text(text_path: Path) -> str | None:
    """Load manually corrected text file if it exists."""
    if text_path.exists():
        return text_path.read_text(encoding="utf-8").strip()
    return None


def group_into_segments(
    words: list[dict], min_dur: float = MIN_DUR, max_dur: float = MAX_DUR
) -> list[dict]:
    """Group words into segments of min_dur-max_dur seconds."""
    if not words:
        return []

    segments = []
    seg_words = []
    seg_start = words[0]["start"]

    for i, w in enumerate(words):
        seg_words.append(w)
        seg_end = w["end"]
        seg_dur = seg_end - seg_start

        if seg_dur >= min_dur and i < len(words) - 1:
            next_w = words[i + 1]
            gap = next_w["start"] - w["end"]
            ends_sentence = w["word"].rstrip().endswith((".", "!", "?"))

            if seg_dur >= max_dur:
                segments.append(_make_segment(seg_words, seg_start, seg_end))
                seg_words = []
                seg_start = next_w["start"]
            elif ends_sentence and gap > PAUSE_GAP:
                segments.append(_make_segment(seg_words, seg_start, seg_end))
                seg_words = []
                seg_start = next_w["start"]
            elif gap > PAUSE_GAP and seg_dur >= min_dur + 2:
                segments.append(_make_segment(seg_words, seg_start, seg_end))
                seg_words = []
                seg_start = next_w["start"]

    if seg_words:
        seg_end = seg_words[-1]["end"]
        seg_dur = seg_end - seg_start
        if seg_dur >= 1.0:
            if segments and seg_dur < min_dur / 2:
                last = segments[-1]
                merged_dur = seg_end - last["start"]
                if merged_dur <= max_dur * 1.2:
                    last["end"] = seg_end
                    last["text"] += " " + " ".join(w["word"] for w in seg_words)
                    last["text"] = last["text"].strip()
                else:
                    segments.append(_make_segment(seg_words, seg_start, seg_end))
            else:
                segments.append(_make_segment(seg_words, seg_start, seg_end))

    return segments


def _make_segment(words: list[dict], start: float, end: float) -> dict:
    text = " ".join(w["word"] for w in words).strip()
    return {"start": start, "end": end, "text": text}


def extract_audio_segment(
    src_path: Path, start: float, end: float, dst_path: Path
) -> float:
    """Extract an audio slice and write as 16kHz mono int16 WAV."""
    info = sf.info(src_path)
    sr = info.samplerate
    start_frame = int(start * sr)
    end_frame = min(int(end * sr), info.frames)
    n_frames = end_frame - start_frame
    if n_frames <= 0:
        return 0.0

    audio, file_sr = sf.read(src_path, start=start_frame, frames=n_frames, dtype="int16")

    if audio.ndim > 1:
        audio = audio[:, 0]

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

        duration = extract_audio_segment(audio_path, seg["start"], seg["end"], seg_wav)
        if duration < 1.0:
            seg_wav.unlink(missing_ok=True)
            continue

        text = seg["text"].strip()
        if not text:
            seg_wav.unlink(missing_ok=True)
            continue

        entries.append({
            "audio_filepath": str(seg_wav.resolve()),
            "text": text,
            "duration": round(duration, 2),
            "source": f"call_center_{source_name}",
        })

    return entries


def create_splits(
    entries: list[dict],
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split entries by source file to avoid data leakage."""
    sources: dict[str, list[dict]] = {}
    for e in entries:
        src = e["source"]
        sources.setdefault(src, []).append(e)

    source_list = sorted(
        sources.keys(),
        key=lambda s: sum(e["duration"] for e in sources[s]),
        reverse=True,
    )

    rng = random.Random(seed)
    total_dur = sum(e["duration"] for e in entries)
    val_target = total_dur * val_ratio
    test_target = total_dur * test_ratio

    val_sources, test_sources, train_sources = [], [], []
    val_dur, test_dur = 0.0, 0.0

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

    train_sources.extend(large_sources)

    if not val_sources and len(train_sources) > 2:
        train_sources.sort(key=lambda s: sum(e["duration"] for e in sources[s]))
        val_sources.append(train_sources.pop(0))
    if not test_sources and len(train_sources) > 2:
        train_sources.sort(key=lambda s: sum(e["duration"] for e in sources[s]))
        test_sources.append(train_sources.pop(0))

    train = [e for s in train_sources for e in sources[s]]
    val = [e for s in val_sources for e in sources[s]]
    test = [e for s in test_sources for e in sources[s]]

    log.info(
        "Call center split: train=%d (%.1fh), val=%d (%.1fh), test=%d (%.1fh)",
        len(train), sum(e["duration"] for e in train) / 3600,
        len(val), sum(e["duration"] for e in val) / 3600,
        len(test), sum(e["duration"] for e in test) / 3600,
    )
    return train, val, test


def write_manifest(entries: list[dict], path: Path) -> None:
    """Write NeMo-compatible JSONL manifest."""
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            row = {
                "audio_filepath": e["audio_filepath"],
                "text": e["text"],
                "duration": e["duration"],
            }
            if "source" in e:
                row["source"] = e["source"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    log.info("Wrote %s (%d entries)", path.name, len(entries))


def merge_manifests(base_path: Path, extra_entries: list[dict], output_path: Path) -> int:
    """Merge existing manifest with new entries. Returns total count."""
    entries = []
    if base_path.exists():
        with open(base_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    entries.extend(extra_entries)
    write_manifest(entries, output_path)
    return len(entries)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare call center audio for NeMo training (v1.5)"
    )
    parser.add_argument(
        "--kotib-output", type=Path,
        default=Path("/root/stt/call_center_output"),
        help="Kotib output dir with raw_responses/ and texts/",
    )
    parser.add_argument(
        "--audio-dir", type=Path,
        default=Path("/root/stt/call_center"),
        help="Directory containing original call center MP3/WAV files",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/root/stt/call_center_nemo"),
        help="Output directory for segments and manifests",
    )
    parser.add_argument(
        "--merge-train", type=Path,
        default=Path("/root/stt/v2_pipeline/manifests/combined_train_norm_v21.jsonl"),
        help="Existing training manifest to merge with",
    )
    parser.add_argument(
        "--merge-val", type=Path,
        default=Path("/root/stt/v2_pipeline/manifests/combined_val_norm_v21.jsonl"),
        help="Existing validation manifest to merge with",
    )
    parser.add_argument(
        "--merge-test", type=Path,
        default=None,
        help="Existing test manifest to merge with (optional)",
    )
    parser.add_argument("--min-dur", type=float, default=MIN_DUR)
    parser.add_argument("--max-dur", type=float, default=MAX_DUR)
    parser.add_argument(
        "--no-merge", action="store_true",
        help="Skip merging with existing manifests (call center only)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    segments_dir = output_dir / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = args.kotib_output / "raw_responses"
    if not raw_dir.exists():
        log.error("Raw responses not found at %s", raw_dir)
        log.error("Run: python3 kotib_transcribe.py call_center/ -o call_center_output/")
        return

    # Build audio path lookup
    audio_lookup = {}
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac"):
        for f in args.audio_dir.glob(ext):
            audio_lookup[f.stem] = f

    # Process each transcribed file
    all_entries = []
    raw_files = sorted(raw_dir.glob("*.json"))
    log.info("Found %d raw response files", len(raw_files))

    for rj in raw_files:
        stem = rj.stem
        audio_path = audio_lookup.get(stem)
        if not audio_path or not audio_path.exists():
            log.warning("Audio not found for %s, skipping", stem)
            continue

        log.info("Processing: %s", stem)
        entries = process_file(
            rj, audio_path, segments_dir, source_name=stem,
            min_dur=args.min_dur, max_dur=args.max_dur,
        )
        log.info("  -> %d segments", len(entries))
        all_entries.extend(entries)

    if not all_entries:
        log.error("No segments produced. Check inputs.")
        return

    total_dur_h = sum(e["duration"] for e in all_entries) / 3600
    log.info(
        "Call center total: %d segments, %.2f hours from %d sources",
        len(all_entries), total_dur_h, len({e["source"] for e in all_entries}),
    )

    # Split call center data
    cc_train, cc_val, cc_test = create_splits(all_entries)

    # Write call center-only manifests
    write_manifest(all_entries, output_dir / "call_center_all.jsonl")
    write_manifest(cc_train, output_dir / "call_center_train.jsonl")
    write_manifest(cc_val, output_dir / "call_center_val.jsonl")
    write_manifest(cc_test, output_dir / "call_center_test.jsonl")

    # Merge with existing 104h data for v1.5 training
    if not args.no_merge:
        merged_dir = output_dir / "merged_v15"
        merged_dir.mkdir(parents=True, exist_ok=True)

        train_count = merge_manifests(
            args.merge_train, cc_train,
            merged_dir / "train_v15.jsonl",
        )
        val_count = merge_manifests(
            args.merge_val, cc_val,
            merged_dir / "val_v15.jsonl",
        )

        # For test, merge if specified, otherwise use call center test only
        if args.merge_test and args.merge_test.exists():
            test_count = merge_manifests(
                args.merge_test, cc_test,
                merged_dir / "test_v15.jsonl",
            )
        else:
            write_manifest(cc_test, merged_dir / "test_v15.jsonl")
            test_count = len(cc_test)

        log.info("=" * 60)
        log.info("Merged v1.5 manifests:")
        log.info("  Train: %d entries (104h + call center)", train_count)
        log.info("  Val:   %d entries", val_count)
        log.info("  Test:  %d entries", test_count)
        log.info("  Dir:   %s", merged_dir)
        log.info("")
        log.info("To train v1.5:")
        log.info("  python3 finetune_nemo.py \\")
        log.info("    --model-path v2_pipeline/models/uzbek_v14_strong_aug/uzbek_v14_strong_aug/final_model.nemo \\")
        log.info("    --train-manifest %s \\", merged_dir / "train_v15.jsonl")
        log.info("    --val-manifest %s \\", merged_dir / "val_v15.jsonl")
        log.info("    --test-manifest %s \\", merged_dir / "test_v15.jsonl")
        log.info("    --name uzbek_v15_call_center \\")
        log.info("    --output-dir v2_pipeline/models/uzbek_v15_call_center \\")
        log.info("    --speed-perturb --telephone-aug \\")
        log.info("    --noise-manifest v2_pipeline/noise_data/manifests/musan_all.jsonl \\")
        log.info("    --epochs 30 --lr 5e-5")

    # Print duration distribution
    durations = [e["duration"] for e in all_entries]
    log.info("")
    log.info("Duration stats: min=%.1fs, max=%.1fs, mean=%.1fs, median=%.1fs",
             min(durations), max(durations),
             sum(durations) / len(durations),
             sorted(durations)[len(durations) // 2])

    bins = [0, 5, 10, 15, 20, 25, 30, 35, 40, float("inf")]
    labels = ["<5s", "5-10s", "10-15s", "15-20s", "20-25s", "25-30s", "30-35s", "35-40s", ">40s"]
    for i in range(len(bins) - 1):
        count = sum(1 for d in durations if bins[i] <= d < bins[i + 1])
        if count:
            bar = "#" * min(count, 40)
            log.info("  %6s: %4d %s", labels[i], count, bar)


if __name__ == "__main__":
    main()
