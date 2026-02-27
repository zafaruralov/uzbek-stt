#!/usr/bin/env python3
"""Conservative quality filter for v2 manifests.

Reads raw+norm manifests, keeps only sane audio/text pairs, writes clean_v1 outputs.
Original manifests are never modified.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/root/stt/v2_pipeline")
MANIFESTS = ROOT / "manifests"
REPORTS = ROOT / "reports"

SPLITS = [
    "open_train",
    "open_val",
    "open_test",
    "test_fleurs",
]

# Conservative thresholds; intended to drop only obvious errors.
MIN_DURATION_SEC = 0.4
MAX_DURATION_SEC = 30.0
MIN_TEXT_CHARS = 2
MAX_TEXT_CHARS = 420
MIN_CHARS_PER_SEC = 3.0
MAX_CHARS_PER_SEC = 30.0
MIN_WORDS_PER_SEC = 0.4
MAX_WORDS_PER_SEC = 5.0
MAX_BAD_CHAR_RATIO = 0.2

ALLOWED_PUNCT = set(" .,!?;:'\"-()[]{}")


def bad_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    bad = 0
    for ch in text:
        if ch.isalpha() or ch.isdigit() or ch.isspace() or ch in ALLOWED_PUNCT:
            continue
        bad += 1
    return bad / len(text)


def reject_reason(text: str, duration: float) -> str | None:
    t = text.strip()
    if not t:
        return "empty_text"
    if duration < MIN_DURATION_SEC:
        return "too_short_audio"
    if duration > MAX_DURATION_SEC:
        return "too_long_audio"
    if len(t) < MIN_TEXT_CHARS:
        return "too_short_text"
    if len(t) > MAX_TEXT_CHARS:
        return "too_long_text"

    cps = len(t) / duration
    wps = len(t.split()) / duration
    bcr = bad_char_ratio(t)

    if cps < MIN_CHARS_PER_SEC:
        return "low_cps"
    if cps > MAX_CHARS_PER_SEC:
        return "high_cps"
    if wps < MIN_WORDS_PER_SEC:
        return "low_wps"
    if wps > MAX_WORDS_PER_SEC:
        return "high_wps"
    if bcr > MAX_BAD_CHAR_RATIO:
        return "bad_char_ratio"
    return None


def process_split(split: str) -> dict:
    raw_in = MANIFESTS / f"{split}_raw.jsonl"
    norm_in = MANIFESTS / f"{split}_norm.jsonl"
    raw_out = MANIFESTS / f"{split}_raw_clean_v1.jsonl"
    norm_out = MANIFESTS / f"{split}_norm_clean_v1.jsonl"

    if not raw_in.exists() or not norm_in.exists():
        raise FileNotFoundError(f"Missing input manifests for split={split}")

    total = 0
    kept = 0
    reasons = Counter()
    mismatch_paths = 0
    seen_audio = set()
    dup_audio = 0

    with raw_in.open("r", encoding="utf-8") as f_raw_in, norm_in.open(
        "r", encoding="utf-8"
    ) as f_norm_in, raw_out.open("w", encoding="utf-8") as f_raw_out, norm_out.open(
        "w", encoding="utf-8"
    ) as f_norm_out:
        for raw_line, norm_line in zip(f_raw_in, f_norm_in):
            total += 1
            raw_rec = json.loads(raw_line)
            norm_rec = json.loads(norm_line)

            raw_path = raw_rec.get("audio_filepath", "")
            norm_path = norm_rec.get("audio_filepath", "")
            if raw_path != norm_path:
                mismatch_paths += 1

            if raw_path in seen_audio:
                dup_audio += 1
                reasons["duplicate_audio"] += 1
                continue
            seen_audio.add(raw_path)

            text = str(raw_rec.get("text", ""))
            duration = float(raw_rec.get("duration", 0.0))
            reason = reject_reason(text, duration)
            if reason is not None:
                reasons[reason] += 1
                continue

            f_raw_out.write(raw_line)
            f_norm_out.write(norm_line)
            kept += 1

    dropped = total - kept
    return {
        "input_raw": str(raw_in),
        "input_norm": str(norm_in),
        "output_raw_clean_v1": str(raw_out),
        "output_norm_clean_v1": str(norm_out),
        "total": total,
        "kept": kept,
        "dropped": dropped,
        "drop_rate_pct": round((100.0 * dropped / total) if total else 0.0, 4),
        "reasons": dict(reasons),
        "mismatch_audio_filepath_between_raw_norm": mismatch_paths,
        "duplicate_audio_paths": dup_audio,
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "version": "filter_v1",
        "thresholds": {
            "MIN_DURATION_SEC": MIN_DURATION_SEC,
            "MAX_DURATION_SEC": MAX_DURATION_SEC,
            "MIN_TEXT_CHARS": MIN_TEXT_CHARS,
            "MAX_TEXT_CHARS": MAX_TEXT_CHARS,
            "MIN_CHARS_PER_SEC": MIN_CHARS_PER_SEC,
            "MAX_CHARS_PER_SEC": MAX_CHARS_PER_SEC,
            "MIN_WORDS_PER_SEC": MIN_WORDS_PER_SEC,
            "MAX_WORDS_PER_SEC": MAX_WORDS_PER_SEC,
            "MAX_BAD_CHAR_RATIO": MAX_BAD_CHAR_RATIO,
        },
        "splits": {},
    }

    for split in SPLITS:
        print(f"[filter] processing {split} ...")
        report["splits"][split] = process_split(split)
        s = report["splits"][split]
        print(
            f"[filter] {split}: kept={s['kept']} dropped={s['dropped']} "
            f"drop_rate={s['drop_rate_pct']}%"
        )

    out_report = REPORTS / "manifest_filter_v1_report.json"
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[filter] report: {out_report}")


if __name__ == "__main__":
    main()
