#!/usr/bin/env python3
"""Filter teacher transcriptions into NeMo-friendly clean manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter teacher manifest.")
    p.add_argument("--input-jsonl", type=Path, required=True)
    p.add_argument("--output-jsonl", type=Path, required=True)
    p.add_argument("--rejected-jsonl", type=Path, required=True)
    p.add_argument("--report-json", type=Path, required=True)
    p.add_argument("--min-duration", type=float, default=0.4)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--min-chars", type=int, default=3)
    p.add_argument("--max-chars", type=int, default=420)
    p.add_argument("--min-cps", type=float, default=2.5)
    p.add_argument("--max-cps", type=float, default=30.0)
    p.add_argument("--min-wps", type=float, default=0.4)
    p.add_argument("--max-wps", type=float, default=5.5)
    p.add_argument("--min-confidence", type=float, default=-1.0, help="Negative disables confidence filter.")
    p.add_argument("--source-label", type=str, default="google_teacher")
    p.add_argument("--license-label", type=str, default="user-provided")
    return p.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text


def bad_char_ratio(text: str) -> float:
    if not text:
        return 1.0
    allowed = set(" .,!?;:'\"-()[]{}")
    bad = 0
    for ch in text:
        if ch.isalpha() or ch.isdigit() or ch.isspace() or ch in allowed:
            continue
        bad += 1
    return bad / len(text)


def main() -> None:
    args = parse_args()
    if not args.input_jsonl.exists():
        raise FileNotFoundError(f"input not found: {args.input_jsonl}")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.rejected_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)

    stats = Counter()
    seen_paths: set[str] = set()

    with args.input_jsonl.open("r", encoding="utf-8") as src, \
            args.output_jsonl.open("w", encoding="utf-8") as out_ok, \
            args.rejected_jsonl.open("w", encoding="utf-8") as out_bad:
        for line in src:
            stats["total"] += 1
            line = line.strip()
            if not line:
                stats["drop_empty_line"] += 1
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats["drop_json"] += 1
                continue

            ap = str(rec.get("audio_filepath", "")).strip()
            if not ap:
                stats["drop_missing_path"] += 1
                continue
            if ap in seen_paths:
                stats["drop_duplicate_path"] += 1
                continue

            txt = normalize_text(str(rec.get("text", "")))
            dur = float(rec.get("duration", 0.0) or 0.0)
            conf = rec.get("teacher_confidence")
            conf_f = None
            if isinstance(conf, (int, float)):
                conf_f = float(conf)

            reason = None
            if not txt:
                reason = "empty_text"
            elif dur < args.min_duration:
                reason = "too_short_duration"
            elif dur > args.max_duration:
                reason = "too_long_duration"
            elif len(txt) < args.min_chars:
                reason = "too_short_text"
            elif len(txt) > args.max_chars:
                reason = "too_long_text"
            else:
                cps = len(txt) / dur if dur > 0 else 999.0
                wps = len(txt.split()) / dur if dur > 0 else 999.0
                bcr = bad_char_ratio(txt)
                if cps < args.min_cps:
                    reason = "low_cps"
                elif cps > args.max_cps:
                    reason = "high_cps"
                elif wps < args.min_wps:
                    reason = "low_wps"
                elif wps > args.max_wps:
                    reason = "high_wps"
                elif bcr > 0.25:
                    reason = "bad_char_ratio"
                elif args.min_confidence >= 0 and conf_f is not None and conf_f < args.min_confidence:
                    reason = "low_confidence"

            if reason is not None:
                stats[f"drop_{reason}"] += 1
                bad = dict(rec)
                bad["reject_reason"] = reason
                out_bad.write(json.dumps(bad, ensure_ascii=False) + "\n")
                continue

            seen_paths.add(ap)
            out = {
                "audio_filepath": ap,
                "text": txt,
                "duration": dur,
                "source": args.source_label,
                "license": args.license_label,
                "teacher": rec.get("teacher", "google_speech_v2"),
                "teacher_model": rec.get("teacher_model", "chirp_3"),
            }
            if conf_f is not None:
                out["teacher_confidence"] = conf_f
            out_ok.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["kept"] += 1

    report = {
        "input_jsonl": str(args.input_jsonl.resolve()),
        "output_jsonl": str(args.output_jsonl.resolve()),
        "rejected_jsonl": str(args.rejected_jsonl.resolve()),
        "stats": dict(stats),
        "params": {
            "min_duration": args.min_duration,
            "max_duration": args.max_duration,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_cps": args.min_cps,
            "max_cps": args.max_cps,
            "min_wps": args.min_wps,
            "max_wps": args.max_wps,
            "min_confidence": args.min_confidence,
        },
    }
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

