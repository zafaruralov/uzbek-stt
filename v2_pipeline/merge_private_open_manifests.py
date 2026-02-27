#!/usr/bin/env python3
"""Build v2.1 combined manifests from private (v1) and open datasets.

Outputs:
- combined_train_norm_v21.jsonl
- combined_val_norm_v21.jsonl
- test_private_norm_v21.jsonl
- test_open_norm_v21.jsonl
- test_fleurs_norm_v21.jsonl
- reports/merge_v21_report.json
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


MIN_DURATION = 0.4
MAX_DURATION = 30.0


@dataclass
class LoadStats:
    total_lines: int = 0
    kept: int = 0
    drop_json_error: int = 0
    drop_missing_field: int = 0
    drop_empty_text: int = 0
    drop_bad_duration: int = 0
    drop_missing_file: int = 0
    drop_duplicate_path: int = 0

    def to_dict(self) -> dict:
        return {
            "total_lines": self.total_lines,
            "kept": self.kept,
            "drop_json_error": self.drop_json_error,
            "drop_missing_field": self.drop_missing_field,
            "drop_empty_text": self.drop_empty_text,
            "drop_bad_duration": self.drop_bad_duration,
            "drop_missing_file": self.drop_missing_file,
            "drop_duplicate_path": self.drop_duplicate_path,
        }


def source_private(split: str) -> str:
    return f"private_v1_{split}"


def source_open(raw: str) -> str:
    return f"open_{raw}" if raw else "open_unknown"


def load_manifest(
    manifest_path: Path,
    source_fn: Callable[[str], str],
    default_source: str,
    default_license: str,
) -> tuple[list[dict], LoadStats]:
    stats = LoadStats()
    out: list[dict] = []
    seen_paths: set[str] = set()

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            stats.total_lines += 1
            line = line.strip()
            if not line:
                stats.drop_json_error += 1
                continue

            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                stats.drop_json_error += 1
                continue

            if "audio_filepath" not in rec or "text" not in rec or "duration" not in rec:
                stats.drop_missing_field += 1
                continue

            raw_path = str(rec["audio_filepath"]).strip()
            text = str(rec["text"]).strip()
            try:
                duration = float(rec["duration"])
            except Exception:
                stats.drop_bad_duration += 1
                continue

            if not text:
                stats.drop_empty_text += 1
                continue
            if duration < MIN_DURATION or duration > MAX_DURATION:
                stats.drop_bad_duration += 1
                continue

            path_obj = Path(raw_path)
            if not path_obj.is_absolute():
                path_obj = (manifest_path.parent / path_obj).resolve()
            else:
                path_obj = path_obj.resolve()

            path_abs = str(path_obj)
            if not path_obj.exists():
                stats.drop_missing_file += 1
                continue
            if path_abs in seen_paths:
                stats.drop_duplicate_path += 1
                continue
            seen_paths.add(path_abs)

            src_raw = str(rec.get("source", default_source)).strip()
            src = source_fn(src_raw) if src_raw else source_fn(default_source)
            lic = str(rec.get("license", default_license)).strip() or default_license

            out.append(
                {
                    "audio_filepath": path_abs,
                    "text": text,
                    "duration": duration,
                    "source": src,
                    "license": lic,
                }
            )
            stats.kept += 1
    return out, stats


def filter_and_merge(records_list: list[list[dict]], blocked_paths: set[str]) -> tuple[list[dict], Counter]:
    """Merge lists while enforcing unique paths and avoiding blocked paths."""
    merged: list[dict] = []
    local_seen: set[str] = set()
    drops = Counter()

    for records in records_list:
        for rec in records:
            p = rec["audio_filepath"]
            if p in blocked_paths:
                drops["drop_overlap_blocked"] += 1
                continue
            if p in local_seen:
                drops["drop_overlap_local"] += 1
                continue
            local_seen.add(p)
            merged.append(rec)
    return merged, drops


def summarize_split(records: list[dict]) -> dict:
    hours = sum(float(r["duration"]) for r in records) / 3600.0
    by_source_count: Counter = Counter()
    by_source_hours: Counter = Counter()
    for r in records:
        s = r["source"]
        by_source_count[s] += 1
        by_source_hours[s] += float(r["duration"]) / 3600.0
    return {
        "samples": len(records),
        "hours": round(hours, 3),
        "by_source": {
            s: {"samples": int(by_source_count[s]), "hours": round(float(by_source_hours[s]), 3)}
            for s in sorted(by_source_count.keys())
        },
    }


def write_jsonl(records: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge private + open manifests for v2.1 training.")
    p.add_argument("--private-train", type=Path, default=Path("/root/stt/nemo_data/manifest_train.jsonl"))
    p.add_argument("--private-val", type=Path, default=Path("/root/stt/nemo_data/manifest_val.jsonl"))
    p.add_argument("--private-test", type=Path, default=Path("/root/stt/nemo_data/manifest_test.jsonl"))
    p.add_argument(
        "--open-train",
        type=Path,
        default=Path("/root/stt/v2_pipeline/manifests/open_train_norm_clean_v1.jsonl"),
    )
    p.add_argument(
        "--open-val",
        type=Path,
        default=Path("/root/stt/v2_pipeline/manifests/open_val_norm_clean_v1.jsonl"),
    )
    p.add_argument(
        "--open-test",
        type=Path,
        default=Path("/root/stt/v2_pipeline/manifests/open_test_norm_clean_v1.jsonl"),
    )
    p.add_argument(
        "--fleurs-test",
        type=Path,
        default=Path("/root/stt/v2_pipeline/manifests/test_fleurs_norm_clean_v1.jsonl"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("/root/stt/v2_pipeline/manifests"))
    p.add_argument("--report-path", type=Path, default=Path("/root/stt/v2_pipeline/reports/merge_v21_report.json"))
    p.add_argument("--private-train-repeat", type=int, default=2, help="Times to include private-train samples in train.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.private_train_repeat < 1:
        raise ValueError("--private-train-repeat must be >= 1")

    for path in [
        args.private_train,
        args.private_val,
        args.private_test,
        args.open_train,
        args.open_val,
        args.open_test,
        args.fleurs_test,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required input manifest: {path}")

    private_train, st_private_train = load_manifest(
        args.private_train, source_fn=lambda _: source_private("train"), default_source="private_v1_train", default_license="private-internal"
    )
    private_val, st_private_val = load_manifest(
        args.private_val, source_fn=lambda _: source_private("val"), default_source="private_v1_val", default_license="private-internal"
    )
    private_test, st_private_test = load_manifest(
        args.private_test, source_fn=lambda _: source_private("test"), default_source="private_v1_test", default_license="private-internal"
    )
    open_train, st_open_train = load_manifest(
        args.open_train, source_fn=source_open, default_source="open_train_unknown", default_license="open-unknown"
    )
    open_val, st_open_val = load_manifest(
        args.open_val, source_fn=source_open, default_source="open_val_unknown", default_license="open-unknown"
    )
    open_test, st_open_test = load_manifest(
        args.open_test, source_fn=source_open, default_source="open_test_unknown", default_license="open-unknown"
    )
    fleurs_test, st_fleurs_test = load_manifest(
        args.fleurs_test, source_fn=source_open, default_source="fleurs_test", default_license="cc-by-4.0"
    )

    # Build held-out test splits first; they are blocked from train/val.
    test_private, drops_test_private = filter_and_merge([private_test], blocked_paths=set())
    blocked_test_paths = {r["audio_filepath"] for r in test_private}

    test_open, drops_test_open = filter_and_merge([open_test], blocked_paths=blocked_test_paths)
    blocked_test_paths.update(r["audio_filepath"] for r in test_open)

    test_fleurs, drops_test_fleurs = filter_and_merge([fleurs_test], blocked_paths=blocked_test_paths)
    blocked_test_paths.update(r["audio_filepath"] for r in test_fleurs)

    # Validation should not overlap tests.
    val_records, drops_val = filter_and_merge([private_val, open_val], blocked_paths=blocked_test_paths)
    blocked_val_test_paths = blocked_test_paths | {r["audio_filepath"] for r in val_records}

    # Train should not overlap val/tests.
    private_train_filtered, drops_private_train = filter_and_merge([private_train], blocked_paths=blocked_val_test_paths)
    open_train_filtered, drops_open_train = filter_and_merge([open_train], blocked_paths=blocked_val_test_paths)

    train_records: list[dict] = []
    train_records.extend(private_train_filtered)
    train_records.extend(open_train_filtered)
    # Upsampling by repetition (default=2 means original + one extra copy).
    for _ in range(args.private_train_repeat - 1):
        train_records.extend(private_train_filtered)
    random.Random(args.seed).shuffle(train_records)

    out_paths = {
        "combined_train": args.out_dir / "combined_train_norm_v21.jsonl",
        "combined_val": args.out_dir / "combined_val_norm_v21.jsonl",
        "test_private": args.out_dir / "test_private_norm_v21.jsonl",
        "test_open": args.out_dir / "test_open_norm_v21.jsonl",
        "test_fleurs": args.out_dir / "test_fleurs_norm_v21.jsonl",
    }
    write_jsonl(train_records, out_paths["combined_train"])
    write_jsonl(val_records, out_paths["combined_val"])
    write_jsonl(test_private, out_paths["test_private"])
    write_jsonl(test_open, out_paths["test_open"])
    write_jsonl(test_fleurs, out_paths["test_fleurs"])

    report = {
        "params": {
            "min_duration_sec": MIN_DURATION,
            "max_duration_sec": MAX_DURATION,
            "private_train_repeat": args.private_train_repeat,
            "seed": args.seed,
        },
        "inputs": {
            "private_train": {"path": str(args.private_train), "stats": st_private_train.to_dict()},
            "private_val": {"path": str(args.private_val), "stats": st_private_val.to_dict()},
            "private_test": {"path": str(args.private_test), "stats": st_private_test.to_dict()},
            "open_train": {"path": str(args.open_train), "stats": st_open_train.to_dict()},
            "open_val": {"path": str(args.open_val), "stats": st_open_val.to_dict()},
            "open_test": {"path": str(args.open_test), "stats": st_open_test.to_dict()},
            "fleurs_test": {"path": str(args.fleurs_test), "stats": st_fleurs_test.to_dict()},
        },
        "overlap_drops": {
            "test_private": dict(drops_test_private),
            "test_open": dict(drops_test_open),
            "test_fleurs": dict(drops_test_fleurs),
            "val": dict(drops_val),
            "private_train": dict(drops_private_train),
            "open_train": dict(drops_open_train),
        },
        "outputs": {
            "combined_train": {"path": str(out_paths["combined_train"]), **summarize_split(train_records)},
            "combined_val": {"path": str(out_paths["combined_val"]), **summarize_split(val_records)},
            "test_private": {"path": str(out_paths["test_private"]), **summarize_split(test_private)},
            "test_open": {"path": str(out_paths["test_open"]), **summarize_split(test_open)},
            "test_fleurs": {"path": str(out_paths["test_fleurs"]), **summarize_split(test_fleurs)},
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({"outputs": report["outputs"], "report_path": str(args.report_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
