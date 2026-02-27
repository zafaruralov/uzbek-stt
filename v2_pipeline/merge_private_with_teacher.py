#!/usr/bin/env python3
"""Merge private v1 data with teacher-labeled data for v1.1 fine-tuning.

Outputs:
- combined_train_teacher_v11.jsonl
- combined_val_teacher_v11.jsonl
- test_private_v11.jsonl
- reports/merge_teacher_v11_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge private manifests with teacher manifest.")
    p.add_argument("--private-train", type=Path, default=Path("/root/stt/nemo_data/manifest_train.jsonl"))
    p.add_argument("--private-val", type=Path, default=Path("/root/stt/nemo_data/manifest_val.jsonl"))
    p.add_argument("--private-test", type=Path, default=Path("/root/stt/nemo_data/manifest_test.jsonl"))
    p.add_argument("--teacher-manifest", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("/root/stt/v2_pipeline/manifests"))
    p.add_argument(
        "--report-json",
        type=Path,
        default=Path("/root/stt/v2_pipeline/reports/merge_teacher_v11_report.json"),
    )
    p.add_argument("--teacher-val-ratio", type=float, default=0.05)
    p.add_argument("--private-train-repeat", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def hash_bucket(s: str) -> float:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    x = int(h[:8], 16)
    return x / 0xFFFFFFFF


def load_jsonl(path: Path, default_source: str, default_license: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ap = str(rec.get("audio_filepath", "")).strip()
            txt = str(rec.get("text", "")).strip()
            if not ap or not txt:
                continue
            d = float(rec.get("duration", 0.0) or 0.0)
            if d <= 0:
                continue
            rows.append(
                {
                    "audio_filepath": str(Path(ap).resolve()),
                    "text": txt,
                    "duration": d,
                    "source": str(rec.get("source", default_source)),
                    "license": str(rec.get("license", default_license)),
                }
            )
    return rows


def dedupe(records: list[dict], blocked: set[str] | None = None) -> tuple[list[dict], int]:
    blocked = blocked or set()
    seen = set(blocked)
    out = []
    dropped = 0
    for r in records:
        p = r["audio_filepath"]
        if p in seen:
            dropped += 1
            continue
        seen.add(p)
        out.append(r)
    return out, dropped


def summarize(name: str, rows: list[dict]) -> dict:
    hours = sum(float(x["duration"]) for x in rows) / 3600.0
    by_source = Counter(x["source"] for x in rows)
    return {
        "name": name,
        "samples": len(rows),
        "hours": round(hours, 3),
        "by_source": dict(sorted(by_source.items())),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    for p in [args.private_train, args.private_val, args.private_test, args.teacher_manifest]:
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")
    if not (0.0 <= args.teacher_val_ratio < 1.0):
        raise ValueError("--teacher-val-ratio must be in [0, 1).")
    if args.private_train_repeat < 1:
        raise ValueError("--private-train-repeat must be >= 1")

    p_train = load_jsonl(args.private_train, "private_v1_train", "private-internal")
    p_val = load_jsonl(args.private_val, "private_v1_val", "private-internal")
    p_test = load_jsonl(args.private_test, "private_v1_test", "private-internal")
    teacher = load_jsonl(args.teacher_manifest, "google_teacher", "user-provided")

    p_test, drop_test_dupe = dedupe(p_test)
    blocked = {r["audio_filepath"] for r in p_test}

    p_val, drop_val_dupe = dedupe(p_val, blocked=blocked)
    blocked.update(r["audio_filepath"] for r in p_val)

    p_train, drop_train_dupe = dedupe(p_train, blocked=blocked)
    blocked.update(r["audio_filepath"] for r in p_train)

    # Teacher split by stable hash.
    t_train: list[dict] = []
    t_val: list[dict] = []
    t_drop_overlap = 0
    for r in teacher:
        p = r["audio_filepath"]
        if p in blocked:
            t_drop_overlap += 1
            continue
        b = hash_bucket(p)
        if b < args.teacher_val_ratio:
            t_val.append(r)
        else:
            t_train.append(r)

    # Repeat private train to protect private domain strength.
    combined_train = list(p_train) + list(t_train)
    for _ in range(args.private_train_repeat - 1):
        combined_train.extend(p_train)
    random.Random(args.seed).shuffle(combined_train)

    combined_val = list(p_val) + list(t_val)
    random.Random(args.seed + 1).shuffle(combined_val)

    out_train = args.out_dir / "combined_train_teacher_v11.jsonl"
    out_val = args.out_dir / "combined_val_teacher_v11.jsonl"
    out_test = args.out_dir / "test_private_v11.jsonl"
    write_jsonl(out_train, combined_train)
    write_jsonl(out_val, combined_val)
    write_jsonl(out_test, p_test)

    report = {
        "inputs": {
            "private_train": str(args.private_train),
            "private_val": str(args.private_val),
            "private_test": str(args.private_test),
            "teacher_manifest": str(args.teacher_manifest),
        },
        "params": {
            "teacher_val_ratio": args.teacher_val_ratio,
            "private_train_repeat": args.private_train_repeat,
            "seed": args.seed,
        },
        "drops": {
            "private_test_dupe": drop_test_dupe,
            "private_val_dupe_or_overlap": drop_val_dupe,
            "private_train_dupe_or_overlap": drop_train_dupe,
            "teacher_overlap_with_private_splits": t_drop_overlap,
        },
        "outputs": {
            "combined_train_teacher_v11": {
                "path": str(out_train.resolve()),
                **summarize("combined_train_teacher_v11", combined_train),
            },
            "combined_val_teacher_v11": {
                "path": str(out_val.resolve()),
                **summarize("combined_val_teacher_v11", combined_val),
            },
            "test_private_v11": {
                "path": str(out_test.resolve()),
                **summarize("test_private_v11", p_test),
            },
            "teacher_split": {
                "teacher_train_samples": len(t_train),
                "teacher_val_samples": len(t_val),
            },
        },
    }
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

