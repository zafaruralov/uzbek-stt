#!/usr/bin/env python3
"""Align audiobook audio chunks to matching PDF text and build training manifest.

Pipeline:
1) Extract PDF text -> sentence list
2) Split audio to fixed chunks
3) Transcribe chunks with local NeMo model (v1)
4) Monotonic fuzzy alignment chunk transcript -> nearby PDF sentences
5) Write aligned manifest + rejects + report
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

import pdfplumber
import soundfile as sf
from rapidfuzz import fuzz

import nemo.collections.asr as nemo_asr
import torch


CYR_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo",
    "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "x", "ц": "s", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "",
    "ы": "i", "ь": "", "э": "e", "ю": "yu", "я": "ya",
    "қ": "q", "ғ": "g'", "ҳ": "h", "ў": "o'", "ѐ": "yo",
    "ʼ": "'", "ʻ": "'", "’": "'", "`": "'",
}


def uzbek_to_latin(text: str) -> str:
    out: list[str] = []
    for ch in text:
        low = ch.lower()
        if low in CYR_TO_LAT:
            repl = CYR_TO_LAT[low]
            out.append(repl)
        else:
            out.append(ch)
    return "".join(out)


def normalize_text(text: str) -> str:
    t = uzbek_to_latin(text).lower()
    t = t.replace("’", "'").replace("ʻ", "'").replace("ʼ", "'").replace("`", "'")
    t = re.sub(r"[^a-z0-9'\\s]", " ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\!\?])\s+", text)
    out = []
    for p in parts:
        s = p.strip()
        if len(s) < 5:
            continue
        out.append(s)
    return out


def extract_pdf_sentences(pdf_path: Path) -> list[str]:
    lines = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt:
                lines.append(txt)
    all_text = "\n".join(lines)
    # light cleanup for frequent artifacts
    all_text = all_text.replace("\u00ad", "")
    all_text = all_text.replace("ﬁ", "fi").replace("ﬂ", "fl")
    return split_sentences(all_text)


def run_ffmpeg_chunk(input_audio: Path, out_dir: Path, segment_sec: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "chunk_%05d.wav"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "segment",
        "-segment_time",
        str(segment_sec),
        "-c:a",
        "pcm_s16le",
        str(pattern),
    ]
    subprocess.check_call(cmd)


def load_transcript_cache(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ap = rec.get("audio_filepath")
            if isinstance(ap, str):
                out[ap] = rec
    return out


def append_cache(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def best_alignment(
    asr_norm: str,
    sentences: list[str],
    sentences_norm: list[str],
    cursor: int,
    lookback: int = 5,
    lookahead: int = 120,
    max_span: int = 8,
    min_len_ratio: float = 0.35,
    max_cursor_jump: int = 0,
) -> tuple[int, int, float, str]:
    j0 = max(0, cursor - lookback)
    j1 = min(len(sentences), cursor + lookahead)
    best_score = -1.0
    best_j = cursor
    best_span = 1
    best_text = ""

    for j in range(j0, j1):
        if max_cursor_jump > 0 and j > cursor and (j - cursor) > max_cursor_jump:
            continue
        for span in range(1, max_span + 1):
            k = j + span
            if k > len(sentences):
                continue
            cand_norm = " ".join(sentences_norm[j:k]).strip()
            if not cand_norm:
                continue
            a_len = len(asr_norm.split())
            c_len = len(cand_norm.split())
            if a_len == 0 or c_len == 0:
                continue
            len_ratio = min(a_len, c_len) / max(a_len, c_len)
            # Combined score: good overlap + preserve order + length sanity.
            s_set = float(fuzz.token_set_ratio(asr_norm, cand_norm))
            s_sort = float(fuzz.token_sort_ratio(asr_norm, cand_norm))
            s_raw = float(fuzz.ratio(asr_norm, cand_norm))
            score = 0.40 * s_set + 0.35 * s_sort + 0.25 * s_raw
            if len_ratio < min_len_ratio:
                score -= (min_len_ratio - len_ratio) * 45.0
            if score > best_score:
                best_score = score
                best_j = j
                best_span = span
                best_text = " ".join(sentences[j:k]).strip()
    return best_j, best_span, best_score, best_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Align audiobook + PDF into training manifest.")
    p.add_argument("--audio-path", type=Path, required=True)
    p.add_argument("--pdf-path", type=Path, required=True)
    p.add_argument(
        "--model-path",
        type=Path,
        default=Path("/root/stt/nemo_experiment/uzbek_fastconformer_finetune/final_model.nemo"),
    )
    p.add_argument("--output-dir", type=Path, default=Path("/root/stt/v2_pipeline/outputs"))
    p.add_argument("--book-id", type=str, default="book")
    p.add_argument("--segment-sec", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--align-threshold", type=float, default=56.0)
    p.add_argument("--min-asr-words", type=int, default=2)
    p.add_argument("--lookback", type=int, default=5)
    p.add_argument("--lookahead", type=int, default=120)
    p.add_argument("--max-span", type=int, default=8)
    p.add_argument("--min-len-ratio", type=float, default=0.35)
    p.add_argument(
        "--max-cursor-jump",
        type=int,
        default=0,
        help="Maximum forward sentence index jump per chunk (0 disables cap).",
    )
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max-chunks", type=int, default=0, help="0=all chunks")
    p.add_argument("--reuse-chunks", action="store_true")
    p.add_argument("--reuse-transcripts", action="store_true")
    p.add_argument("--source-label", type=str, default="pdf_aligned_user_book")
    p.add_argument("--license-label", type=str, default="user-provided")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_path.exists():
        raise FileNotFoundError(f"audio not found: {args.audio_path}")
    if not args.pdf_path.exists():
        raise FileNotFoundError(f"pdf not found: {args.pdf_path}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model not found: {args.model_path}")

    book_dir = args.output_dir / "aligned_chunks" / args.book_id
    report_dir = args.output_dir / "reports"
    manifest_dir = args.output_dir / "manifests"
    transcript_cache = report_dir / f"{args.book_id}_chunk_transcripts.jsonl"
    manifest_out = manifest_dir / f"{args.book_id}_aligned_manifest.jsonl"
    rejects_out = manifest_dir / f"{args.book_id}_aligned_rejects.jsonl"
    report_out = report_dir / f"{args.book_id}_alignment_report.json"
    for d in [book_dir, report_dir, manifest_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[align] extracting PDF sentences...")
    sentences = extract_pdf_sentences(args.pdf_path)
    if not sentences:
        raise RuntimeError("No sentences extracted from PDF.")
    sentences_norm = [normalize_text(s) for s in sentences]
    print(f"[align] pdf sentences: {len(sentences)}")

    # Chunk audio
    chunks = sorted(book_dir.glob("chunk_*.wav"))
    if not chunks or not args.reuse_chunks:
        print("[align] chunking audio with ffmpeg...")
        # clear previous chunks only if not reusing
        if not args.reuse_chunks and chunks:
            for c in chunks:
                c.unlink(missing_ok=True)
        run_ffmpeg_chunk(args.audio_path, book_dir, args.segment_sec)
        chunks = sorted(book_dir.glob("chunk_*.wav"))
    if args.max_chunks > 0:
        chunks = chunks[: args.max_chunks]
    if not chunks:
        raise RuntimeError("No audio chunks available.")
    print(f"[align] chunks to process: {len(chunks)}")

    # Load model lazily only if transcript cache misses occur.
    requested_device = args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    model = None
    model_device = requested_device

    cache = load_transcript_cache(transcript_cache) if args.reuse_transcripts else {}
    print(f"[align] cached transcripts: {len(cache)}")

    stats = Counter()
    aligned_rows: list[dict] = []
    rejected_rows: list[dict] = []
    cursor = 0

    for i, chunk in enumerate(chunks, start=1):
        ap = str(chunk.resolve())
        dur = float(sf.info(str(chunk)).duration)

        if ap in cache:
            asr_text = str(cache[ap].get("asr_text", "")).strip()
            stats["transcript_cache_hit"] += 1
        else:
            if model is None:
                print(f"[align] loading ASR model on {model_device} ...")
                model = nemo_asr.models.ASRModel.restore_from(str(args.model_path), map_location="cpu")
                model = model.to(model_device)
                model.eval()
            pred = model.transcribe([ap], batch_size=args.batch_size, verbose=False)[0]
            if not isinstance(pred, str):
                pred = getattr(pred, "text", str(pred))
            asr_text = str(pred).strip()
            append_cache(
                transcript_cache,
                {"audio_filepath": ap, "duration": dur, "asr_text": asr_text},
            )
            stats["transcript_new"] += 1

        asr_norm = normalize_text(asr_text)
        if len(asr_norm.split()) < args.min_asr_words:
            stats["reject_short_asr"] += 1
            rejected_rows.append(
                {
                    "audio_filepath": ap,
                    "duration": dur,
                    "asr_text": asr_text,
                    "reason": "short_asr",
                }
            )
            continue

        j, span, score, aligned_text = best_alignment(
            asr_norm,
            sentences,
            sentences_norm,
            cursor,
            lookback=args.lookback,
            lookahead=args.lookahead,
            max_span=args.max_span,
            min_len_ratio=args.min_len_ratio,
            max_cursor_jump=args.max_cursor_jump,
        )
        if score < args.align_threshold:
            stats["reject_low_align_score"] += 1
            rejected_rows.append(
                {
                    "audio_filepath": ap,
                    "duration": dur,
                    "asr_text": asr_text,
                    "align_score": round(score, 3),
                    "best_sentence_idx": j,
                    "best_span": span,
                    "best_candidate": aligned_text,
                    "reason": "low_align_score",
                }
            )
            continue

        row = {
            "audio_filepath": ap,
            "text": normalize_text(aligned_text),
            "duration": dur,
            "source": args.source_label,
            "license": args.license_label,
            "align_score": round(score, 3),
            "sentence_idx": j,
            "sentence_span": span,
            "asr_text": asr_text,
        }
        aligned_rows.append(row)
        cursor = max(cursor, j + span - 1)
        stats["aligned_ok"] += 1

        if i % 25 == 0:
            print(
                f"[align] {i}/{len(chunks)} processed | ok={stats['aligned_ok']} "
                f"reject={stats['reject_short_asr'] + stats['reject_low_align_score']} cursor={cursor}"
            )

    with manifest_out.open("w", encoding="utf-8") as f:
        for r in aligned_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with rejects_out.open("w", encoding="utf-8") as f:
        for r in rejected_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    hours_ok = sum(r["duration"] for r in aligned_rows) / 3600.0
    hours_rej = sum(r["duration"] for r in rejected_rows) / 3600.0
    report = {
        "book_id": args.book_id,
        "audio_path": str(args.audio_path.resolve()),
        "pdf_path": str(args.pdf_path.resolve()),
        "model_path": str(args.model_path.resolve()),
        "device": requested_device,
        "segment_sec": args.segment_sec,
        "align_threshold": args.align_threshold,
        "lookback": args.lookback,
        "lookahead": args.lookahead,
        "max_span": args.max_span,
        "min_len_ratio": args.min_len_ratio,
        "max_cursor_jump": args.max_cursor_jump,
        "chunks_total": len(chunks),
        "aligned_rows": len(aligned_rows),
        "rejected_rows": len(rejected_rows),
        "aligned_hours": round(hours_ok, 3),
        "rejected_hours": round(hours_rej, 3),
        "stats": dict(stats),
        "outputs": {
            "manifest": str(manifest_out.resolve()),
            "rejects": str(rejects_out.resolve()),
            "transcript_cache": str(transcript_cache.resolve()),
        },
    }
    report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
