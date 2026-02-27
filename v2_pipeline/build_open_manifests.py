#!/usr/bin/env python3
"""Build stable NeMo manifests from downloaded open Uzbek datasets.

CPU/disk only workflow:
- USC: uses extracted wav/txt pairs directly
- Common Voice Uzbek: exports audio arrays to stable WAV paths
- FLEURS uz_uz: exports test split to stable WAV paths

Writes both raw and normalized manifests incrementally.
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict

import numpy as np
import soundfile as sf
from datasets import load_dataset, load_from_disk

ROOT = Path('/root/stt/v2_pipeline')
DATA = ROOT / 'data'
AUDIO_OUT = ROOT / 'audio'
MANIFESTS = ROOT / 'manifests'
REPORTS = ROOT / 'reports'

USC_ROOT = DATA / 'usc_extracted' / 'ISSAI_USC'
CV_PARQUET = str(DATA / 'common_voice_uzbek_snapshot' / 'data' / '*.parquet')
FLEURS_DIR = DATA / 'fleurs_uz_uz'

SR_TARGET = 16000


def normalize_text(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('’', "'").replace('`', "'").replace('ʻ', "'").replace('ʼ', "'")
    text = text.replace('“', '"').replace('”', '"')
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
    return text


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_mono(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)


def resample_linear(arr: np.ndarray, src_sr: int, tgt_sr: int = SR_TARGET) -> np.ndarray:
    if src_sr == tgt_sr:
        return arr
    ratio = tgt_sr / src_sr
    new_len = max(1, int(round(len(arr) * ratio)))
    x_old = np.arange(len(arr), dtype=np.float64)
    x_new = np.linspace(0, len(arr) - 1, new_len, dtype=np.float64)
    out = np.interp(x_new, x_old, arr.astype(np.float64)).astype(np.float32)
    return out


def export_audio_dict(audio: Dict, out_wav: Path) -> float:
    if out_wav.exists():
        return float(sf.info(str(out_wav)).duration)

    arr = np.asarray(audio['array'], dtype=np.float32)
    sr = int(audio['sampling_rate'])
    arr = to_mono(arr)
    arr = resample_linear(arr, sr, SR_TARGET)

    ensure_dir(out_wav.parent)
    sf.write(str(out_wav), arr, SR_TARGET, subtype='PCM_16')
    return float(len(arr) / SR_TARGET)


def make_record(audio_filepath: str, text: str, duration: float, source: str, license_name: str, **extra) -> Dict:
    r = {
        'audio_filepath': audio_filepath,
        'text': text,
        'duration': float(duration),
        'source': source,
        'license': license_name,
    }
    r.update(extra)
    return r


def write_record(files, split: str, rec: Dict) -> None:
    files[split]['raw'].write(json.dumps(rec, ensure_ascii=False) + '\n')
    norm = dict(rec)
    norm['text'] = normalize_text(norm['text'])
    files[split]['norm'].write(json.dumps(norm, ensure_ascii=False) + '\n')


def open_manifest_files():
    ensure_dir(MANIFESTS)
    paths = {
        'train': {
            'raw': MANIFESTS / 'open_train_raw.jsonl',
            'norm': MANIFESTS / 'open_train_norm.jsonl',
        },
        'val': {
            'raw': MANIFESTS / 'open_val_raw.jsonl',
            'norm': MANIFESTS / 'open_val_norm.jsonl',
        },
        'test_open': {
            'raw': MANIFESTS / 'open_test_raw.jsonl',
            'norm': MANIFESTS / 'open_test_norm.jsonl',
        },
        'test_fleurs': {
            'raw': MANIFESTS / 'test_fleurs_raw.jsonl',
            'norm': MANIFESTS / 'test_fleurs_norm.jsonl',
        },
    }
    # overwrite on each run for deterministic output
    files = {}
    for key, p in paths.items():
        files[key] = {
            'raw': p['raw'].open('w', encoding='utf-8'),
            'norm': p['norm'].open('w', encoding='utf-8'),
        }
    return files, paths


def close_manifest_files(files):
    for group in files.values():
        group['raw'].close()
        group['norm'].close()


def split_cv(idx: int) -> str:
    return 'val' if (idx % 100) < 5 else 'train'


def process_usc(files, counts):
    print('[USC] processing extracted wav/txt pairs...')
    split_map = {'train': 'train', 'dev': 'val', 'test': 'test_open'}
    for usc_split, out_split in split_map.items():
        wavs = sorted((USC_ROOT / usc_split).glob('*.wav'))
        print(f'[USC] {usc_split}: {len(wavs)} wav files')
        for i, wav in enumerate(wavs, 1):
            txt = wav.with_suffix('.txt')
            if not txt.exists():
                continue
            text = txt.read_text(encoding='utf-8', errors='ignore').strip()
            if not text:
                continue
            dur = float(sf.info(str(wav)).duration)
            rec = make_record(
                audio_filepath=str(wav.resolve()),
                text=text,
                duration=dur,
                source=f'usc_{usc_split}',
                license_name='mit',
            )
            write_record(files, out_split, rec)
            counts[out_split] += 1
            if i % 5000 == 0:
                print(f'[USC] {usc_split}: {i}/{len(wavs)} processed')


def process_cv(files, counts):
    print('[CV] loading parquet snapshot...')
    ds = load_dataset('parquet', data_files={'train': CV_PARQUET})['train']
    total = len(ds)
    print(f'[CV] rows: {total}')
    base = AUDIO_OUT / 'common_voice_uzbek'

    for i, row in enumerate(ds):
        text = (row.get('normalized_text') or row.get('sentence') or '').strip()
        if not text:
            continue
        out_wav = base / f'cv_{i:06d}.wav'
        dur = export_audio_dict(row['audio'], out_wav)
        rec = make_record(
            audio_filepath=str(out_wav.resolve()),
            text=text,
            duration=dur,
            source='common_voice_uzbek',
            license_name='cc0-1.0',
            cv_idx=i,
        )
        out_split = split_cv(i)
        write_record(files, out_split, rec)
        counts[out_split] += 1
        if (i + 1) % 2000 == 0:
            print(f'[CV] {i+1}/{total} processed')


def process_fleurs_test(files, counts):
    print('[FLEURS] loading uz_uz from disk...')
    ds = load_from_disk(str(FLEURS_DIR))['test']
    total = len(ds)
    print(f'[FLEURS] test rows: {total}')
    base = AUDIO_OUT / 'fleurs_uz_uz' / 'test'
    for i, row in enumerate(ds):
        text = (row.get('transcription') or '').strip()
        if not text:
            continue
        out_wav = base / f'fleurs_test_{i:05d}.wav'
        dur = export_audio_dict(row['audio'], out_wav)
        rec = make_record(
            audio_filepath=str(out_wav.resolve()),
            text=text,
            duration=dur,
            source='fleurs_test',
            license_name='cc-by-4.0',
            fleurs_id=row.get('id'),
        )
        write_record(files, 'test_fleurs', rec)
        counts['test_fleurs'] += 1
        if (i + 1) % 200 == 0:
            print(f'[FLEURS] {i+1}/{total} processed')


def main() -> None:
    ensure_dir(AUDIO_OUT)
    ensure_dir(MANIFESTS)
    ensure_dir(REPORTS)

    files, paths = open_manifest_files()
    counts = {'train': 0, 'val': 0, 'test_open': 0, 'test_fleurs': 0}

    try:
        process_usc(files, counts)
        process_cv(files, counts)
        process_fleurs_test(files, counts)
    finally:
        close_manifest_files(files)

    report = {
        'counts': counts,
        'manifests': {k: {kk: str(vv) for kk, vv in p.items()} for k, p in paths.items()},
        'audio_export_dir': str(AUDIO_OUT.resolve()),
    }
    (REPORTS / 'manifest_build_report.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
