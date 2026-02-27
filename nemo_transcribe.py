"""
NeMo Parakeet-TDT 0.6B v3 yordamida audio fayllarni transcribe qilish
va NeMo manifest formatida saqlash (fine-tuning uchun).

DIQQAT: Parakeet-TDT 0.6B v3 modeli o'zbek tilini qo'llab-quvvatlamaydi.
Natijalar noisy pseudo-label sifatida ishlatiladi.
Fine-tuning uchun qo'lda tekshirish va tuzatish talab etiladi.

Foydalanish:
    python nemo_transcribe.py /path/to/audio/ -o /path/to/output/ --srt --txt
    python nemo_transcribe.py audio.mp3 --device cuda:1
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("nemo_transcribe")

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".opus", ".wma", ".aac"}
SAMPLE_RATE = 16000
DEFAULT_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
MAX_DURATION_FULL_ATTN = 1440   # 24 min (full attention limit)
MAX_DURATION_LOCAL_ATTN = 10800  # 3 hours (local attention limit)


def resolve_model_reference(model_ref: str) -> str:
    """Use local HuggingFace cache .nemo when available to avoid network dependency."""
    p = Path(model_ref).expanduser()
    if p.is_file():
        return str(p.resolve())

    if "/" not in model_ref:
        return model_ref

    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    snapshots_dir = hf_home / "hub" / f"models--{model_ref.replace('/', '--')}" / "snapshots"
    if not snapshots_dir.exists():
        return model_ref

    try:
        snapshots = sorted(
            [d for d in snapshots_dir.iterdir() if d.is_dir()],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return model_ref

    for snap in snapshots:
        for nemo_file in sorted(snap.glob("*.nemo")):
            if nemo_file.exists():
                return str(nemo_file.resolve())
    return model_ref


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TranscriptionResult:
    audio_filepath: str
    text: str = ""
    duration: float = 0.0
    word_timestamps: List[dict] = field(default_factory=list)
    segment_timestamps: List[dict] = field(default_factory=list)
    original_filepath: str = ""
    success: bool = True
    error: Optional[str] = None


def _result_to_payload(result: TranscriptionResult) -> Dict[str, Any]:
    """Serialize TranscriptionResult for worker stdout JSON."""
    return {
        "audio_filepath": result.audio_filepath,
        "text": result.text,
        "duration": result.duration,
        "segment_timestamps": result.segment_timestamps,
        "success": result.success,
        "error": result.error,
    }


def _payload_to_result(payload: Dict[str, Any], fallback_audio: str) -> TranscriptionResult:
    """Deserialize worker JSON payload into TranscriptionResult."""
    return TranscriptionResult(
        audio_filepath=payload.get("audio_filepath") or fallback_audio,
        text=payload.get("text", "") or "",
        duration=float(payload.get("duration", 0.0) or 0.0),
        segment_timestamps=payload.get("segment_timestamps") or [],
        success=bool(payload.get("success", False)),
        error=payload.get("error"),
    )


# ---------------------------------------------------------------------------
# Audio preprocessing (patterns from stt_service_new.py)
# ---------------------------------------------------------------------------

class AudioPreprocessor:
    """Audio fayllarni 16 kHz mono WAV ga o'girish."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate

    # -- internal helpers ---------------------------------------------------

    def _load_audio_pyav(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """PyAV orqali audio yuklash (MP3, WAV, FLAC, OGG, ...)."""
        import av

        container = av.open(audio_path)
        stream = container.streams.audio[0]
        orig_sr = stream.rate

        frames = []
        for packet in container.demux(audio=0):
            try:
                for frame in packet.decode():
                    frames.append(frame.to_ndarray())
            except av.error.InvalidDataError:
                continue  # buzilgan frame'larni o'tkazib yuborish

        container.close()

        if not frames:
            raise RuntimeError(f"Audio fayldan ma'lumot o'qib bo'lmadi: {audio_path}")

        # shape = (channels, samples)
        audio = np.concatenate(frames, axis=1)

        # Stereo -> Mono
        if audio.shape[0] > 1:
            audio = audio.mean(axis=0)
        else:
            audio = audio[0]

        # float32 ga o'tkazish
        if audio.dtype != np.float32:
            if np.issubdtype(audio.dtype, np.integer):
                max_val = np.iinfo(audio.dtype).max
                audio = audio.astype(np.float32) / max_val
            else:
                audio = audio.astype(np.float32)

        return audio, orig_sr

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        from scipy.signal import resample

        num_samples = int(len(audio) * target_sr / orig_sr)
        return resample(audio, num_samples)

    def _save_wav(self, audio: np.ndarray, path: str, sample_rate: int) -> None:
        from scipy.io import wavfile

        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wavfile.write(path, sample_rate, audio_int16)

    # -- public API ---------------------------------------------------------

    def load_and_convert(
        self, input_path: str, output_dir: str, overwrite: bool = False
    ) -> Tuple[Optional[str], float]:
        """Audio faylni 16 kHz mono WAV ga o'girib saqlash.

        Returns (wav_path, duration_seconds) or (None, 0.0) on failure.
        """
        inp = Path(input_path)
        wav_path = str(Path(output_dir) / (inp.stem + ".wav"))

        if not overwrite and Path(wav_path).exists():
            duration = NeMoTranscriber._get_duration_fast(wav_path)
            logger.info(f"  Mavjud WAV ishlatilmoqda: {wav_path} ({duration:.1f}s)")
            return wav_path, duration

        try:
            audio, orig_sr = self._load_audio_pyav(input_path)

            if orig_sr != self.sample_rate:
                logger.debug(f"  Resampling: {orig_sr}Hz -> {self.sample_rate}Hz")
                audio = self._resample(audio, orig_sr, self.sample_rate)

            self._save_wav(audio, wav_path, self.sample_rate)
            duration = len(audio) / self.sample_rate
            return wav_path, duration

        except Exception as e:
            logger.error(f"  Audio o'girishda xatolik ({inp.name}): {e}")
            return None, 0.0


# ---------------------------------------------------------------------------
# NeMo transcription
# ---------------------------------------------------------------------------

class NeMoTranscriber:
    """Parakeet-TDT 0.6B v3 model bilan transcribe."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        use_local_attention: bool = False,
    ):
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            logger.error(
                "NeMo topilmadi! O'rnatish uchun:\n"
                "  pip install 'nemo_toolkit[asr]'"
            )
            sys.exit(1)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Qurilma: {self.device}")

        model_source = resolve_model_reference(model_name)
        if model_source != model_name:
            logger.info(f"Lokal model cache ishlatilmoqda: {model_source}")

        logger.info(f"Model yuklanmoqda: {model_source} ...")
        if Path(model_source).is_file():
            self.model = nemo_asr.models.ASRModel.restore_from(restore_path=model_source)
        else:
            self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_source)

        if "cuda" in self.device:
            self.model = self.model.to(self.device)

        if use_local_attention:
            logger.info("Local attention yoqilmoqda (3 soatgacha audio uchun)...")
            self.model.change_attention_model(
                self_attention_model="rel_pos_local_attn",
                att_context_size=[256, 256],
            )

        self.model.eval()
        self.use_local_attention = use_local_attention
        logger.info("Model tayyor!")

    @staticmethod
    def _get_duration_fast(path: str) -> float:
        """Get WAV duration from file size without loading into memory."""
        import struct
        with open(path, "rb") as f:
            f.read(4)  # RIFF
            f.read(4)  # file size
            f.read(4)  # WAVE
            # Find 'fmt ' chunk
            while True:
                chunk_id = f.read(4)
                if len(chunk_id) < 4:
                    break
                chunk_size = struct.unpack("<I", f.read(4))[0]
                if chunk_id == b"fmt ":
                    fmt_data = f.read(chunk_size)
                    channels = struct.unpack("<H", fmt_data[2:4])[0]
                    sample_rate = struct.unpack("<I", fmt_data[4:8])[0]
                    bits_per_sample = struct.unpack("<H", fmt_data[14:16])[0]
                    continue
                elif chunk_id == b"data":
                    data_size = chunk_size
                    bytes_per_sample = bits_per_sample // 8
                    return data_size / (sample_rate * channels * bytes_per_sample)
                else:
                    f.seek(chunk_size, 1)
        # Fallback
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        return len(data) / sr

    def transcribe_files(
        self,
        wav_paths: List[str],
        batch_size: int = 8,
        timestamps: bool = True,
        group_size: int = 10,
    ) -> List[TranscriptionResult]:
        """WAV fayllar ro'yxatini transcribe qilish (RAM tejash uchun gruppalar bilan)."""
        import gc

        results: List[TranscriptionResult] = []

        # Process in groups to avoid RAM OOM
        for g_start in range(0, len(wav_paths), group_size):
            g_end = min(g_start + group_size, len(wav_paths))
            group = wav_paths[g_start:g_end]
            logger.info(f"Gruppa {g_start // group_size + 1}: fayllar {g_start + 1}-{g_end}/{len(wav_paths)}")

            try:
                outputs = self.model.transcribe(
                    group, batch_size=batch_size, timestamps=timestamps
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
                logger.warning(f"Gruppa xatolik ({e}), birma-bir qayta urinish...")
                outputs = None

            if outputs is not None:
                for i, hyp in enumerate(outputs):
                    dur = self._get_duration_fast(group[i])
                    r = TranscriptionResult(
                        audio_filepath=group[i],
                        text=hyp.text if hasattr(hyp, "text") else str(hyp),
                        duration=dur,
                    )
                    if timestamps and hasattr(hyp, "timestamp") and hyp.timestamp:
                        r.word_timestamps = hyp.timestamp.get("word", [])
                        r.segment_timestamps = hyp.timestamp.get("segment", [])
                    results.append(r)
            else:
                # Fallback: one-by-one for this group
                for i, wp in enumerate(group):
                    logger.info(f"  [{g_start + i + 1}/{len(wav_paths)}] {Path(wp).name}")
                    try:
                        out = self.model.transcribe([wp], batch_size=1, timestamps=timestamps)
                        hyp = out[0]
                        dur = self._get_duration_fast(wp)
                        r = TranscriptionResult(
                            audio_filepath=wp,
                            text=hyp.text if hasattr(hyp, "text") else str(hyp),
                            duration=dur,
                        )
                        if timestamps and hasattr(hyp, "timestamp") and hyp.timestamp:
                            r.word_timestamps = hyp.timestamp.get("word", [])
                            r.segment_timestamps = hyp.timestamp.get("segment", [])
                        results.append(r)
                    except Exception as e:
                        logger.error(f"  Xatolik ({Path(wp).name}): {e}")
                        results.append(
                            TranscriptionResult(
                                audio_filepath=wp, success=False, error=str(e)
                            )
                        )

            # Free memory between groups
            gc.collect()
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        return results


# ---------------------------------------------------------------------------
# Subprocess isolation mode (avoids poisoned CUDA state)
# ---------------------------------------------------------------------------

def _parse_worker_payload(stdout: str) -> Optional[Dict[str, Any]]:
    """Worker prints one JSON object to stdout; parse robustly from the tail."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def run_single_file_worker(args: argparse.Namespace) -> int:
    """Worker mode: transcribe exactly one WAV, print JSON payload, exit."""
    wav_path = str(Path(args.input).resolve())
    result = TranscriptionResult(audio_filepath=wav_path, success=False)

    try:
        if not Path(wav_path).exists():
            result.error = f"Topilmadi: {wav_path}"
        else:
            transcriber = NeMoTranscriber(
                model_name=args.model,
                device=args.device,
                use_local_attention=args.local_attention,
            )
            out = transcriber.transcribe_files(
                wav_paths=[wav_path],
                batch_size=1,
                timestamps=not args.no_timestamps,
                group_size=1,
            )
            if out:
                result = out[0]
            else:
                result.error = "Worker bo'sh natija qaytardi"
    except Exception as e:
        result.error = str(e)
        result.success = False

    print(json.dumps(_result_to_payload(result), ensure_ascii=False), flush=True)
    return 0 if (result.success and result.text.strip()) else 1


def transcribe_files_subprocess(
    wav_paths: List[str],
    original_paths: List[str],
    args: argparse.Namespace,
    manifest_path: str,
    writer: "ManifestWriter",
) -> Tuple[List[TranscriptionResult], int]:
    """Transcribe each file in isolated subprocess and append outputs incrementally."""
    manifest = Path(manifest_path)
    failures_path = manifest.parent / "failures.jsonl"
    script_path = str(Path(__file__).resolve())

    if args.no_resume:
        manifest.unlink(missing_ok=True)
        failures_path.unlink(missing_ok=True)
        logger.info("Resume o'chirilgan: mavjud manifest/failures fayllar tozalandi.")

    processed = set()
    if not args.no_resume:
        processed = ManifestWriter.load_processed_audio_paths(manifest_path)
        if processed:
            logger.info(f"Resume topildi: {len(processed)} ta fayl allaqachon yozilgan, skip qilinadi.")

    results: List[TranscriptionResult] = []
    skipped_existing = 0

    txt_dir = str(manifest.parent / "texts")
    srt_dir = str(manifest.parent / "srt")

    total = len(wav_paths)
    for idx, wav_path in enumerate(wav_paths, 1):
        wav_abs = str(Path(wav_path).resolve())

        if wav_abs in processed:
            skipped_existing += 1
            logger.info(f"[{idx}/{total}] Skip (already done): {Path(wav_path).name}")
            continue

        logger.info(f"[{idx}/{total}] Subprocess transcribe: {Path(wav_path).name}")

        cmd = [
            sys.executable,
            script_path,
            wav_abs,
            "-m", args.model,
            "-b", "1",
            "--worker-single-file",
        ]
        if args.device:
            cmd.extend(["--device", args.device])
        if args.local_attention:
            cmd.append("--local-attention")
        if args.no_timestamps:
            cmd.append("--no-timestamps")

        proc = subprocess.run(cmd, capture_output=True, text=True)
        payload = _parse_worker_payload(proc.stdout)

        if payload is None:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
            err = stderr_tail or f"Worker xatolik kodi: {proc.returncode}"
            result = TranscriptionResult(
                audio_filepath=wav_abs,
                success=False,
                error=err,
            )
        else:
            result = _payload_to_result(payload, wav_abs)
            if not result.error and proc.returncode != 0 and not (result.success and result.text.strip()):
                result.error = f"Worker xatolik kodi: {proc.returncode}"

        if idx <= len(original_paths):
            result.original_filepath = original_paths[idx - 1]

        writer.append_result(result, manifest_path)
        if args.txt:
            writer.write_text_files([result], txt_dir)
        if args.srt:
            writer.write_srt_files([result], srt_dir)

        results.append(result)

    return results, skipped_existing


# ---------------------------------------------------------------------------
# Manifest and output writing
# ---------------------------------------------------------------------------

class ManifestWriter:
    """NeMo manifest va qo'shimcha fayllarni yozish."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # -- NeMo JSONL manifest -----------------------------------------------

    def write_manifest(self, results: List[TranscriptionResult], manifest_path: str) -> None:
        manifest = Path(manifest_path)
        manifest_dir = manifest.parent
        success_count = 0
        failure_count = 0
        failures_path = manifest_dir / "failures.jsonl"

        with open(manifest, "w", encoding="utf-8") as mf, \
             open(failures_path, "w", encoding="utf-8") as ff:
            for r in results:
                if r.success and r.text.strip():
                    # audio_filepath relative to manifest directory
                    try:
                        rel = os.path.relpath(r.audio_filepath, str(manifest_dir))
                    except ValueError:
                        rel = r.audio_filepath
                    entry = {
                        "audio_filepath": rel,
                        "text": r.text.strip(),
                        "duration": round(r.duration, 3),
                    }
                    mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    success_count += 1
                else:
                    entry = {
                        "audio_filepath": r.audio_filepath,
                        "error": r.error or "empty transcription",
                    }
                    ff.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    failure_count += 1

        logger.info(f"Manifest: {success_count} ta yozuv -> {manifest}")
        if failure_count:
            logger.warning(f"Xatoliklar: {failure_count} ta -> {failures_path}")
        else:
            failures_path.unlink(missing_ok=True)

    @staticmethod
    def _resolve_audio_path(raw_path: str, base_dir: Path) -> str:
        p = Path(raw_path)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        return str(p)

    @staticmethod
    def load_processed_audio_paths(manifest_path: str) -> Set[str]:
        """Load already processed file paths from manifest + failures (for resume)."""
        manifest = Path(manifest_path)
        manifest_dir = manifest.parent
        failures_path = manifest_dir / "failures.jsonl"
        processed: Set[str] = set()

        for src in (manifest, failures_path):
            if not src.exists():
                continue
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    audio = row.get("audio_filepath")
                    if not audio:
                        continue
                    processed.add(ManifestWriter._resolve_audio_path(audio, manifest_dir))
        return processed

    @staticmethod
    def count_jsonl_rows(path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def append_result(self, result: TranscriptionResult, manifest_path: str) -> None:
        """Append a single result to manifest/failures incrementally."""
        manifest = Path(manifest_path)
        manifest_dir = manifest.parent
        manifest_dir.mkdir(parents=True, exist_ok=True)
        failures_path = manifest_dir / "failures.jsonl"

        if result.success and result.text.strip():
            try:
                rel = os.path.relpath(result.audio_filepath, str(manifest_dir))
            except ValueError:
                rel = result.audio_filepath
            entry = {
                "audio_filepath": rel,
                "text": result.text.strip(),
                "duration": round(result.duration, 3),
            }
            with open(manifest, "a", encoding="utf-8") as mf:
                mf.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            entry = {
                "audio_filepath": result.audio_filepath,
                "error": result.error or "empty transcription",
            }
            with open(failures_path, "a", encoding="utf-8") as ff:
                ff.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # -- Per-file text -----------------------------------------------------

    def write_text_files(self, results: List[TranscriptionResult], output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            if r.success and r.text.strip():
                stem = Path(r.audio_filepath).stem
                (out / f"{stem}.txt").write_text(r.text.strip(), encoding="utf-8")

    # -- SRT subtitles -----------------------------------------------------

    @staticmethod
    def _seconds_to_srt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    def write_srt_files(self, results: List[TranscriptionResult], output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for r in results:
            if not r.success or not r.segment_timestamps:
                continue
            stem = Path(r.audio_filepath).stem
            lines = []
            for idx, seg in enumerate(r.segment_timestamps, 1):
                start = self._seconds_to_srt(seg.get("start", 0))
                end = self._seconds_to_srt(seg.get("end", 0))
                text = seg.get("segment", seg.get("text", ""))
                lines.append(f"{idx}\n{start} --> {end}\n{text}\n")
            (out / f"{stem}.srt").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def discover_audio_files(input_dir: str, recursive: bool = False) -> List[Path]:
    d = Path(input_dir)
    glob_fn = d.rglob if recursive else d.glob
    files = sorted(
        f for f in glob_fn("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if files:
        ext_counts = {}
        for f in files:
            ext = f.suffix.lower()
            ext_counts[ext] = ext_counts.get(ext, 0) + 1
        logger.info(f"Topilgan fayllar: {', '.join(f'{e}: {c}' for e, c in ext_counts.items())}")
    return files


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "NVIDIA Parakeet-TDT 0.6B v3 yordamida audio fayllarni transcribe qilish "
            "va NeMo manifest formatida saqlash (fine-tuning uchun).\n\n"
            "DIQQAT: O'zbek tili natively qo'llab-quvvatlanmaydi. "
            "Natijalar pseudo-label sifatida — qo'lda tekshirish kerak."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "input", help="Audio fayllar papkasi yoki bitta audio fayl"
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Natijalar papkasi (default: <input>/nemo_output/)",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL,
        help=f"NeMo model nomi yoki path (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-b", "--batch-size", type=int, default=8,
        help="Batch size (default: 8). Kattaroq = tezroq, lekin ko'proq VRAM",
    )
    parser.add_argument(
        "--no-timestamps", action="store_true",
        help="Timestamp'larni o'chirish (tezroq, lekin SRT chiqmaydi)",
    )
    parser.add_argument(
        "--local-attention", action="store_true",
        help="Local attention rejimi (3 soatgacha uzun audio uchun)",
    )
    parser.add_argument(
        "--subprocess-per-file", action="store_true",
        help="Har bir faylni alohida subprocess'da transcribe qilish (eng barqaror, sekinroq)",
    )
    parser.add_argument(
        "--force-inprocess", action="store_true",
        help="Local attention bo'lsa ham eski in-process gruppa rejimini majburlash",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Subprocess rejimida mavjud manifest/failures dagi progressni e'tiborsiz qoldirish",
    )
    parser.add_argument("--srt", action="store_true", help="SRT subtitle fayllar yaratish")
    parser.add_argument("--txt", action="store_true", help="Har bir fayl uchun .txt yaratish")
    parser.add_argument(
        "--recursive", action="store_true",
        help="Papka ichidagi barcha sub-papkalarni ham qidirish",
    )
    parser.add_argument(
        "--manifest-name", default="manifest.jsonl",
        help="Manifest fayl nomi (default: manifest.jsonl)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Qurilma: 'cuda', 'cuda:0', 'cuda:1', 'cpu'. Avtomatik tanlanadi",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Mavjud WAV fayllarni qayta yozish",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Batafsil log",
    )
    parser.add_argument(
        "--worker-single-file", action="store_true", help=argparse.SUPPRESS,
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_argparser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    resolved_model = resolve_model_reference(args.model)
    if resolved_model != args.model:
        logger.info(f"Model cache topildi, onlayn yuklash o'rniga lokal model ishlatiladi: {resolved_model}")
        args.model = resolved_model

    if args.worker_single_file:
        sys.exit(run_single_file_worker(args))

    # -- Resolve paths ------------------------------------------------------
    input_path = Path(args.input).resolve()
    if input_path.is_file():
        audio_files = [input_path]
        default_output = input_path.parent / "nemo_output"
    elif input_path.is_dir():
        audio_files = discover_audio_files(str(input_path), recursive=args.recursive)
        default_output = input_path / "nemo_output"
    else:
        logger.error(f"Topilmadi: {input_path}")
        sys.exit(1)

    if not audio_files:
        logger.error("Audio fayllar topilmadi!")
        sys.exit(1)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Audio fayllar: {len(audio_files)}")
    logger.info(f"Natijalar papkasi: {output_dir}")
    logger.info(f"Model: {args.model}")

    logger.warning(
        "DIQQAT: Parakeet-TDT 0.6B v3 o'zbek tilini natively qo'llab-quvvatlamaydi. "
        "Natijalar pseudo-label — fine-tuning uchun qo'lda tekshiring!"
    )

    # -- Step 1: Audio preprocessing ----------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Audio fayllarni 16 kHz mono WAV ga o'girish...")
    logger.info("=" * 60)

    preprocessor = AudioPreprocessor(sample_rate=SAMPLE_RATE)
    wav_paths: List[str] = []
    original_paths: List[str] = []
    durations: List[float] = []
    skipped: List[str] = []

    for i, audio_file in enumerate(audio_files, 1):
        logger.info(f"[{i}/{len(audio_files)}] {audio_file.name}")
        wav_path, duration = preprocessor.load_and_convert(
            str(audio_file), str(wav_dir), overwrite=args.overwrite
        )
        if wav_path:
            wav_paths.append(wav_path)
            original_paths.append(str(audio_file))
            durations.append(duration)
            logger.info(f"  -> {Path(wav_path).name} ({duration:.1f}s)")
        else:
            skipped.append(str(audio_file))

    logger.info(f"O'girildi: {len(wav_paths)}, O'tkazib yuborildi: {len(skipped)}")

    if not wav_paths:
        logger.error("Hech bir fayl muvaffaqiyatli o'girilmadi!")
        sys.exit(1)

    # -- Check long files ---------------------------------------------------
    max_dur = MAX_DURATION_LOCAL_ATTN if args.local_attention else MAX_DURATION_FULL_ATTN
    long_indices = [i for i, d in enumerate(durations) if d > max_dur]
    if long_indices:
        for idx in long_indices:
            logger.warning(
                f"  {Path(wav_paths[idx]).name} ({durations[idx]:.0f}s) "
                f"> {max_dur / 60:.0f} min limit — o'tkazib yuboriladi"
            )
        if not args.local_attention:
            logger.warning("--local-attention flag bilan 3 soatgacha audio ishlaydi.")
        valid = [i for i in range(len(wav_paths)) if i not in long_indices]
        wav_paths = [wav_paths[i] for i in valid]
        original_paths = [original_paths[i] for i in valid]
        durations = [durations[i] for i in valid]

    if not wav_paths:
        logger.error("Barcha fayllar juda uzun!")
        sys.exit(1)

    writer = ManifestWriter(str(output_dir))
    manifest_path = str(output_dir / args.manifest_name)
    failures_path = Path(manifest_path).parent / "failures.jsonl"

    # -- Step 2: Transcribe -------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: NeMo model yuklanmoqda va transcribe qilinmoqda...")
    logger.info("=" * 60)

    use_subprocess_mode = args.subprocess_per_file or (
        args.local_attention and len(wav_paths) > 1 and not args.force_inprocess
    )

    t0 = time.time()
    skipped_existing = 0

    if use_subprocess_mode:
        logger.warning(
            "Subprocess isolation rejimi yoqildi: har bir fayl alohida process'da "
            "transcribe qilinadi (sekinroq, lekin CUDA uchun ancha barqaror)."
        )
        results, skipped_existing = transcribe_files_subprocess(
            wav_paths=wav_paths,
            original_paths=original_paths,
            args=args,
            manifest_path=manifest_path,
            writer=writer,
        )
    else:
        transcriber = NeMoTranscriber(
            model_name=args.model,
            device=args.device,
            use_local_attention=args.local_attention,
        )
        results = transcriber.transcribe_files(
            wav_paths=wav_paths,
            batch_size=args.batch_size,
            timestamps=not args.no_timestamps,
        )
        for i, r in enumerate(results):
            if i < len(original_paths):
                r.original_filepath = original_paths[i]

    elapsed = time.time() - t0

    successful = [r for r in results if r.success and r.text.strip()]
    failed = [r for r in results if not (r.success and r.text.strip())]
    total_audio = sum(r.duration for r in successful)

    logger.info(f"Transcribe vaqti: {elapsed:.1f}s")
    logger.info(
        f"Joriy run -> Muvaffaqiyatli: {len(successful)}, "
        f"Xatolik: {len(failed)}, Resume skip: {skipped_existing}"
    )
    if elapsed > 0:
        logger.info(f"Jami audio: {total_audio:.1f}s, RTFx: {total_audio / elapsed:.1f}x")

    # -- Step 3: Write outputs -----------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Natijalar saqlanmoqda...")
    logger.info("=" * 60)

    if use_subprocess_mode:
        logger.info("Subprocess rejimi: manifest/text/srt har fayldan keyin incremental yozildi.")
        if args.txt:
            logger.info(f"Text fayllar: {output_dir / 'texts'}")
        if args.srt:
            logger.info(f"SRT fayllar: {output_dir / 'srt'}")
    else:
        writer.write_manifest(results, manifest_path)

        if args.txt:
            txt_dir = str(output_dir / "texts")
            writer.write_text_files(results, txt_dir)
            logger.info(f"Text fayllar: {txt_dir}")

        if args.srt:
            srt_dir = str(output_dir / "srt")
            writer.write_srt_files(results, srt_dir)
            logger.info(f"SRT fayllar: {srt_dir}")

    manifest_rows = ManifestWriter.count_jsonl_rows(Path(manifest_path))
    failure_rows = ManifestWriter.count_jsonl_rows(failures_path)

    # -- Summary -------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("YAKUNIY NATIJA:")
    logger.info(f"  Audio fayllar:  {len(audio_files)}")
    logger.info(f"  Muvaffaqiyatli: {manifest_rows}")
    logger.info(f"  Xatolik:        {failure_rows + len(skipped)}")
    logger.info(f"  Manifest:       {manifest_path}")
    logger.info(f"  WAV papka:      {wav_dir}")
    logger.info("=" * 60)
    logger.info(
        "ESLATMA: Natijalar pseudo-label sifatida yaratildi. "
        "Fine-tuning uchun matnlarni qo'lda tekshiring va tuzating!"
    )


if __name__ == "__main__":
    main()
