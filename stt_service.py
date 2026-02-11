import torch
import numpy as np
import av
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UzbekSTTService:
    def __init__(self, model_name="RubaiLab/kotib_call_base_stt_15", offline_mode=False):
        """
        Uzbek STT modelni yuklash

        Args:
            model_name: Model nomi yoki local path
            offline_mode: True bo'lsa, faqat cache'dan yuklaydi (internet'siz)
        """
        logger.info("Model yuklanmoqda...")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Qurilma: {self.device}")
        logger.info(f"Offline mode: {offline_mode}")

        # Whisper chunk parametrlari
        self.chunk_length_seconds = 30  # Maksimal 30 soniya
        self.sample_rate = 16000

        try:
            logger.info("Processor yuklanmoqda...")
            self.processor = WhisperProcessor.from_pretrained(
                model_name,
                token=True,
                local_files_only=offline_mode
            )

            logger.info("Model yuklanmoqda...")
            self.model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                token=True,
                local_files_only=offline_mode
            )

            self.model.to(self.device)
            self.model.eval()

            logger.info("Model muvaffaqiyatli yuklandi!")

        except OSError as e:
            if offline_mode or "Temporary failure in name resolution" in str(e):
                logger.error("\n" + "=" * 60)
                logger.error("❌ MODEL CACHE'DA TOPILMADI!")
                logger.error("=" * 60)
                logger.error("Model hali to'liq yuklanmagan.")
                logger.error("\n💡 Yechim:")
                logger.error("   1. Internet'ga ulaning")
                logger.error("   2. Quyidagi buyruqni bajaring:")
                logger.error("      python download_model.py")
                logger.error("   3. Model yuklangandan keyin offline ishlaydi")
                logger.error("=" * 60)
            raise

    def load_audio(self, audio_path):
        """Audio faylni yuklash va tayyorlash (PyAV orqali)"""
        audio_path = Path(audio_path)
        file_ext = audio_path.suffix.lower()

        logger.info(f"Fayl formati: {file_ext}")

        try:
            container = av.open(str(audio_path))
            audio_stream = container.streams.audio[0]

            resampler = av.AudioResampler(
                format='s16',
                layout='mono',
                rate=16000,
            )

            frames = []
            for packet in container.demux(audio=0):
                try:
                    for frame in packet.decode():
                        for resampled in resampler.resample(frame):
                            frames.append(resampled.to_ndarray())
                except av.error.InvalidDataError:
                    logger.debug("Buzilgan paket o'tkazib yuborildi")

            container.close()

            if not frames:
                raise RuntimeError("Audio faylda ma'lumot topilmadi")

            audio = np.concatenate(frames, axis=1).flatten().astype(np.float32) / 32768.0

            logger.info(f"✅ PyAV bilan yuklandi (16kHz mono, {len(audio) / 16000:.2f}s)")
            return audio

        except Exception as e:
            raise RuntimeError(
                f"Audio faylni yuklashda xatolik:\n"
                f"  PyAV: {str(e)}\n"
                f"Fayl formati qo'llab-quvvatlanmaydi yoki fayl buzilgan."
            )

    def transcribe_chunk(self, audio_chunk):
        """Bitta chunk'ni transcribe qilish"""
        inputs = self.processor(
            audio_chunk,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            predicted_ids = self.model.generate(**inputs)

        text = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True
        )[0]

        return text

    def transcribe(self, audio_path):
        """Audio faylni matunga o'girish (uzun audiolar uchun chunking)"""
        try:
            logger.info(f"Audio yuklanyapti: {audio_path}")

            # Audio yuklash
            audio_array = self.load_audio(audio_path)

            duration = len(audio_array) / self.sample_rate
            logger.info(f"Audio uzunligi: {duration:.2f} soniya")

            # Chunk length in samples
            chunk_length = self.chunk_length_seconds * self.sample_rate

            # Agar audio 30 soniyadan qisqa bo'lsa - oddiy transcribe
            if len(audio_array) <= chunk_length:
                logger.info("Audio processinga tayyorlanmoqda...")
                logger.info("Transcription boshlandi...")

                text = self.transcribe_chunk(audio_array)

                logger.info(f"Transcription tugadi!")

                return {
                    "success": True,
                    "text": text,
                    "language": "uz",
                    "duration": duration,
                    "chunks": 1
                }

            # Uzun audio uchun - chunklarga bo'lib transcribe qilish
            logger.info(f"Uzun audio! Bo'laklarga bo'linmoqda...")

            # Overlap bilan chunking (5 soniya overlap)
            overlap_length = 5 * self.sample_rate  # 5 soniya overlap
            stride = chunk_length - overlap_length

            chunks = []
            transcriptions = []

            start = 0
            chunk_num = 0

            while start < len(audio_array):
                end = min(start + chunk_length, len(audio_array))
                chunk = audio_array[start:end]

                chunk_num += 1
                chunk_duration = len(chunk) / self.sample_rate
                logger.info(
                    f"📝 Chunk {chunk_num}: {start / self.sample_rate:.1f}s - {end / self.sample_rate:.1f}s ({chunk_duration:.1f}s)")

                # Transcribe chunk
                text = self.transcribe_chunk(chunk)
                transcriptions.append(text)

                logger.info(f"   Matn: {text[:80]}...")

                # Keyingi chunkga o'tish
                start += stride

            # Barcha transcriptionslarni birlashtirish
            full_text = " ".join(transcriptions)

            logger.info(f"✅ Transcription tugadi! {chunk_num} ta chunk")

            return {
                "success": True,
                "text": full_text,
                "language": "uz",
                "duration": duration,
                "chunks": chunk_num
            }

        except Exception as e:
            logger.error(f"Xatolik: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "success": False,
                "error": str(e)
            }


# Test uchun
if __name__ == "__main__":
    logger.info("=== STT Service Test ===")

    # Offline mode'ni tekshirish
    import sys

    offline = "--offline" in sys.argv

    # Service yaratish
    try:
        service = UzbekSTTService(offline_mode=offline)
        logger.info("✅ Service tayyor!")

        # Audio fayl topish
        audio_files = [arg for arg in sys.argv[1:] if arg != "--offline"]

        if audio_files:
            audio_file = audio_files[0]
            logger.info(f"Audio fayl test qilinmoqda: {audio_file}")
            result = service.transcribe(audio_file)

            print("\n" + "=" * 60)
            print("NATIJA:")
            print("=" * 60)
            if result['success']:
                print(f"📝 Matn: {result['text']}")
                print(f"🌐 Til: {result['language']}")
                print(f"⏱️  Davomiyligi: {result['duration']:.2f} soniya")
                print(f"📊 Bo'laklar soni: {result['chunks']}")
            else:
                print(f"❌ Xatolik: {result['error']}")
            print("=" * 60)
        else:
            logger.info("Test uchun audio fayl yo'q")
            logger.info("\nIshlatish:")
            logger.info("  python stt_service.py audio.mp3")
            logger.info("  python stt_service.py audio.m4a")
            logger.info("  python stt_service.py --offline audio.wav")
            logger.info("\n💡 Uzun audiolar avtomatik bo'laklarga bo'linadi!")
            logger.info("   Har bir bo'lak: 30 soniya (5 soniya overlap)")

    except Exception as e:
        logger.error(f"❌ Xatolik: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)