#!/usr/bin/env python3
"""Fine-tune NVIDIA Parakeet TDT 0.6B v3 for Uzbek ASR.

This script:
1. Downloads parakeet-tdt-0.6b-v3 (CC-BY-4.0, 600M params)
2. Trains a new BPE tokenizer on Uzbek training text
3. Replaces the model's tokenizer (reinitializes decoder)
4. Fine-tunes on Uzbek data (104h + call center)
5. Saves checkpoints and final model

Architecture: FastConformer encoder + TDT (Token-and-Duration Transducer) decoder
- TDT jointly predicts tokens AND durations (how many frames to skip)
- 2-3x faster inference than standard transducers
- Dual-head joint network: token distribution + duration distribution
- Durations: [0, 1, 2, 3, 4] — model learns to skip frames

Key difference from CTC-only training (v1.0-v1.4):
- TDT uses autoregressive prediction network (like RNN-T but faster)
- The encoder is reused from pretrained model (multilingual features)
- Only decoder/joint are reinitialized when tokenizer changes

Usage:
    # Step 1: Prepare tokenizer (run once)
    python3 finetune_parakeet_tdt.py --prepare-tokenizer

    # Step 2: Fine-tune
    python3 finetune_parakeet_tdt.py --train

    # Full pipeline
    python3 finetune_parakeet_tdt.py --prepare-tokenizer --train
"""

import argparse
import json
import logging
import os
import random
import types
from pathlib import Path

import numpy as np
import torch
import lightning.pytorch as pl
from omegaconf import OmegaConf, open_dict
from scipy.signal import butter, sosfilt

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.parts.preprocessing.perturb import Perturbation, register_perturbation
from nemo.utils.exp_manager import exp_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Defaults ──

PRETRAINED_MODEL = "nvidia/parakeet-tdt-0.6b-v3"
DEFAULT_TRAIN_MANIFEST = "/root/stt/v2_pipeline/manifests/combined_train_norm_v21.jsonl"
DEFAULT_VAL_MANIFEST = "/root/stt/v2_pipeline/manifests/combined_val_norm_v21.jsonl"
DEFAULT_TEST_MANIFEST = "/root/stt/v2_pipeline/manifests/test_fleurs_norm_v21.jsonl"
DEFAULT_OUTPUT_DIR = Path("/root/stt/v2_pipeline/models/parakeet_tdt_uzbek")
DEFAULT_TOKENIZER_DIR = Path("/root/stt/v2_pipeline/tokenizers/uzbek_bpe_1024")
VOCAB_SIZE = 1024  # BPE vocabulary size for Uzbek


# ── Telephone Perturbation (reuse from finetune_nemo.py) ──

class TelephonePerturbation(Perturbation):
    """Simulate telephone codec: bandpass 300-3400Hz + 8kHz downsample/upsample."""

    def __init__(self, prob=0.5, lowcut=300, highcut=3400, codec_sr=8000, filter_order=5):
        self._prob = prob
        self.lowcut = lowcut
        self.highcut = highcut
        self.codec_sr = codec_sr
        self.filter_order = filter_order

    def perturb(self, data):
        if random.random() > self._prob:
            return
        sr = data.sample_rate
        samples = data.samples.copy()
        nyq = sr / 2
        low = self.lowcut / nyq
        high = min(self.highcut / nyq, 0.99)
        sos = butter(self.filter_order, [low, high], btype="band", output="sos")
        samples = sosfilt(sos, samples).astype(np.float32)
        down_len = int(len(samples) * self.codec_sr / sr)
        if down_len > 0:
            indices = np.linspace(0, len(samples) - 1, down_len).astype(int)
            downsampled = samples[indices]
            up_len = len(data.samples)
            indices = np.linspace(0, len(downsampled) - 1, up_len).astype(int)
            samples = downsampled[indices]
        data._samples = samples

    def max_augmentation_length(self, length):
        return length


register_perturbation("telephone", TelephonePerturbation)


# ── Tokenizer Preparation ──

def extract_texts_from_manifest(manifest_path: Path) -> list[str]:
    """Extract all text entries from a NeMo manifest."""
    texts = []
    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entry = json.loads(line)
                text = entry.get("text", "").strip()
                if text:
                    texts.append(text)
    return texts


def prepare_tokenizer(
    train_manifest: Path,
    output_dir: Path,
    vocab_size: int = VOCAB_SIZE,
):
    """Train a SentencePiece BPE tokenizer on Uzbek training text.

    Uses NeMo's tokenizer tools to create a BPE tokenizer compatible
    with the Parakeet model. The tokenizer must cover all Uzbek characters:
    - Latin: a-z, A-Z, oʻ, gʻ (with ʻ modifier letter)
    - Special: ' (apostrophe variants)
    - Punctuation: . , ! ? - : ; ( )
    """
    import sentencepiece as spm

    output_dir.mkdir(parents=True, exist_ok=True)
    text_file = output_dir / "train_text.txt"

    log.info("Extracting texts from %s...", train_manifest)
    texts = extract_texts_from_manifest(train_manifest)
    log.info("Extracted %d text lines", len(texts))

    # Write texts to file for SentencePiece training
    with open(text_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")

    # Uzbek-specific characters that must be in the vocabulary
    uzbek_chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    uzbek_chars.extend(["ʻ", "'", "'", "ʼ"])  # Uzbek apostrophe variants
    uzbek_chars.extend(list(".,!?-:;()\" "))
    # Ensure oʻ and gʻ combinations work
    user_defined_symbols = ",".join(sorted(set(uzbek_chars)))

    model_prefix = str(output_dir / "tokenizer")

    log.info("Training SentencePiece BPE tokenizer (vocab_size=%d)...", vocab_size)
    spm.SentencePieceTrainer.train(
        input=str(text_file),
        model_prefix=model_prefix,
        model_type="bpe",
        vocab_size=vocab_size,
        character_coverage=1.0,  # Full coverage for Latin-script Uzbek
        pad_id=0,
        unk_id=1,
        bos_id=-1,
        eos_id=-1,
        # NeMo expects these specific IDs
        user_defined_symbols=user_defined_symbols,
        max_sentence_length=16384,
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
        num_threads=8,
        byte_fallback=True,  # Handle any unexpected characters
    )

    # Verify tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load(model_prefix + ".model")

    test_sentences = [
        "salom dunyo",
        "o'zbekiston respublikasi",
        "jinoyat kodeksi",
        "bosh prokuratura",
        "qonunchilik palatasi",
        "ma'lumot olish huquqi",
    ]

    log.info("Tokenizer verification:")
    for sent in test_sentences:
        tokens = sp.encode(sent, out_type=str)
        log.info("  '%s' -> %s (%d tokens)", sent, tokens, len(tokens))

    log.info("Tokenizer saved to: %s", output_dir)
    log.info("Vocab size: %d", sp.get_piece_size())

    # Create NeMo-compatible tokenizer config
    # NeMo expects: tokenizer_dir/tokenizer.model + vocab.txt
    vocab_file = output_dir / "vocab.txt"
    with open(vocab_file, "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(sp.id_to_piece(i) + "\n")

    log.info("Vocab written to: %s", vocab_file)
    return output_dir


# ── Training ──

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Parakeet TDT 0.6B for Uzbek ASR")
    parser.add_argument("--prepare-tokenizer", action="store_true", help="Train BPE tokenizer")
    parser.add_argument("--train", action="store_true", help="Run fine-tuning")
    parser.add_argument(
        "--model-name", type=str, default=PRETRAINED_MODEL,
        help="Pretrained model name (HuggingFace) or path to .nemo file",
    )
    parser.add_argument("--train-manifest", type=str, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--val-manifest", type=str, default=DEFAULT_VAL_MANIFEST)
    parser.add_argument("--test-manifest", type=str, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tokenizer-dir", type=Path, default=DEFAULT_TOKENIZER_DIR)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-encoder", type=int, default=5)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--name", type=str, default="parakeet_tdt_uzbek")
    parser.add_argument("--accumulate-grad-batches", type=int, default=8)
    parser.add_argument("--speed-perturb", action="store_true", default=False)
    parser.add_argument("--noise-manifest", type=str, default=None)
    parser.add_argument("--noise-prob", type=float, default=0.5)
    parser.add_argument("--min-snr-db", type=float, default=5)
    parser.add_argument("--max-snr-db", type=float, default=20)
    parser.add_argument("--telephone-aug", action="store_true", default=False)
    parser.add_argument("--telephone-prob", type=float, default=0.3)
    parser.add_argument(
        "--train-mode", type=str, default="tdt",
        choices=["tdt", "ctc-only"],
        help="Training mode: tdt = full TDT+CTC hybrid, ctc-only = CTC path only",
    )
    parser.add_argument("--precision", type=str, default="bf16-mixed")
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    if not args.prepare_tokenizer and not args.train:
        parser.print_help()
        return

    # ── Step 1: Prepare tokenizer ──
    if args.prepare_tokenizer:
        prepare_tokenizer(
            train_manifest=Path(args.train_manifest),
            output_dir=args.tokenizer_dir,
            vocab_size=args.vocab_size,
        )
        if not args.train:
            log.info("Tokenizer prepared. Run with --train to start fine-tuning.")
            return

    # ── Step 2: Load model ──
    model_name = args.model_name
    if Path(model_name).exists():
        log.info("Loading model from local path: %s", model_name)
        model = nemo_asr.models.ASRModel.restore_from(model_name, map_location="cpu")
    else:
        log.info("Downloading pretrained model: %s", model_name)
        model = nemo_asr.models.ASRModel.from_pretrained(model_name, map_location="cpu")

    log.info("Model loaded: %s (%.1fM params)", type(model).__name__,
             sum(p.numel() for p in model.parameters()) / 1e6)

    # ── Step 3: Replace tokenizer with Uzbek BPE ──
    tokenizer_dir = args.tokenizer_dir
    tokenizer_model = tokenizer_dir / "tokenizer.model"

    if not tokenizer_model.exists():
        log.error("Tokenizer not found at %s. Run with --prepare-tokenizer first.", tokenizer_model)
        return

    log.info("Replacing tokenizer with Uzbek BPE from %s...", tokenizer_dir)

    # Get original vocab size for comparison
    orig_vocab = model.tokenizer.vocab_size if hasattr(model.tokenizer, 'vocab_size') else "unknown"
    log.info("Original tokenizer vocab size: %s", orig_vocab)

    # Use NeMo's change_vocabulary method which handles decoder reinitialization
    # For TDT hybrid models, this reinitializes both the TDT decoder/joint AND CTC decoder
    model.change_vocabulary(
        new_tokenizer_dir=str(tokenizer_dir),
        new_tokenizer_type="bpe",
    )

    new_vocab = model.tokenizer.vocab_size if hasattr(model.tokenizer, 'vocab_size') else "unknown"
    log.info("New tokenizer vocab size: %s", new_vocab)
    log.info("Decoder reinitialized for Uzbek vocabulary")

    # ── Step 4: Configure data ──
    with open_dict(model.cfg):
        # Training data
        model.cfg.train_ds.manifest_filepath = args.train_manifest
        model.cfg.train_ds.batch_size = args.batch_size
        model.cfg.train_ds.num_workers = 4
        model.cfg.train_ds.pin_memory = True
        model.cfg.train_ds.max_duration = 30
        model.cfg.train_ds.min_duration = 2.0

        # Speed perturbation
        if args.speed_perturb:
            model.cfg.train_ds.speed_perturb = True
            log.info("Speed perturbation ENABLED")

        # Stronger SpecAugment
        any_aug = args.speed_perturb or args.noise_manifest or args.telephone_aug
        if any_aug and hasattr(model.cfg, "spec_augment"):
            model.cfg.spec_augment.freq_masks = 3
            model.cfg.spec_augment.time_masks = 12
            log.info("SpecAugment strengthened: freq_masks=3, time_masks=12")

        # Noise augmentation
        augmentor_cfg = {}
        if args.noise_manifest:
            noise_path = Path(args.noise_manifest)
            if not noise_path.exists():
                raise FileNotFoundError(f"Noise manifest not found: {noise_path}")
            augmentor_cfg["noise"] = {
                "prob": args.noise_prob,
                "manifest_path": str(noise_path),
                "min_snr_db": args.min_snr_db,
                "max_snr_db": args.max_snr_db,
            }
            log.info("Noise augmentation ENABLED: prob=%.1f, SNR=[%.0f,%.0f]dB",
                      args.noise_prob, args.min_snr_db, args.max_snr_db)

        # Telephone simulation
        if args.telephone_aug:
            augmentor_cfg["telephone"] = {"prob": args.telephone_prob}
            log.info("Telephone simulation ENABLED: prob=%.1f", args.telephone_prob)

        if augmentor_cfg:
            model.cfg.train_ds.augmentor = OmegaConf.create(augmentor_cfg)

        # Validation data
        model.cfg.validation_ds.manifest_filepath = args.val_manifest
        model.cfg.validation_ds.batch_size = args.batch_size
        model.cfg.validation_ds.num_workers = 4
        model.cfg.validation_ds.pin_memory = True

        # Test data
        model.cfg.test_ds.manifest_filepath = args.test_manifest
        model.cfg.test_ds.batch_size = args.batch_size
        model.cfg.test_ds.num_workers = 4
        model.cfg.test_ds.pin_memory = True

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    log.info("Data configured: train=%s, val=%s", args.train_manifest, args.val_manifest)

    # ── Step 5: Configure optimizer ──
    with open_dict(model.cfg):
        model.cfg.optim.name = "adamw"
        model.cfg.optim.lr = args.lr
        model.cfg.optim.weight_decay = 1e-4
        model.cfg.optim.sched.name = "CosineAnnealing"
        model.cfg.optim.sched.warmup_steps = 1000  # More warmup for larger model
        model.cfg.optim.sched.min_lr = 1e-6

    log.info("Optimizer: AdamW, lr=%.1e, warmup=1000 steps", args.lr)

    # ── Step 6: Freeze encoder ──
    if args.freeze_encoder > 0:
        model.encoder.freeze()
        log.info("Encoder FROZEN for first %d epochs", args.freeze_encoder)

    # ── Step 7: CTC-only mode (optional) ──
    if args.train_mode == "ctc-only":
        log.info("CTC-only mode: disabling TDT loss path")
        # Import the CTC-only training functions from the existing script
        from finetune_nemo import enable_ctc_only_training
        enable_ctc_only_training(model)
    else:
        log.info("TDT mode: training with full TDT + CTC hybrid loss")
        # Disable CUDA graphs to avoid driver issues
        os.environ["NEMO_DISABLE_CUDA_GRAPHS"] = "1"
        with open_dict(model.cfg):
            if "decoding" in model.cfg and model.cfg.decoding is not None:
                if "greedy" in model.cfg.decoding and model.cfg.decoding.greedy is not None:
                    if "use_cuda_graph_decoder" in model.cfg.decoding.greedy:
                        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        if hasattr(model, "decoding") and hasattr(model.decoding, "use_cuda_graph_decoder"):
            model.decoding.use_cuda_graph_decoder = False

    # ── Step 8: Trainer ──
    trainer_cfg = {
        "devices": args.gpus,
        "accelerator": "gpu",
        "max_epochs": args.epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "gradient_clip_val": 1.0,
        "precision": args.precision,
        "log_every_n_steps": 50,
        "val_check_interval": 0.5,
        "enable_checkpointing": False,
        "logger": False,
        "num_sanity_val_steps": 2,
    }

    if args.max_steps > 0:
        trainer_cfg["max_steps"] = args.max_steps

    if args.gpus > 1:
        trainer_cfg["strategy"] = "ddp"

    trainer = pl.Trainer(**trainer_cfg)

    # ── Step 9: Experiment manager ──
    output_dir = args.output_dir.resolve()
    exp_cfg = {
        "exp_dir": str(output_dir),
        "name": args.name,
        "checkpoint_callback_params": {
            "monitor": "val_wer",
            "mode": "min",
            "save_top_k": 3,
            "always_save_nemo": True,
        },
        "resume_if_exists": True,
        "resume_ignore_no_checkpoint": True,
    }
    exp_manager(trainer, cfg=OmegaConf.create(exp_cfg))
    log.info("Experiment dir: %s/%s", output_dir, args.name)

    # ── Step 10: Encoder unfreezing callback ──
    if args.freeze_encoder > 0:
        class UnfreezeEncoderCallback(pl.Callback):
            def __init__(self, unfreeze_at_epoch: int):
                self.unfreeze_at_epoch = unfreeze_at_epoch
                self.unfrozen = False

            def on_train_epoch_start(self, trainer, pl_module):
                if not self.unfrozen and trainer.current_epoch >= self.unfreeze_at_epoch:
                    pl_module.encoder.unfreeze()
                    self.unfrozen = True
                    log.info("═══ ENCODER UNFROZEN at epoch %d ═══", trainer.current_epoch)

        trainer.callbacks.append(UnfreezeEncoderCallback(args.freeze_encoder))

    # ── Step 11: Train! ──
    eff_batch = args.batch_size * args.accumulate_grad_batches * args.gpus
    log.info("Starting training: %d epochs, batch=%d, effective_batch=%d, gpus=%d",
             args.epochs, args.batch_size, eff_batch, args.gpus)
    log.info("Model: %s (%s)", PRETRAINED_MODEL, args.train_mode)
    log.info("Precision: %s", args.precision)
    trainer.fit(model)

    # ── Step 12: Test & Save ──
    if trainer.is_global_zero:
        log.info("Running test evaluation...")
        model.setup_test_data(model.cfg.test_ds)
        trainer.test(model)

        final_path = output_dir / args.name / "final_model.nemo"
        model.save_to(str(final_path))
        log.info("Final model saved to: %s", final_path)
        log.info("")
        log.info("To serve this model:")
        log.info("  python3 serve_parakeet_uzbek.py --model-path %s", final_path)


if __name__ == "__main__":
    main()
