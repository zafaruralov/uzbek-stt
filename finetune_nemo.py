#!/usr/bin/env python3
"""Fine-tune a NeMo ASR model on Uzbek audiobook data.

This script:
1. Loads a pre-trained FastConformer Hybrid RNNT-CTC model (already Uzbek)
2. Points it at our segmented training data
3. Fine-tunes with a low learning rate to improve on our specific domain
4. Saves checkpoints and the final model

Key concepts:
- We use PyTorch Lightning (via NeMo) which handles the training loop
- The model is a "Hybrid" — it trains with both CTC and RNNT loss simultaneously
- We freeze the encoder initially (optional) to prevent catastrophic forgetting
- Learning rate is kept low since the model already knows Uzbek
"""

import argparse
import logging
import types
from pathlib import Path

import torch
import lightning.pytorch as pl  # must use 'lightning' not 'pytorch_lightning' to match NeMo's base class
from omegaconf import OmegaConf, open_dict

import nemo.collections.asr as nemo_asr
from nemo.collections.asr.data.audio_to_text_dali import DALIOutputs
from nemo.collections.asr.losses.rnnt import RNNTLoss
from nemo.core.classes.mixins import AccessMixin
from nemo.utils.exp_manager import exp_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Path to the pre-trained Uzbek model (.nemo file)
# ──────────────────────────────────────────────
DEFAULT_MODEL_PATH = (
    "/root/.cache/huggingface/hub/models--RubaiLab--fastconformer_aligner_uzbek"
    "/snapshots/89d28427c00bf146db07ac1acadd55dcb4a090d1"
    "/FastConformer-Hybrid-RNNT-CTC-Uzbek.nemo"
)

# ──────────────────────────────────────────────
# Default manifests (can be overridden by CLI args)
# ──────────────────────────────────────────────
DEFAULT_DATA_DIR = Path("/root/stt/nemo_data")
DEFAULT_TRAIN_MANIFEST = str(DEFAULT_DATA_DIR / "manifest_train.jsonl")
DEFAULT_VAL_MANIFEST = str(DEFAULT_DATA_DIR / "manifest_val.jsonl")
DEFAULT_TEST_MANIFEST = str(DEFAULT_DATA_DIR / "manifest_test.jsonl")


def enable_rnnt_pytorch_loss(model):
    """Switch RNNT loss to pure PyTorch implementation (slow, but avoids numba CUDA kernels)."""
    with open_dict(model.cfg):
        if "loss" not in model.cfg or model.cfg.loss is None:
            model.cfg.loss = OmegaConf.create({})
        model.cfg.loss.loss_name = "pytorch"
        model.cfg.loss.pytorch_kwargs = OmegaConf.create({})

    # Rebuild loss module so runtime change takes effect.
    model.loss = RNNTLoss(
        num_classes=model.joint.num_classes_with_blank - 1,
        reduction=model.cfg.get("rnnt_reduction", "mean_batch"),
        loss_name="pytorch",
        loss_kwargs={},
    )
    log.info("RNNT loss switched to `pytorch` (debug-safe mode; much slower).")


def enable_ctc_only_training(model):
    """Monkey-patch hybrid model to train/eval with CTC loss only (no RNNT kernel usage)."""

    # Disable interCTC extras to keep the path simple and deterministic.
    with open_dict(model.cfg):
        if "interctc" in model.cfg:
            model.cfg.interctc.loss_weights = []
            model.cfg.interctc.apply_at_layers = []
        if "aux_ctc" in model.cfg:
            model.cfg.aux_ctc.ctc_loss_weight = 1.0

    # Prevent parent epoch-end hooks from expecting hybrid-specific ctc_* aggregates.
    model.ctc_loss_weight = 0.0

    # RNNT decoder/joint are not needed in CTC-only mode.
    model.decoder.freeze()
    model.joint.freeze()

    def training_step_ctc_only(self, batch, batch_nb):
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        log_probs = self.ctc_decoder(encoder_output=encoded)
        loss_value = self.ctc_loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )

        if hasattr(self, "_trainer") and self._trainer is not None:
            log_every_n_steps = self._trainer.log_every_n_steps
            sample_id = self._trainer.global_step
        else:
            log_every_n_steps = 1
            sample_id = batch_nb
        compute_wer = ((sample_id + 1) % log_every_n_steps) == 0

        tensorboard_logs = {
            "learning_rate": self._optimizer.param_groups[0]["lr"],
            "global_step": torch.tensor(self.trainer.global_step, dtype=torch.float32),
            "train_ctc_loss": loss_value,
            "train_loss": loss_value,
        }

        if compute_wer:
            self.ctc_wer.update(
                predictions=log_probs,
                targets=transcript,
                targets_lengths=transcript_len,
                predictions_lengths=encoded_len,
            )
            ctc_wer, _, _ = self.ctc_wer.compute()
            self.ctc_wer.reset()
            tensorboard_logs["training_batch_wer_ctc"] = ctc_wer

        self.log_dict(tensorboard_logs)
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        return {"loss": loss_value}

    def validation_pass_ctc_only(self, batch, batch_idx, dataloader_idx):
        if self.is_interctc_enabled():
            AccessMixin.set_access_enabled(access_enabled=True, guid=self.model_guid)

        signal, signal_len, transcript, transcript_len = batch
        if isinstance(batch, DALIOutputs) and batch.has_processed_signal:
            encoded, encoded_len = self.forward(processed_signal=signal, processed_signal_length=signal_len)
        else:
            encoded, encoded_len = self.forward(input_signal=signal, input_signal_length=signal_len)
        del signal

        log_probs = self.ctc_decoder(encoder_output=encoded)
        ctc_loss = self.ctc_loss(
            log_probs=log_probs,
            targets=transcript,
            input_lengths=encoded_len,
            target_lengths=transcript_len,
        )

        self.ctc_wer.update(
            predictions=log_probs,
            targets=transcript,
            targets_lengths=transcript_len,
            predictions_lengths=encoded_len,
        )
        ctc_wer, ctc_wer_num, ctc_wer_denom = self.ctc_wer.compute()
        self.ctc_wer.reset()

        logs = {
            "val_loss": ctc_loss,
            "val_ctc_loss": ctc_loss,
            "val_wer": ctc_wer,
            "val_wer_num": ctc_wer_num,
            "val_wer_denom": ctc_wer_denom,
        }
        self.log("global_step", torch.tensor(self.trainer.global_step, dtype=torch.float32))
        if AccessMixin.is_access_enabled(self.model_guid):
            AccessMixin.reset_registry(self)
        return logs

    def multi_validation_epoch_end_ctc_only(self, outputs, dataloader_idx=0):
        if not outputs or not all(isinstance(x, dict) for x in outputs):
            logging.warning("No outputs to process for validation_epoch_end")
            return {}
        val_loss_mean = torch.stack([x["val_loss"] for x in outputs]).mean()
        wer_num = torch.stack([x["val_wer_num"] for x in outputs]).sum()
        wer_denom = torch.stack([x["val_wer_denom"] for x in outputs]).sum()
        tensorboard_logs = {"val_loss": val_loss_mean, "val_wer": wer_num.float() / wer_denom}
        return {"val_loss": val_loss_mean, "log": tensorboard_logs}

    def multi_test_epoch_end_ctc_only(self, outputs, dataloader_idx=0):
        if not outputs or not all(isinstance(x, dict) for x in outputs):
            logging.warning("No outputs to process for test_epoch_end")
            return {}
        test_loss_mean = torch.stack([x["test_loss"] for x in outputs]).mean()
        wer_num = torch.stack([x["test_wer_num"] for x in outputs]).sum()
        wer_denom = torch.stack([x["test_wer_denom"] for x in outputs]).sum()
        tensorboard_logs = {"test_loss": test_loss_mean, "test_wer": wer_num.float() / wer_denom}
        return {"test_loss": test_loss_mean, "log": tensorboard_logs}

    model.training_step = types.MethodType(training_step_ctc_only, model)
    model.validation_pass = types.MethodType(validation_pass_ctc_only, model)
    model.multi_validation_epoch_end = types.MethodType(multi_validation_epoch_end_ctc_only, model)
    model.multi_test_epoch_end = types.MethodType(multi_test_epoch_end_ctc_only, model)
    log.info("CTC-only mode enabled: RNNT loss path is bypassed in train/val/test.")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune NeMo ASR on Uzbek data")
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to base .nemo model.",
    )
    parser.add_argument(
        "--train-manifest",
        type=str,
        default=DEFAULT_TRAIN_MANIFEST,
        help="Path to training manifest JSONL.",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default=DEFAULT_VAL_MANIFEST,
        help="Path to validation manifest JSONL.",
    )
    parser.add_argument(
        "--test-manifest",
        type=str,
        default=DEFAULT_TEST_MANIFEST,
        help="Path to test manifest JSONL.",
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Number of training epochs (full passes over the dataset)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size per GPU. Lower if you get CUDA OOM",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4,
        help="Peak learning rate. Keep low for fine-tuning (1e-4 to 3e-4)",
    )
    parser.add_argument(
        "--freeze-encoder", type=int, default=5,
        help="Freeze encoder for this many epochs (0 to disable). "
             "Helps prevent 'catastrophic forgetting' of what the model already knows.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/root/stt/nemo_experiment"),
        help="Directory for checkpoints, logs, and the final model",
    )
    parser.add_argument(
        "--gpus", type=int, default=1,
        help="Number of GPUs to use (1 or 2)",
    )
    parser.add_argument(
        "--name", type=str, default="uzbek_fastconformer_finetune",
        help="Experiment name (used in log directories)",
    )
    parser.add_argument(
        "--train-mode",
        type=str,
        default="ctc-only",
        choices=["rnnt", "rnnt-smoke", "ctc-only"],
        help=(
            "Training mode: "
            "`rnnt` = original hybrid RNNT+CTC, "
            "`rnnt-smoke` = RNNT with pure-pytorch loss for crash debugging, "
            "`ctc-only` = bypass RNNT and train only CTC path."
        ),
    )
    parser.add_argument(
        "--smoke-steps",
        type=int,
        default=200,
        help="Max steps for `rnnt-smoke` mode (<=0 disables max_steps cap).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=0,
        help="Global max training steps (any mode). <=0 means disabled.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        help="Override trainer precision (e.g., `32`, `bf16-mixed`).",
    )
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=4,
        help="Gradient accumulation steps.",
    )
    args = parser.parse_args()
    model_path = Path(args.model_path)
    train_manifest = Path(args.train_manifest)
    val_manifest = Path(args.val_manifest)
    test_manifest = Path(args.test_manifest)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")
    for p in (train_manifest, val_manifest, test_manifest):
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")

    # ──────────────────────────────────────────
    # STEP 1: Load the pre-trained model
    # ──────────────────────────────────────────
    # .nemo files are self-contained archives: model weights + config + tokenizer
    # restore_from() unpacks everything and gives us a ready-to-use model
    log.info("Loading pre-trained model from %s", model_path)
    model = nemo_asr.models.ASRModel.restore_from(str(model_path), map_location="cpu")
    log.info("Model loaded: %s (%.1fM params)", type(model).__name__,
             sum(p.numel() for p in model.parameters()) / 1e6)
    log.info("Train mode: %s", args.train_mode)

    if args.train_mode == "rnnt-smoke":
        enable_rnnt_pytorch_loss(model)
    elif args.train_mode == "ctc-only":
        enable_ctc_only_training(model)

    # ──────────────────────────────────────────
    # STEP 2: Update the model's data configuration
    # ──────────────────────────────────────────
    # The model remembers its original training config. We override it to point
    # at our data. OmegaConf is NeMo's config system (like a fancy dictionary).
    #
    # "open_dict" lets us modify the config even if it was frozen/structured
    with open_dict(model.cfg):
        # --- Training data ---
        model.cfg.train_ds.manifest_filepath = str(train_manifest)
        model.cfg.train_ds.batch_size = args.batch_size
        model.cfg.train_ds.num_workers = 4  # parallel data loading threads
        model.cfg.train_ds.pin_memory = True  # faster CPU→GPU transfer
        model.cfg.train_ds.max_duration = 30  # match our segment max length
        model.cfg.train_ds.min_duration = 2.0  # skip very short segments

        # --- Validation data ---
        model.cfg.validation_ds.manifest_filepath = str(val_manifest)
        model.cfg.validation_ds.batch_size = args.batch_size
        model.cfg.validation_ds.num_workers = 4
        model.cfg.validation_ds.pin_memory = True

        # --- Test data ---
        model.cfg.test_ds.manifest_filepath = str(test_manifest)
        model.cfg.test_ds.batch_size = args.batch_size
        model.cfg.test_ds.num_workers = 4
        model.cfg.test_ds.pin_memory = True

    # Tell the model to re-create its data loaders with the new config
    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    log.info("Data configured: train=%s, val=%s, test=%s", train_manifest, val_manifest, test_manifest)

    # ──────────────────────────────────────────
    # STEP 3: Configure the optimizer & scheduler
    # ──────────────────────────────────────────
    # For fine-tuning, we use a lower learning rate than training from scratch.
    #
    # CosineAnnealing: starts at lr, gradually decreases to min_lr following
    # a cosine curve. This is gentler than step-decay schedulers.
    #
    # Warmup: the first N steps use an even smaller lr, ramping up to the peak.
    # This prevents the model from making wild updates at the start.
    with open_dict(model.cfg):
        model.cfg.optim.name = "adamw"  # Adam with weight decay (standard)
        model.cfg.optim.lr = args.lr
        model.cfg.optim.weight_decay = 1e-4  # small regularization to prevent overfitting
        model.cfg.optim.sched.name = "CosineAnnealing"
        model.cfg.optim.sched.warmup_steps = 500  # gentle ramp-up
        model.cfg.optim.sched.min_lr = 1e-6  # don't let lr go to zero

    log.info("Optimizer: AdamW, lr=%.1e, warmup=500 steps, cosine → %.1e",
             args.lr, 1e-6)

    # ──────────────────────────────────────────
    # STEP 4: (Optional) Freeze encoder initially
    # ──────────────────────────────────────────
    # "Freezing" = setting requires_grad=False on parameters so they don't update.
    # Why? The encoder already learned great Uzbek audio features. If we update
    # everything at once, we might destroy that knowledge ("catastrophic forgetting").
    # Strategy: freeze encoder for a few epochs, only train decoders, then unfreeze.
    if args.freeze_encoder > 0:
        model.encoder.freeze()
        log.info("Encoder FROZEN for first %d epochs (only decoders will train)", args.freeze_encoder)

    # ──────────────────────────────────────────
    # STEP 5: Set up the Trainer (PyTorch Lightning)
    # ──────────────────────────────────────────
    # PyTorch Lightning's Trainer handles:
    # - The training loop (forward, loss, backward, optimizer step)
    # - Validation after each epoch
    # - Checkpointing (saving model snapshots)
    # - GPU management, mixed precision, logging
    #
    # We don't write any training loop code — Lightning does it all.
    default_precision = "bf16-mixed"
    if args.train_mode == "rnnt-smoke":
        default_precision = "32"
    trainer_precision = args.precision if args.precision is not None else default_precision

    trainer_cfg = {
        "devices": args.gpus,
        "accelerator": "gpu",
        "max_epochs": args.epochs,
        "accumulate_grad_batches": args.accumulate_grad_batches,
        # ↑ effective_batch = batch_size × accumulate × gpus
        "gradient_clip_val": 1.0,  # prevent exploding gradients
        "precision": trainer_precision,
        "log_every_n_steps": 50,
        "val_check_interval": 0.5,  # validate twice per epoch
        "enable_checkpointing": False,  # let exp_manager handle checkpointing instead
        "logger": False,  # exp_manager handles logging (see below)
        "num_sanity_val_steps": 2,  # quick validation sanity check at start
    }

    if args.max_steps > 0:
        trainer_cfg["max_steps"] = args.max_steps
    elif args.train_mode == "rnnt-smoke" and args.smoke_steps > 0:
        trainer_cfg["max_steps"] = args.smoke_steps

    # Add DDP strategy for multi-GPU
    if args.gpus > 1:
        trainer_cfg["strategy"] = "ddp"

    trainer = pl.Trainer(**trainer_cfg)

    # ──────────────────────────────────────────
    # STEP 6: Set up experiment manager
    # ──────────────────────────────────────────
    # NeMo's exp_manager wraps Lightning's logging/checkpointing with extras:
    # - TensorBoard logs (loss curves, WER over time)
    # - Automatic .nemo file saving (the best model, packaged and ready to use)
    # - Resume from checkpoint if training is interrupted
    output_dir = args.output_dir.resolve()
    exp_cfg = {
        "exp_dir": str(output_dir),
        "name": args.name,
        "checkpoint_callback_params": {
            "monitor": "val_wer",  # save the model with lowest Word Error Rate
            "mode": "min",
            "save_top_k": 3,  # keep 3 best checkpoints
            "always_save_nemo": True,  # also save as .nemo (self-contained)
        },
        "resume_if_exists": True,  # resume training if we restart the script
        "resume_ignore_no_checkpoint": True,
    }
    exp_manager(trainer, cfg=OmegaConf.create(exp_cfg))
    log.info("Experiment dir: %s/%s", output_dir, args.name)

    # ──────────────────────────────────────────
    # STEP 7: Encoder unfreezing callback
    # ──────────────────────────────────────────
    if args.freeze_encoder > 0:
        class UnfreezeEncoderCallback(pl.Callback):
            """Unfreeze the encoder after N epochs.

            This is a PyTorch Lightning "callback" — a hook that runs at
            specific points during training. on_train_epoch_start runs
            at the beginning of each epoch.
            """
            def __init__(self, unfreeze_at_epoch: int):
                self.unfreeze_at_epoch = unfreeze_at_epoch
                self.unfrozen = False

            def on_train_epoch_start(self, trainer, pl_module):
                if not self.unfrozen and trainer.current_epoch >= self.unfreeze_at_epoch:
                    pl_module.encoder.unfreeze()
                    self.unfrozen = True
                    log.info("═══ ENCODER UNFROZEN at epoch %d ═══", trainer.current_epoch)

        trainer.callbacks.append(UnfreezeEncoderCallback(args.freeze_encoder))

    # ──────────────────────────────────────────
    # STEP 8: Train!
    # ──────────────────────────────────────────
    log.info("Starting training: %d epochs, batch=%d, gpus=%d", args.epochs, args.batch_size, args.gpus)
    log.info("Effective batch size: %d (batch=%d × accumulate=%d × gpus=%d)",
             args.batch_size * args.accumulate_grad_batches * args.gpus,
             args.batch_size, args.accumulate_grad_batches, args.gpus)
    log.info("Trainer precision: %s", trainer_precision)
    if args.train_mode == "rnnt-smoke" and args.smoke_steps > 0:
        log.info("Smoke mode active: max_steps=%d", args.smoke_steps)
    if args.max_steps > 0:
        log.info("Global max_steps active: %d", args.max_steps)
    trainer.fit(model)

    # ──────────────────────────────────────────
    # STEP 9: Test the final model
    # ──────────────────────────────────────────
    if trainer.is_global_zero:
        log.info("Running test evaluation...")
        model.setup_test_data(model.cfg.test_ds)
        trainer.test(model)

        # Save the final model as .nemo
        final_path = output_dir / args.name / "final_model.nemo"
        model.save_to(str(final_path))
        log.info("Final model saved to: %s", final_path)


if __name__ == "__main__":
    main()
