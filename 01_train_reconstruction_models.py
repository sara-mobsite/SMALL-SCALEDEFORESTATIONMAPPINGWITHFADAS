#!/usr/bin/env python3
"""
run_4_reconstruction_models_50epochs.py

Sequential experiment:
    1) SegFormer-B0
    2) UNet + ResNet-50
    3) PSPNet + ResNet-50
    4) DeepLabV3+ + ResNet-50

For every model:
    - same train / validation / test data
    - same normalization
    - same batch size and optimizer settings
    - train FROM SCRATCH (no pretrained weights)
    - maximum 50 epochs
    - early stopping patience = 5
    - save BEST validation checkpoint
    - reload BEST checkpoint
    - evaluate the COMPLETE test set
    - compute overall test MSE + RMSE
    - compute VV-only and VH-only test MSE
    - save per-sample test MSE/RMSE
    - free model/loaders/optimizer and clear CUDA cache
    - then run the next architecture

Data:
    Set DATA_ROOT below to the local dataset directory containing:
        train/processed, validation/processed, and test/processed

Expected counts:
    train      32722
    validation 2736
    test        1519

Outputs:
    Set OUTPUT_ROOT below to the directory where checkpoints, histories,
    test metrics, and per-sample results should be written.

Install requirements if needed:
    pip install segmentation-models-pytorch timm transformers
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

# Keep Transformers on the PyTorch path only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    import segmentation_models_pytorch as smp
except ImportError as exc:
    raise ImportError(
        "Missing segmentation_models_pytorch. Install with:\n"
        "pip install segmentation-models-pytorch timm"
    ) from exc

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError as exc:
    raise ImportError(
        "Missing transformers. Install with:\n"
        "pip install transformers"
    ) from exc


# =============================================================================
# PATHS
# =============================================================================

# DATA_ROOT must point to the dataset "forest" directory.
# Required structure:
#   DATA_ROOT/
#       train/processed/       <- training Sentinel-1 VV/VH TIFF patches
#       validation/processed/  <- validation Sentinel-1 VV/VH TIFF patches
#       test/processed/        <- reconstruction test TIFF patches
DATA_ROOT = Path("PATH/TO/DATASET/forest")

TRAIN_DIR = DATA_ROOT / "train" / "processed"
VAL_DIR = DATA_ROOT / "validation" / "processed"
TEST_DIR = DATA_ROOT / "test" / "processed"

# OUTPUT_ROOT is where all model checkpoints and numerical results are saved.
# The script creates one subdirectory per reconstruction architecture.
OUTPUT_ROOT = Path("PATH/TO/RESULTS/four_models_reconstruction_50epochs")


# =============================================================================
# DATASET COUNTS
# =============================================================================

EXPECTED_TRAIN = 32722
EXPECTED_VAL = 2736
EXPECTED_TEST = 1519


# =============================================================================
# COMMON EXPERIMENT SETTINGS
# =============================================================================

MODEL_ORDER = (
    "segformer",
    "unet",
    "pspnet",
    "deeplabv3plus",
)

CNN_ENCODER = "resnet50"

MAX_EPOCHS = 50
BATCH_SIZE = 80

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 6e-6

STEP_SIZE = 5
GAMMA = 0.7

EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.0

MIN_VALUE = -15.0
MAX_VALUE = -3.0

SEED = 42
NUM_WORKERS = 2

USE_AMP = True

# Useful if the long script is interrupted after one or more models finished.
# A model is skipped only if its FINAL test-results JSON already exists.
SKIP_COMPLETED_MODELS = True

# Catch empty / malformed TIFFs before spending hours training.
VALIDATE_TIFF_HEADERS = True


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# FILES / TIFF READING
# =============================================================================

def find_tiffs(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Directory not found: {root}")

    files: list[Path] = []

    for pattern in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
        files.extend(root.glob(pattern))

    files = sorted(set(files))

    if not files:
        raise FileNotFoundError(f"No TIFF files found in: {root}")

    return files


def validate_headers(files: list[Path], split: str) -> None:
    print(f"\nChecking TIFF headers: {split} ({len(files)} files)")

    bad: list[tuple[str, str]] = []

    for i, path in enumerate(files, start=1):
        try:
            if path.stat().st_size <= 0:
                raise RuntimeError("0-byte file")

            with tifffile.TiffFile(path) as tif:
                if len(tif.pages) == 0:
                    raise RuntimeError("TIFF has no pages")

        except Exception as exc:
            bad.append((str(path), repr(exc)))

        if i % 5000 == 0 or i == len(files):
            print(f"  checked {i}/{len(files)}")

    if bad:
        lines = [f"Invalid TIFFs found in {split}: {len(bad)}"]

        for path, error in bad:
            lines.append(path)
            lines.append(f"  {error}")

        raise RuntimeError("\n".join(lines))

    print(f"  {split}: OK")


def read_two_band_tiff(path: Path) -> np.ndarray:
    """
    Read the actual pixel array with tifffile.

    Expected:
        [2,H,W] or [H,W,2]
    """
    image = np.asarray(
        tifffile.imread(path),
        dtype=np.float32,
    )

    image = np.squeeze(image)

    if image.ndim != 3:
        raise ValueError(
            f"Unexpected TIFF shape {image.shape}: {path}"
        )

    if image.shape[0] == 2:
        pass

    elif image.shape[-1] == 2:
        image = np.moveaxis(image, -1, 0)

    else:
        raise ValueError(
            f"Could not find 2-channel VV/VH axis in {image.shape}: {path}"
        )

    if not np.isfinite(image).all():
        raise ValueError(f"NaN/Inf values found: {path}")

    return image


# =============================================================================
# DATASET
# =============================================================================

class SARDataset(Dataset):
    def __init__(
        self,
        files: list[Path],
        min_value: float,
        max_value: float,
    ):
        self.files = files
        self.min_value = float(min_value)
        self.scale = float(max_value - min_value)

        if self.scale <= 0:
            raise ValueError("max_value must be > min_value")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]

        image = read_two_band_tiff(path)

        # Same normalization as the previous SegFormer experiment:
        #
        # -15 dB -> 0.0
        #  -9 dB -> 0.5
        #  -3 dB -> 1.0
        #
        # Intentionally NO clipping.
        image = (
            image - self.min_value
        ) / self.scale

        return torch.from_numpy(image), str(path)


# =============================================================================
# SEGFORMER-B0
# =============================================================================

def make_segformer_b0_config() -> SegformerConfig:
    """
    Explicit MiT-B0 / SegFormer-B0 configuration, initialized from scratch.
    """
    return SegformerConfig(
        num_channels=2,
        num_labels=2,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        semantic_loss_ignore_index=255,
    )


class SegFormerReconstructor(nn.Module):
    def __init__(self):
        super().__init__()

        config = make_segformer_b0_config()

        self.net = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(
            pixel_values=x
        ).logits

        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Continuous reconstruction, not a class probability.
        return torch.sigmoid(logits)


# =============================================================================
# UNET / PSPNET / DEEPLABV3+
# =============================================================================

class CNNReconstructor(nn.Module):
    def __init__(self, architecture: str):
        super().__init__()

        common = dict(
            encoder_name=CNN_ENCODER,
            encoder_weights=None,
            in_channels=2,
            classes=2,
            activation=None,
        )

        if architecture == "unet":
            self.net = smp.Unet(**common)

        elif architecture == "pspnet":
            self.net = smp.PSPNet(**common)

        elif architecture == "deeplabv3plus":
            self.net = smp.DeepLabV3Plus(**common)

        else:
            raise ValueError(f"Unsupported architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return torch.sigmoid(logits)


def build_model(model_name: str) -> nn.Module:
    if model_name == "segformer":
        return SegFormerReconstructor()

    return CNNReconstructor(model_name)


# =============================================================================
# LOADERS
# =============================================================================

def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = None

    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=False,
        drop_last=False,
        generator=generator,
    )


def create_data(
    train_files: list[Path],
    val_files: list[Path],
    test_files: list[Path],
    device: torch.device,
):
    train_set = SARDataset(
        train_files,
        MIN_VALUE,
        MAX_VALUE,
    )

    val_set = SARDataset(
        val_files,
        MIN_VALUE,
        MAX_VALUE,
    )

    test_set = SARDataset(
        test_files,
        MIN_VALUE,
        MAX_VALUE,
    )

    train_loader = make_loader(
        train_set,
        BATCH_SIZE,
        True,
        device,
        SEED,
    )

    val_loader = make_loader(
        val_set,
        BATCH_SIZE,
        False,
        device,
        SEED,
    )

    test_loader = make_loader(
        test_set,
        BATCH_SIZE,
        False,
        device,
        SEED,
    )

    return (
        train_set,
        val_set,
        test_set,
        train_loader,
        val_loader,
        test_loader,
    )


# =============================================================================
# LOSS
# =============================================================================

def per_sample_mse(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    One MSE per sample, averaged over:
        VV/VH + all HxW pixels.
    """
    return (
        reconstruction - target
    ).square().flatten(1).mean(1)


def training_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return per_sample_mse(
        reconstruction,
        target,
    ).mean()


# =============================================================================
# VALIDATION
# =============================================================================

@torch.no_grad()
def evaluate_mse(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.eval()

    total = 0.0
    count = 0

    for images, _ in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            reconstruction = model(images)

            losses = per_sample_mse(
                reconstruction,
                images,
            )

        total += losses.sum().item()
        count += images.shape[0]

    return total / max(count, 1)


# =============================================================================
# TEST COMPLETE SET USING BEST CHECKPOINT
# =============================================================================

@torch.no_grad()
def evaluate_complete_test_set(
    model_name: str,
    model: nn.Module,
    test_loader: DataLoader,
    test_set: Dataset,
    device: torch.device,
    amp_enabled: bool,
    model_dir: Path,
) -> dict:
    model.eval()

    total_mse_sum = 0.0
    total_samples = 0

    channel_error_sum = np.zeros(2, dtype=np.float64)
    channel_value_count = np.zeros(2, dtype=np.int64)

    rows: list[dict] = []

    for batch_index, (images, paths) in enumerate(
        test_loader,
        start=1,
    ):
        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            reconstruction = model(images)

            squared_error = (
                reconstruction - images
            ).square()

            sample_losses = (
                squared_error
                .flatten(1)
                .mean(1)
            )

        sample_losses_np = (
            sample_losses
            .float()
            .cpu()
            .numpy()
        )

        total_mse_sum += float(
            sample_losses_np.sum()
        )

        total_samples += len(
            sample_losses_np
        )

        channel_error_sum += (
            squared_error
            .float()
            .sum(dim=(0, 2, 3))
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        values_per_channel = (
            images.shape[0]
            * images.shape[2]
            * images.shape[3]
        )

        channel_value_count += values_per_channel

        for path, mse_value in zip(
            paths,
            sample_losses_np,
        ):
            mse_value = float(mse_value)

            rows.append(
                {
                    "source": path,
                    "test_mse": mse_value,
                    "test_rmse": math.sqrt(
                        max(mse_value, 0.0)
                    ),
                }
            )

        if (
            batch_index % 5 == 0
            or total_samples == len(test_set)
        ):
            print(
                f"{model_name:15s} | "
                f"test {total_samples}/{len(test_set)}"
            )

    test_mse = total_mse_sum / max(total_samples, 1)
    test_rmse = math.sqrt(max(test_mse, 0.0))

    channel_mse = (
        channel_error_sum
        / np.maximum(channel_value_count, 1)
    )

    per_sample_csv = (
        model_dir
        / "test_per_sample_mse.csv"
    )

    with per_sample_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "source",
                "test_mse",
                "test_rmse",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return {
        "test_samples": int(total_samples),
        "test_mse": float(test_mse),
        "test_rmse": float(test_rmse),
        "test_vv_mse": float(channel_mse[0]),
        "test_vh_mse": float(channel_mse[1]),
        "per_sample_test_csv": str(per_sample_csv),
    }


# =============================================================================
# CLEANUP
# =============================================================================

def clear_resources(model_name: str) -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

        free_bytes, total_bytes = torch.cuda.mem_get_info()

        print(
            f"GPU after {model_name} cleanup: "
            f"{free_bytes / (1024**3):.2f} GB free / "
            f"{total_bytes / (1024**3):.2f} GB total"
        )


# =============================================================================
# RUN ONE MODEL
# =============================================================================

def run_model(
    model_name: str,
    train_files: list[Path],
    val_files: list[Path],
    test_files: list[Path],
    device: torch.device,
) -> dict:

    model_dir = OUTPUT_ROOT / model_name
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    final_json = (
        model_dir
        / "best_weight_test_results.json"
    )

    if (
        SKIP_COMPLETED_MODELS
        and final_json.is_file()
    ):
        print(
            f"\n{model_name}: completed result already exists. "
            f"Skipping this model."
        )

        return json.loads(
            final_json.read_text(
                encoding="utf-8"
            )
        )

    seed_everything(SEED)

    checkpoint_path = (
        model_dir
        / f"best_{model_name}.pt"
    )

    history_path = (
        model_dir
        / "training_history.csv"
    )

    # Initialize all large references so cleanup is safe even if an error occurs.
    train_set = None
    val_set = None
    test_set = None
    train_loader = None
    val_loader = None
    test_loader = None
    model = None
    optimizer = None
    scheduler = None
    scaler = None
    best_state = None

    try:
        (
            train_set,
            val_set,
            test_set,
            train_loader,
            val_loader,
            test_loader,
        ) = create_data(
            train_files,
            val_files,
            test_files,
            device,
        )

        model = build_model(
            model_name
        ).to(device)

        total_parameters = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_parameters = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print("\n" + "=" * 110)
        print(f"TRAINING {model_name.upper()}")
        print("=" * 110)
        print(
            f"Parameters: {total_parameters:,} total, "
            f"{trainable_parameters:,} trainable"
        )
        print(
            f"Train={len(train_set)}, "
            f"Val={len(val_set)}, "
            f"Test={len(test_set)}"
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=STEP_SIZE,
            gamma=GAMMA,
        )

        amp_enabled = (
            USE_AMP
            and device.type == "cuda"
        )

        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

        best_val_mse = math.inf
        best_epoch = 0
        stale_epochs = 0

        train_start = time.time()

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as stream:

            writer = csv.writer(stream)

            writer.writerow(
                [
                    "epoch",
                    "train_mse",
                    "val_mse",
                    "learning_rate",
                    "is_best",
                    "epochs_without_improvement",
                ]
            )

            for epoch in range(
                1,
                MAX_EPOCHS + 1,
            ):
                model.train()

                train_sum = 0.0
                train_count = 0

                for step, (images, _) in enumerate(
                    train_loader,
                    start=1,
                ):
                    images = images.to(
                        device,
                        non_blocking=True,
                    )

                    optimizer.zero_grad(
                        set_to_none=True
                    )

                    try:
                        with torch.autocast(
                            device_type=device.type,
                            enabled=amp_enabled,
                        ):
                            reconstruction = model(
                                images
                            )

                            loss = training_loss(
                                reconstruction,
                                images,
                            )

                        scaler.scale(
                            loss
                        ).backward()

                        scaler.step(
                            optimizer
                        )

                        scaler.update()

                    except torch.cuda.OutOfMemoryError:
                        optimizer.zero_grad(
                            set_to_none=True
                        )

                        print(
                            "\nCUDA OUT OF MEMORY. "
                            "Batch size remains fixed at 80; "
                            "the script does not silently change it."
                        )

                        raise

                    batch_n = images.shape[0]

                    train_sum += (
                        loss.item()
                        * batch_n
                    )

                    train_count += batch_n

                    if (
                        step % 50 == 0
                        or step == len(train_loader)
                    ):
                        print(
                            f"{model_name:15s} | "
                            f"epoch={epoch:02d}/{MAX_EPOCHS:02d} | "
                            f"step={step:04d}/{len(train_loader):04d} | "
                            f"batch_mse={loss.item():.8f}"
                        )

                train_mse = (
                    train_sum
                    / max(train_count, 1)
                )

                val_mse = evaluate_mse(
                    model,
                    val_loader,
                    device,
                    amp_enabled,
                )

                current_lr = (
                    optimizer.param_groups[0]["lr"]
                )

                is_best = (
                    val_mse
                    < best_val_mse
                    - MIN_DELTA
                )

                if is_best:
                    best_val_mse = val_mse
                    best_epoch = epoch
                    stale_epochs = 0

                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "model": model_name,
                            "best_epoch": best_epoch,
                            "best_val_mse": best_val_mse,
                            "max_epochs": MAX_EPOCHS,
                            "batch_size": BATCH_SIZE,
                            "learning_rate": LEARNING_RATE,
                            "weight_decay": WEIGHT_DECAY,
                            "step_size": STEP_SIZE,
                            "gamma": GAMMA,
                            "patience": EARLY_STOPPING_PATIENCE,
                            "min_delta": MIN_DELTA,
                            "min_value": MIN_VALUE,
                            "max_value": MAX_VALUE,
                            "seed": SEED,
                        },
                        checkpoint_path,
                    )

                    print(
                        f"Saved BEST checkpoint: {checkpoint_path}"
                    )

                else:
                    stale_epochs += 1

                writer.writerow(
                    [
                        epoch,
                        train_mse,
                        val_mse,
                        current_lr,
                        int(is_best),
                        stale_epochs,
                    ]
                )

                stream.flush()

                print(
                    f"{model_name:15s} | "
                    f"epoch={epoch:02d} | "
                    f"train_mse={train_mse:.8f} | "
                    f"val_mse={val_mse:.8f} | "
                    f"lr={current_lr:.3e} | "
                    f"best_epoch={best_epoch} | "
                    f"no_improve={stale_epochs}/{EARLY_STOPPING_PATIENCE}"
                )

                if (
                    stale_epochs
                    >= EARLY_STOPPING_PATIENCE
                ):
                    print(
                        f"EARLY STOPPING for {model_name}: "
                        f"{EARLY_STOPPING_PATIENCE} consecutive epochs "
                        f"without validation improvement."
                    )
                    break

                scheduler.step()

        training_seconds = (
            time.time()
            - train_start
        )

        # ---------------------------------------------------------------------
        # LOAD BEST WEIGHTS
        # ---------------------------------------------------------------------

        if not checkpoint_path.is_file():
            raise RuntimeError(
                f"No checkpoint saved for {model_name}"
            )

        best_state = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(
            best_state["model_state"]
        )

        print("\n" + "-" * 110)
        print(f"{model_name.upper()} BEST CHECKPOINT")
        print("-" * 110)
        print(
            f"Best epoch   : {best_state['best_epoch']}"
        )
        print(
            f"Best val MSE : {best_state['best_val_mse']:.10f}"
        )
        print("-" * 110)

        # ---------------------------------------------------------------------
        # COMPLETE TEST SET WITH BEST WEIGHTS
        # ---------------------------------------------------------------------

        test_start = time.time()

        test_metrics = evaluate_complete_test_set(
            model_name,
            model,
            test_loader,
            test_set,
            device,
            amp_enabled,
            model_dir,
        )

        test_seconds = (
            time.time()
            - test_start
        )

        result = {
            "model": model_name,
            "architecture_detail": (
                "MiT-B0 / SegFormer-B0"
                if model_name == "segformer"
                else f"{model_name} + {CNN_ENCODER}"
            ),
            "pretrained": False,
            "parameters": int(total_parameters),
            "trainable_parameters": int(trainable_parameters),
            "train_samples": int(len(train_set)),
            "validation_samples": int(len(val_set)),
            "test_samples": int(test_metrics["test_samples"]),
            "max_epochs": MAX_EPOCHS,
            "best_epoch": int(best_state["best_epoch"]),
            "best_val_mse": float(best_state["best_val_mse"]),
            "test_mse": float(test_metrics["test_mse"]),
            "test_rmse": float(test_metrics["test_rmse"]),
            "test_vv_mse": float(test_metrics["test_vv_mse"]),
            "test_vh_mse": float(test_metrics["test_vh_mse"]),
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "loss": "pixel-wise MSE",
            "scheduler": "StepLR",
            "step_size": STEP_SIZE,
            "gamma": GAMMA,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_delta": MIN_DELTA,
            "normalization": "(x_db + 15) / 12; no clipping",
            "output_activation": "sigmoid",
            "seed": SEED,
            "amp": amp_enabled,
            "checkpoint": str(checkpoint_path),
            "per_sample_test_csv": test_metrics["per_sample_test_csv"],
            "training_seconds": float(training_seconds),
            "test_seconds": float(test_seconds),
        }

        final_json.write_text(
            json.dumps(
                result,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("\n" + "=" * 110)
        print(f"{model_name.upper()} BEST-WEIGHT TEST RESULTS")
        print("=" * 110)
        print(
            f"Best epoch  : {result['best_epoch']}"
        )
        print(
            f"Best val MSE: {result['best_val_mse']:.10f}"
        )
        print(
            f"Test MSE    : {result['test_mse']:.10f}"
        )
        print(
            f"Test RMSE   : {result['test_rmse']:.10f}"
        )
        print(
            f"VV test MSE : {result['test_vv_mse']:.10f}"
        )
        print(
            f"VH test MSE : {result['test_vh_mse']:.10f}"
        )
        print("=" * 110)

        return result

    finally:
        print(
            f"\nReleasing CPU/GPU resources after {model_name}..."
        )

        # Drop large references.
        best_state = None
        model = None
        optimizer = None
        scheduler = None
        scaler = None

        train_loader = None
        val_loader = None
        test_loader = None

        train_set = None
        val_set = None
        test_set = None

        gc.collect()
        clear_resources(model_name)


# =============================================================================
# FINAL COMPARISON
# =============================================================================

def write_comparison(results: list[dict]) -> tuple[Path, Path]:
    csv_path = (
        OUTPUT_ROOT
        / "four_models_best_weight_test_comparison.csv"
    )

    json_path = (
        OUTPUT_ROOT
        / "four_models_best_weight_test_comparison.json"
    )

    fields = [
        "model",
        "architecture_detail",
        "pretrained",
        "parameters",
        "trainable_parameters",
        "train_samples",
        "validation_samples",
        "test_samples",
        "max_epochs",
        "best_epoch",
        "best_val_mse",
        "test_mse",
        "test_rmse",
        "test_vv_mse",
        "test_vh_mse",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "optimizer",
        "loss",
        "scheduler",
        "step_size",
        "gamma",
        "early_stopping_patience",
        "min_delta",
        "normalization",
        "output_activation",
        "seed",
        "amp",
        "checkpoint",
        "per_sample_test_csv",
        "training_seconds",
        "test_seconds",
    ]

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(results)

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
        ),
        encoding="utf-8",
    )

    return csv_path, json_path


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 110)
    print("4-MODEL SEQUENTIAL RECONSTRUCTION EXPERIMENT")
    print("=" * 110)
    print(f"Device      : {device}")

    if device.type == "cuda":
        print(
            f"GPU         : {torch.cuda.get_device_name(0)}"
        )

        free_bytes, total_bytes = (
            torch.cuda.mem_get_info()
        )

        print(
            f"GPU memory  : "
            f"{free_bytes / (1024**3):.2f} GB free / "
            f"{total_bytes / (1024**3):.2f} GB total"
        )

    print(f"Data root   : {DATA_ROOT}")
    print(f"Output root : {OUTPUT_ROOT}")
    print(f"Models      : {', '.join(MODEL_ORDER)}")
    print(f"Max epochs  : {MAX_EPOCHS}")
    print(
        f"Early stop  : patience={EARLY_STOPPING_PATIENCE}"
    )
    print(f"Batch size  : {BATCH_SIZE}")
    print("=" * 110)

    disk = shutil.disk_usage(
        OUTPUT_ROOT
    )

    print(
        f"Free disk space at output location: "
        f"{disk.free / (1024**3):.2f} GB"
    )

    train_files = find_tiffs(
        TRAIN_DIR
    )

    val_files = find_tiffs(
        VAL_DIR
    )

    test_files = find_tiffs(
        TEST_DIR
    )

    print(
        f"Dataset counts: "
        f"train={len(train_files)}, "
        f"val={len(val_files)}, "
        f"test={len(test_files)}"
    )

    # Strict comparison: fail if the local copy is incomplete/different.
    if len(train_files) != EXPECTED_TRAIN:
        raise RuntimeError(
            f"Expected {EXPECTED_TRAIN} train TIFFs, "
            f"found {len(train_files)}"
        )

    if len(val_files) != EXPECTED_VAL:
        raise RuntimeError(
            f"Expected {EXPECTED_VAL} validation TIFFs, "
            f"found {len(val_files)}"
        )

    if len(test_files) != EXPECTED_TEST:
        raise RuntimeError(
            f"Expected {EXPECTED_TEST} test TIFFs, "
            f"found {len(test_files)}"
        )

    if VALIDATE_TIFF_HEADERS:
        validate_headers(
            train_files,
            "train",
        )

        validate_headers(
            val_files,
            "validation",
        )

        validate_headers(
            test_files,
            "test",
        )

    results: list[dict] = []

    experiment_start = time.time()

    for index, model_name in enumerate(
        MODEL_ORDER,
        start=1,
    ):
        print("\n" + "#" * 110)
        print(
            f"MODEL {index}/{len(MODEL_ORDER)}: "
            f"{model_name.upper()}"
        )
        print("#" * 110)

        # Cleanup barrier before constructing every architecture.
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result = run_model(
            model_name,
            train_files,
            val_files,
            test_files,
            device,
        )

        results.append(result)

        # Cleanup barrier after every architecture.
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_seconds = (
        time.time()
        - experiment_start
    )

    csv_path, json_path = write_comparison(
        results
    )

    print("\n" + "=" * 110)
    print("FINAL BEST-WEIGHT TEST COMPARISON")
    print("=" * 110)

    for result in results:
        print(
            f"{result['model']:16s} | "
            f"best_epoch={result['best_epoch']:02d} | "
            f"val_MSE={result['best_val_mse']:.10f} | "
            f"test_MSE={result['test_mse']:.10f} | "
            f"test_RMSE={result['test_rmse']:.10f}"
        )

    print("=" * 110)
    print(
        f"Total experiment time: "
        f"{total_seconds / 3600:.2f} h"
    )
    print(f"Comparison CSV : {csv_path}")
    print(f"Comparison JSON: {json_path}")
    print(f"Results root   : {OUTPUT_ROOT}")
    print("=" * 110)


if __name__ == "__main__":
    main()
