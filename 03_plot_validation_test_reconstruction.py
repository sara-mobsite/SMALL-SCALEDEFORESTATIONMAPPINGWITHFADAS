#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot representative validation and test reconstruction examples for FADAS U-Net.

Figure layout:
    Row 1: Validation VV | Validation VH | Validation reconstruction error
    Row 2: Test VV       | Test VH       | Test reconstruction error

The validation and test examples are cropped to the same spatial pixel size.
No resizing/interpolation is applied to the SAR data.

Path configuration
------------------
TRAINING_SCRIPT:
    Path to 01_train_reconstruction_models.py. The script imports the exact
    SARDataset, normalization constants, model definition, and TIFF reader.

DATA_ROOT:
    External dataset root. Expected:
        DATA_ROOT/validation/processed/
        DATA_ROOT/test/processed/

RESULTS_ROOT:
    Directory containing the trained U-Net checkpoint:
        RESULTS_ROOT/unet/best_unet.pt

METRICS_ROOT:
    Directory containing 622_975_2022.shp. This is only used to prefer a test
    example intersecting the 2022 reference-deforestation polygons.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box
import torch
from torch.utils.data import DataLoader


# =============================================================================
# PATHS
# =============================================================================

TRAINING_SCRIPT = Path(__file__).resolve().parent / "01_train_reconstruction_models.py"

# Dataset root from the external SAR dataset repository.
DATA_ROOT = Path("PATH/TO/DATASET/forest")

SOURCE_DIRS = {
    "validation": DATA_ROOT / "validation" / "processed",
    "test": DATA_ROOT / "test" / "processed",
}

# Output root produced by 01_train_reconstruction_models.py.
RESULTS_ROOT = Path("PATH/TO/RESULTS/four_models_reconstruction_50epochs")

# Reference-vector directory. Must contain 622_975_2022.shp if automatic
# reference-overlap selection is desired.
METRICS_ROOT = Path("PATH/TO/DATASET/metrics")

MODEL_NAME = "unet"
CHECKPOINT = RESULTS_ROOT / MODEL_NAME / f"best_{MODEL_NAME}.pt"

OUTPUT_DIR = RESULTS_ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURE = OUTPUT_DIR / "fadas_unet_validation_test_vv_vh_error_same_size.png"

# Optional filename fragments to force a particular example.
# Leave as None for automatic selection.
MANUAL_FILE_CONTAINS = {
    "validation": None,
    "test": None,
}

BATCH_SIZE = 64
NUM_WORKERS = 2
USE_AMP = True
TOP_K_TEST = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# IMPORT EXACT TRAINING DEFINITIONS
# =============================================================================

def import_training_module():
    if not TRAINING_SCRIPT.is_file():
        raise FileNotFoundError(f"Training script not found: {TRAINING_SCRIPT}")
    spec = importlib.util.spec_from_file_location("reconstruction_training", TRAINING_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import: {TRAINING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


training = import_training_module()

if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        "U-Net checkpoint not found. Train the models first or copy the published "
        f"checkpoint to: {CHECKPOINT}"
    )

model = training.build_model(MODEL_NAME).to(DEVICE)
state = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(state["model_state"])
model.eval()


# =============================================================================
# GEOREFERENCE / REFERENCE DATA
# =============================================================================

def read_georef(path: Path):
    with rasterio.open(path) as src:
        return src.transform, src.crs, src.bounds, src.height, src.width


reference_path = METRICS_ROOT / "622_975_2022.shp"
REFERENCE_GDF = gpd.read_file(reference_path) if reference_path.is_file() else None


def intersects_reference(path: Path) -> bool:
    if REFERENCE_GDF is None:
        return False
    _, crs, bounds, _, _ = read_georef(path)
    if crs is None:
        return False
    ref = REFERENCE_GDF.to_crs(crs)
    geom = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    return bool(ref.intersects(geom).any())


# =============================================================================
# AUTOMATIC SAMPLE SELECTION
# =============================================================================

@torch.no_grad()
def score_split(split_name: str):
    files = training.find_tiffs(SOURCE_DIRS[split_name])
    dataset = training.SARDataset(files, training.MIN_VALUE, training.MAX_VALUE)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=False,
        drop_last=False,
    )
    amp_enabled = USE_AMP and DEVICE.type == "cuda"
    results = []

    for images, paths in loader:
        images = images.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, enabled=amp_enabled):
            reconstruction = model(images)
            error = (reconstruction - images).square().mean(dim=1)

        for error_map, path_string in zip(error.float().cpu().numpy(), paths):
            results.append({
                "path": Path(path_string),
                "score": float(np.percentile(error_map, 99.0)),
            })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results


def find_manual_patch(split_name: str):
    fragment = MANUAL_FILE_CONTAINS[split_name]
    if fragment is None:
        return None
    files = training.find_tiffs(SOURCE_DIRS[split_name])
    matches = [Path(p) for p in files if fragment in Path(p).name]
    if not matches:
        raise FileNotFoundError(f"No {split_name} file contains: {fragment}")
    return matches[0]


selected_paths = {}

for split_name in ("validation", "test"):
    manual = find_manual_patch(split_name)
    if manual is not None:
        selected_paths[split_name] = manual
        continue

    ranked = score_split(split_name)
    if not ranked:
        raise RuntimeError(f"No samples found for {split_name}")

    if split_name == "test" and REFERENCE_GDF is not None:
        selected = next(
            (item["path"] for item in ranked[:TOP_K_TEST]
             if intersects_reference(item["path"])),
            ranked[0]["path"],
        )
    else:
        selected = ranked[0]["path"]

    selected_paths[split_name] = selected


# =============================================================================
# RUN U-NET AND RECOVER INPUT IN dB
# =============================================================================

@torch.no_grad()
def process_one_patch(path: Path):
    dataset = training.SARDataset([path], training.MIN_VALUE, training.MAX_VALUE)
    item = dataset[0]
    image = item[0] if isinstance(item, (tuple, list)) else item

    image_batch = image.unsqueeze(0).to(DEVICE)
    amp_enabled = USE_AMP and DEVICE.type == "cuda"

    with torch.autocast(device_type=DEVICE.type, enabled=amp_enabled):
        reconstruction = model(image_batch)
        error = (reconstruction - image_batch).square().mean(dim=1)

    image_normalized = image.float().cpu().numpy()
    error_map = error[0].float().cpu().numpy()

    minimum = float(training.MIN_VALUE)
    maximum = float(training.MAX_VALUE)
    image_db = image_normalized * (maximum - minimum) + minimum

    transform, crs, bounds, raster_h, raster_w = read_georef(path)

    return {
        "path": path,
        "db": image_db,
        "error": error_map,
        "transform": transform,
        "crs": crs,
        "bounds": bounds,
        "raster_h": raster_h,
        "raster_w": raster_w,
    }


plot_data = {
    split: process_one_patch(selected_paths[split])
    for split in ("validation", "test")
}


# =============================================================================
# CROP BOTH TO THE SAME PIXEL SIZE
# =============================================================================

sizes = {
    split: plot_data[split]["db"].shape[1:]
    for split in ("validation", "test")
}
COMMON_H = min(h for h, _ in sizes.values())
COMMON_W = min(w for _, w in sizes.values())


def crop_to_common_size(sample, target_h, target_w):
    image = sample["db"]
    error = sample["error"]
    _, h, w = image.shape

    y0 = (h - target_h) // 2
    x0 = (w - target_w) // 2
    y1 = y0 + target_h
    x1 = x0 + target_w

    raster_h = sample["raster_h"]
    raster_w = sample["raster_w"]

    window = Window(
        col_off=x0 / w * raster_w,
        row_off=y0 / h * raster_h,
        width=target_w / w * raster_w,
        height=target_h / h * raster_h,
    )

    left, bottom, right, top = window_bounds(window, sample["transform"])

    result = sample.copy()
    result["db"] = image[:, y0:y1, x0:x1]
    result["error"] = error[y0:y1, x0:x1]
    result["bounds"] = rasterio.coords.BoundingBox(left, bottom, right, top)
    return result


for split in ("validation", "test"):
    plot_data[split] = crop_to_common_size(plot_data[split], COMMON_H, COMMON_W)


# =============================================================================
# SHARED DISPLAY SCALES
# =============================================================================

def robust_limits(array, low=2.0, high=98.0):
    values = array[np.isfinite(array)]
    lo = float(np.percentile(values, low))
    hi = float(np.percentile(values, high))
    if hi <= lo:
        hi = lo + 1e-6
    return lo, hi


all_vv = np.concatenate([plot_data[s]["db"][0].ravel() for s in ("validation", "test")])
all_vh = np.concatenate([plot_data[s]["db"][1].ravel() for s in ("validation", "test")])
all_error = np.concatenate([plot_data[s]["error"].ravel() for s in ("validation", "test")])

VV_LOW, VV_HIGH = robust_limits(all_vv, 2, 98)
VH_LOW, VH_HIGH = robust_limits(all_vh, 2, 98)
ERROR_LOW, ERROR_HIGH = robust_limits(all_error, 1, 99)


# =============================================================================
# SCALE BAR
# =============================================================================

def patch_width_metres(bounds, crs):
    width = float(bounds.right - bounds.left)
    if crs is not None and crs.is_geographic:
        latitude = (bounds.top + bounds.bottom) / 2.0
        width *= 111320.0 * math.cos(math.radians(latitude))
    return width


def choose_scale_bar(width_m):
    candidates = [
        (5000, "5 km"), (2000, "2 km"), (1000, "1 km"),
        (500, "500 m"), (200, "200 m"), (100, "100 m"), (50, "50 m"),
    ]
    for distance, label in candidates:
        fraction = distance / width_m
        if 0.15 <= fraction <= 0.35:
            return distance, label
    return 50, "50 m"


def add_scale_bar(ax, bounds, crs):
    width_m = patch_width_metres(bounds, crs)
    if width_m <= 0:
        return
    scale_m, label = choose_scale_bar(width_m)
    fraction = min(max(scale_m / width_m, 0.10), 0.35)
    x0, y = 0.07, 0.075
    x1 = x0 + fraction

    ax.plot([x0, x1], [y, y], transform=ax.transAxes,
            color="black", linewidth=6, solid_capstyle="butt", zorder=20)
    ax.plot([x0, x1], [y, y], transform=ax.transAxes,
            color="white", linewidth=3, solid_capstyle="butt", zorder=21)
    ax.text((x0 + x1) / 2, y + 0.025, label, transform=ax.transAxes,
            color="white", fontsize=12, fontweight="bold",
            ha="center", va="bottom", zorder=22,
            bbox=dict(facecolor="black", edgecolor="none", alpha=0.35, pad=1.2))


# =============================================================================
# PLOT: 2 ROWS x 3 COLUMNS
# =============================================================================

plt.rcParams.update({"font.size": 14, "axes.titlesize": 16})

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
row_labels = {"validation": "Validation", "test": "Test"}
last_error = None

for row, split in enumerate(("validation", "test")):
    data = plot_data[split]

    ax = axes[row, 0]
    ax.imshow(data["db"][0], cmap="gray", vmin=VV_LOW, vmax=VV_HIGH,
              interpolation="nearest", aspect="equal")
    ax.set_title(f"{row_labels[split]} — Sentinel-1 VV")
    add_scale_bar(ax, data["bounds"], data["crs"])
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[row, 1]
    ax.imshow(data["db"][1], cmap="gray", vmin=VH_LOW, vmax=VH_HIGH,
              interpolation="nearest", aspect="equal")
    ax.set_title(f"{row_labels[split]} — Sentinel-1 VH")
    add_scale_bar(ax, data["bounds"], data["crs"])
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[row, 2]
    last_error = ax.imshow(
        data["error"], cmap="magma",
        vmin=ERROR_LOW, vmax=ERROR_HIGH,
        interpolation="nearest", aspect="equal",
    )
    ax.set_title(f"{row_labels[split]} — Reconstruction error")
    add_scale_bar(ax, data["bounds"], data["crs"])
    ax.set_xticks([]); ax.set_yticks([])

fig.subplots_adjust(left=0.04, right=0.90, top=0.92, bottom=0.06,
                    wspace=0.08, hspace=0.10)

# Dedicated colorbar axis outside the image grid, avoiding overlap.
cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
cbar = fig.colorbar(last_error, cax=cax)
cbar.set_label("Reconstruction error", fontsize=15)

fig.suptitle(
    "FADAS (U-Net): Sentinel-1 VV/VH and reconstruction error",
    fontsize=20, fontweight="bold", y=0.97,
)

fig.savefig(OUTPUT_FIGURE, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print("Validation:", selected_paths["validation"])
print("Test:", selected_paths["test"])
print("Common plotted size:", COMMON_H, "x", COMMON_W)
print("Saved:", OUTPUT_FIGURE)
