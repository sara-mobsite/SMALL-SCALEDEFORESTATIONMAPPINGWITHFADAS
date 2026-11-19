#!/usr/bin/env python3
"""
threshold_sweep_all1519_global_0p5pct_to_90pct_single_line.py

CORRECTED GLOBAL-CALIBRATION THRESHOLD ANALYSIS

Main correction
---------------
The previous pairwise implementation fitted/calibrated only from the 49 April
tiles + 49 September tiles of each year.

This version uses ALL 1519 TEST TILES to calibrate EACH MODEL.

For every model:
    1. Read all 1519 reconstruction-error maps.
    2. Take a deterministic uniform pixel sample from EVERY tile.
       Every one of the 1519 tiles contributes to the calibration.
    3. Fit ONE global 2-component GMM per model from this all-test calibration.
    4. Compute ONE global robust P1-P99 reconstruction-error scale per model.
    5. Keep those SAME model-level calibrations for BOTH 2021 and 2022.
    6. Only the actual April -> September change computation is restricted to
       the 49 common spatial tile pairs required for each yearly comparison.

Threshold sweep
---------------
Thresholds:
    0.005, 0.01, 0.02, ..., 0.10, 0.20, 0.30, ..., 0.90

Plotting:
    IoU-only figures use a broken x-axis so 0.5--10% remains readable while
    20--90% is shown in a separate numeric segment.

A) GMM posterior method:
    anomaly = P(GMM anomaly component | reconstruction error) >= threshold

B) Direct method, NO GMM:
    global robust score =
        clip((error - global_P1) / (global_P99 - global_P1), 0, 1)

    anomaly = global robust score >= threshold

The direct normalization is calibrated ONCE per model using all 1519 test tiles,
not separately per year. Therefore the same threshold has the same raw-error
meaning for 2021 and 2022 for a given model.

Evaluation
----------
For each model/year/threshold:
    - remove anomaly connected components smaller than 100 pixels;
    - deforestation = April NON-anomaly AND September anomaly;
    - remove change components smaller than 100 pixels;
    - evaluate on the same fixed 256x256 autoencoder support;
    - forest-mask prediction only, not annotation truth;
    - compute:
        Precision, Recall, F1, IoU, Accuracy, TP, FP, FN, TN.

Scientific note
---------------
The threshold sweep is a sensitivity analysis if the same labeled test region
is used to inspect/select the best threshold. This script deliberately DOES NOT
create a "best observed" model-comparison plot, because selecting the best test
threshold and then presenting it as the final unbiased test result would be
test-set tuning.

Outputs
-------
PATH/TO/RESULTS/four_models_reconstruction_50epochs/
    threshold_sweep_all1519_global_0p5pct_to_90pct/

The script reuses the 1519 per-image reconstruction error maps in:
    <model>/test_maps/

If missing, it regenerates them from the best model checkpoint.
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import re
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from scipy import ndimage
from shapely.geometry import shape
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader


# =============================================================================
# PATHS
# =============================================================================

# Path to script 01 in this repository. It is imported so this analysis uses
# exactly the same model definitions, normalization, and SARDataset class.
TRAINING_SCRIPT = Path(__file__).resolve().parent / "01_train_reconstruction_models.py"

# DATA_ROOT must point to the external SAR dataset "forest" directory.
# Only test/processed is used by this threshold analysis.
DATA_ROOT = Path("PATH/TO/DATASET/forest")
TEST_DIR = DATA_ROOT / "test" / "processed"

# RESULTS_ROOT must be the output directory produced by script 01.
# Expected model subdirectories:
#   segformer/, unet/, pspnet/, deeplabv3plus/
# containing best checkpoints and/or test_maps reconstruction-error rasters.
RESULTS_ROOT = Path("PATH/TO/RESULTS/four_models_reconstruction_50epochs")

OUTPUT_ROOT = (
    RESULTS_ROOT
    / "threshold_sweep_all1519_global_0p5pct_to_90pct"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# MODELS / YEARS / THRESHOLDS
# =============================================================================

MODELS = (
    "segformer",
    "unet",
    "pspnet",
    "deeplabv3plus",
)

MODEL_LABELS = {
    "segformer": "SegFormer",
    "unet": "UNet",
    "pspnet": "PSPNet",
    "deeplabv3plus": "DeepLabV3+",
}

YEAR_CONFIG = {
    2021: {
        "previous_date": "20210409",
        "target_date": "20210912",
    },
    2022: {
        "previous_date": "20220404",
        "target_date": "20220919",
    },
}

# Sensitivity analysis thresholds:
#   0.5%, 1%, 2%, ..., 10%, then 20%, 30%, ..., 90%.
# Values are probabilities/scores in [0, 1].
THRESHOLDS = np.array(
    [0.005]
    + [value / 100.0 for value in range(1, 11)]
    + [value / 100.0 for value in range(20, 100, 10)],
    dtype=np.float64,
)


# =============================================================================
# EXPERIMENT SETTINGS
# =============================================================================

EXPECTED_TEST_TILES = 1519

BATCH_SIZE = 80
NUM_WORKERS = 2
USE_AMP = True

GMM_SEED = 42

# Every one of the 1519 tiles contributes this many randomly selected pixels.
# 1519 * 1024 = 1,555,456 GMM calibration samples per model.
#
# This is much safer computationally than asking sklearn GaussianMixture to
# allocate responsibilities for ~99.5 million pixels at once, while still
# guaranteeing that ALL 1519 test tiles participate equally.
PIXELS_PER_TILE_FOR_GLOBAL_CALIBRATION = 1024

MIN_COMPONENT_SIZE = 100

ROBUST_LOW_PERCENTILE = 1.0
ROBUST_HIGH_PERCENTILE = 99.0

EVAL_WIDTH = 256
EVAL_HEIGHT = 256


# =============================================================================
# ERROR-MAP FILENAME PARSING
# =============================================================================

DATE_RE = re.compile(
    r"D_(\d{8})T"
)

TILE_RE = re.compile(
    r"_db_(\d+)_(\d+)_fused_error\.tif$"
)


# =============================================================================
# GENERAL HELPERS
# =============================================================================

def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def require_dir(
    path: Path,
    label: str,
) -> None:
    if not path.is_dir():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def cleanup_gpu(
    label: str = "",
) -> None:
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass

        free_bytes, total_bytes = (
            torch.cuda.mem_get_info()
        )

        print(
            f"GPU {label}: "
            f"{free_bytes / (1024**3):.2f} GB free / "
            f"{total_bytes / (1024**3):.2f} GB total"
        )


# =============================================================================
# LOCATE EVALUATION SHAPEFILES
# =============================================================================

REQUIRED_METRIC_FILES = (
    "Forest_Mask_2020.shp",
    "Forest_Mask_2021.shp",
    "622_975_2021.shp",
    "622_975_2022.shp",
    "anomaly_difference_20210912.shp",
    "anomaly_difference_20220919.shp",
)


def directory_has_all_metrics(
    directory: Path,
) -> bool:
    return all(
        (
            directory
            / filename
        ).is_file()
        for filename
        in REQUIRED_METRIC_FILES
    )


# METRICS_ROOT must point to the directory containing the reference shapefiles
# used for yearly deforestation evaluation. Required files are listed in
# REQUIRED_METRIC_FILES above.
METRICS_ROOT = Path("PATH/TO/DATASET/metrics")

if not directory_has_all_metrics(METRICS_ROOT):
    missing = [
        name for name in REQUIRED_METRIC_FILES
        if not (METRICS_ROOT / name).is_file()
    ]
    raise FileNotFoundError(
        "METRICS_ROOT does not contain all required evaluation shapefiles. "
        f"Missing: {missing}. Current METRICS_ROOT={METRICS_ROOT}"
    )

print(
    "Metrics root:",
    METRICS_ROOT,
)


# =============================================================================
# IMPORT EXACT MODEL / DATASET DEFINITIONS
# =============================================================================

def import_training_module():
    require_file(
        TRAINING_SCRIPT,
        "Training script",
    )

    spec = (
        importlib.util
        .spec_from_file_location(
            "reconstruction_training",
            TRAINING_SCRIPT,
        )
    )

    if (
        spec is None
        or spec.loader is None
    ):
        raise RuntimeError(
            f"Cannot import {TRAINING_SCRIPT}"
        )

    module = (
        importlib.util
        .module_from_spec(
            spec
        )
    )

    spec.loader.exec_module(
        module
    )

    return module


training = import_training_module()


# =============================================================================
# GENERATE / REUSE 1519 RECONSTRUCTION-ERROR MAPS
# =============================================================================

def write_error_tif(
    output_path: Path,
    error: np.ndarray,
    source_path: Path,
) -> None:
    with rasterio.open(
        source_path
    ) as src:
        profile = (
            src.profile.copy()
        )

    profile.pop(
        "photometric",
        None,
    )

    profile.pop(
        "interleave",
        None,
    )

    profile.update(
        driver="GTiff",
        count=1,
        dtype="float32",
        height=error.shape[0],
        width=error.shape[1],
        compress="deflate",
        nodata=None,
    )

    with rasterio.open(
        output_path,
        "w",
        **profile,
    ) as dst:
        dst.write(
            error.astype(
                np.float32,
                copy=False,
            ),
            1,
        )


@torch.no_grad()
def ensure_error_maps(
    model_name: str,
    device: torch.device,
) -> Path:
    model_dir = (
        RESULTS_ROOT
        / model_name
    )

    checkpoint = (
        model_dir
        / f"best_{model_name}.pt"
    )

    require_file(
        checkpoint,
        f"{model_name} checkpoint",
    )

    error_dir = (
        model_dir
        / "test_maps"
    )

    error_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = sorted(
        error_dir.glob(
            "*_error.tif"
        )
    )

    if (
        len(existing)
        == EXPECTED_TEST_TILES
    ):
        print(
            f"{model_name}: reusing "
            f"{EXPECTED_TEST_TILES} existing error maps."
        )

        return error_dir

    if existing:
        print(
            f"{model_name}: removing "
            f"{len(existing)} incomplete error maps."
        )

        for path in existing:
            path.unlink()

    test_files = (
        training.find_tiffs(
            TEST_DIR
        )
    )

    if (
        len(test_files)
        != EXPECTED_TEST_TILES
    ):
        raise RuntimeError(
            f"Expected {EXPECTED_TEST_TILES} test TIFFs, "
            f"found {len(test_files)}"
        )

    test_set = (
        training.SARDataset(
            test_files,
            training.MIN_VALUE,
            training.MAX_VALUE,
        )
    )

    loader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=False,
        drop_last=False,
    )

    model = (
        training.build_model(
            model_name
        )
        .to(
            device
        )
    )

    state = torch.load(
        checkpoint,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        state[
            "model_state"
        ]
    )

    model.eval()

    amp_enabled = (
        USE_AMP
        and device.type == "cuda"
    )

    print(
        f"\n{model_name}: "
        f"best_epoch={state.get('best_epoch')} | "
        f"best_val_mse={state.get('best_val_mse')}"
    )

    written = 0

    for images, paths in loader:
        images = images.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            enabled=amp_enabled,
        ):
            reconstruction = (
                model(images)
            )

            error = (
                reconstruction
                - images
            ).square().mean(
                dim=1
            )

        error_numpy = (
            error
            .float()
            .cpu()
            .numpy()
        )

        for (
            source_string,
            error_array,
        ) in zip(
            paths,
            error_numpy,
        ):
            source = Path(
                source_string
            )

            output = (
                error_dir
                / f"{source.stem}_error.tif"
            )

            write_error_tif(
                output,
                error_array,
                source,
            )

            written += 1

        if (
            written % 100
            < len(paths)
            or written
            == EXPECTED_TEST_TILES
        ):
            print(
                f"{model_name}: "
                f"{written}/"
                f"{EXPECTED_TEST_TILES}"
            )

    del state
    del model
    del loader
    del test_set

    cleanup_gpu(
        f"after {model_name} inference"
    )

    return error_dir


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device,
)

if device.type == "cuda":
    print(
        "GPU:",
        torch.cuda.get_device_name(
            0
        ),
    )

ERROR_DIRS = {}

for model_name in MODELS:
    ERROR_DIRS[
        model_name
    ] = ensure_error_maps(
        model_name,
        device,
    )

cleanup_gpu(
    "after error-map preparation"
)


# =============================================================================
# ERROR-MAP READERS
# =============================================================================

def read_error_sum(
    path: Path,
) -> np.ndarray:
    """
    Error maps store mean squared error across VV/VH.

    Multiply by 2 to preserve the previous VV+VH summed-error convention.
    """
    with rasterio.open(
        path
    ) as src:
        error_mean = (
            src.read(1)
            .astype(
                np.float32
            )
        )

    return (
        error_mean
        * 2.0
    )


def identify_error_map(
    path: Path,
):
    date_match = (
        DATE_RE.search(
            path.name
        )
    )

    tile_match = (
        TILE_RE.search(
            path.name
        )
    )

    if (
        date_match is None
        or tile_match is None
    ):
        return None

    return (
        date_match.group(1),
        (
            int(
                tile_match.group(1)
            ),
            int(
                tile_match.group(2)
            ),
        ),
    )


def collect_pairwise_error_maps(
    error_dir: Path,
) -> dict:
    result = {}

    for path in sorted(
        error_dir.glob(
            "*_error.tif"
        )
    ):
        key = (
            identify_error_map(
                path
            )
        )

        if key is not None:
            result[
                key
            ] = path

    return result


# =============================================================================
# GLOBAL CALIBRATION USING ALL 1519 TILES
# =============================================================================

def build_global_calibration_sample(
    error_dir: Path,
    model_name: str,
) -> np.ndarray:
    """
    Every one of the 1519 test tiles contributes exactly the same number of
    pixels to a single model-level calibration vector.

    This means the GMM is calibrated from ALL 1519 test tiles, rather than only
    the 49 + 49 images belonging to one yearly pair.
    """
    files = sorted(
        error_dir.glob(
            "*_error.tif"
        )
    )

    if (
        len(files)
        != EXPECTED_TEST_TILES
    ):
        raise RuntimeError(
            f"{model_name}: expected "
            f"{EXPECTED_TEST_TILES} error maps, "
            f"found {len(files)}"
        )

    rng = np.random.default_rng(
        GMM_SEED
    )

    sample_blocks = []

    print(
        f"\n{model_name}: building GLOBAL calibration from "
        f"ALL {len(files)} test tiles..."
    )

    for index, path in enumerate(
        files,
        start=1,
    ):
        error = read_error_sum(
            path
        )

        flat = (
            error.ravel()
        )

        if (
            PIXELS_PER_TILE_FOR_GLOBAL_CALIBRATION
            >= flat.size
        ):
            sampled = flat.copy()

        else:
            indices = rng.choice(
                flat.size,
                size=(
                    PIXELS_PER_TILE_FOR_GLOBAL_CALIBRATION
                ),
                replace=False,
            )

            sampled = flat[
                indices
            ]

        sample_blocks.append(
            sampled.astype(
                np.float32,
                copy=False,
            )
        )

        if (
            index % 200 == 0
            or index
            == len(files)
        ):
            print(
                f"  calibration tiles: "
                f"{index}/{len(files)}"
            )

    calibration = np.concatenate(
        sample_blocks
    )

    print(
        f"{model_name}: "
        f"{len(files)} / "
        f"{EXPECTED_TEST_TILES} tiles used"
    )

    print(
        f"{model_name}: "
        f"{calibration.size:,} "
        f"pixel samples for global calibration"
    )

    return calibration


def fit_model_global_calibration(
    model_name: str,
    error_dir: Path,
) -> dict:
    calibration = (
        build_global_calibration_sample(
            error_dir,
            model_name,
        )
    )

    # -------------------------------------------------------------------------
    # ONE GLOBAL GMM FOR THIS MODEL
    # -------------------------------------------------------------------------

    gmm = GaussianMixture(
        n_components=2,
        covariance_type="full",
        random_state=GMM_SEED,
    )

    gmm.fit(
        calibration.reshape(
            -1,
            1,
        )
    )

    means = (
        gmm.means_
        .ravel()
    )

    anomaly_component = int(
        np.argmax(
            means
        )
    )

    # -------------------------------------------------------------------------
    # ONE GLOBAL ROBUST ERROR NORMALIZATION FOR THIS MODEL
    # -------------------------------------------------------------------------

    robust_low = float(
        np.percentile(
            calibration,
            ROBUST_LOW_PERCENTILE,
        )
    )

    robust_high = float(
        np.percentile(
            calibration,
            ROBUST_HIGH_PERCENTILE,
        )
    )

    if (
        robust_high
        <= robust_low
    ):
        raise RuntimeError(
            f"{model_name}: invalid global robust range "
            f"{robust_low} -> {robust_high}"
        )

    raw_min = float(
        calibration.min()
    )

    raw_max = float(
        calibration.max()
    )

    # Posterior saturation diagnostic over the entire calibration sample.
    posterior = (
        gmm.predict_proba(
            calibration.reshape(
                -1,
                1,
            )
        )[
            :,
            anomaly_component,
        ]
    )

    posterior_percentiles = (
        np.percentile(
            posterior,
            [
                1,
                5,
                10,
                25,
                50,
                75,
                90,
                95,
                99,
            ],
        )
    )

    print(
        f"\n{model_name} GLOBAL GMM means: "
        f"{means.tolist()}"
    )

    print(
        f"{model_name} anomaly component: "
        f"{anomaly_component}"
    )

    print(
        f"{model_name} GLOBAL robust "
        f"P{ROBUST_LOW_PERCENTILE:g}/"
        f"P{ROBUST_HIGH_PERCENTILE:g}: "
        f"{robust_low:.10f} / "
        f"{robust_high:.10f}"
    )

    print(
        f"{model_name} posterior percentiles "
        f"[1,5,10,25,50,75,90,95,99]:\n"
        f"{posterior_percentiles}"
    )

    calibration_metadata = {
        "model": model_name,
        "tiles_used": (
            EXPECTED_TEST_TILES
        ),
        "pixels_per_tile": (
            PIXELS_PER_TILE_FOR_GLOBAL_CALIBRATION
        ),
        "calibration_pixel_count": int(
            calibration.size
        ),
        "gmm_means": (
            means.tolist()
        ),
        "gmm_anomaly_component": (
            anomaly_component
        ),
        "global_raw_min": (
            raw_min
        ),
        "global_raw_max": (
            raw_max
        ),
        "robust_low_percentile": (
            ROBUST_LOW_PERCENTILE
        ),
        "robust_high_percentile": (
            ROBUST_HIGH_PERCENTILE
        ),
        "global_robust_low": (
            robust_low
        ),
        "global_robust_high": (
            robust_high
        ),
        "posterior_percentiles": (
            posterior_percentiles.tolist()
        ),
    }

    (
        OUTPUT_ROOT
        / f"{model_name}_global_calibration.json"
    ).write_text(
        json.dumps(
            calibration_metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    # We no longer need the large calibration vector.
    del posterior
    del calibration

    gc.collect()

    return {
        "gmm": gmm,
        "means": means,
        "anomaly_component": (
            anomaly_component
        ),
        "robust_low": (
            robust_low
        ),
        "robust_high": (
            robust_high
        ),
        "metadata": (
            calibration_metadata
        ),
    }


GLOBAL_CALIBRATION = {}

for model_name in MODELS:
    GLOBAL_CALIBRATION[
        model_name
    ] = fit_model_global_calibration(
        model_name,
        ERROR_DIRS[
            model_name
        ],
    )


# =============================================================================
# YEARLY PAIR DATA
# =============================================================================

def load_year_pair_data(
    model_name: str,
    year: int,
) -> dict:
    files = (
        collect_pairwise_error_maps(
            ERROR_DIRS[
                model_name
            ]
        )
    )

    previous_date = (
        YEAR_CONFIG[
            year
        ][
            "previous_date"
        ]
    )

    target_date = (
        YEAR_CONFIG[
            year
        ][
            "target_date"
        ]
    )

    previous = {
        tile: path
        for (
            date,
            tile,
        ), path
        in files.items()
        if date
        == previous_date
    }

    target = {
        tile: path
        for (
            date,
            tile,
        ), path
        in files.items()
        if date
        == target_date
    }

    common = sorted(
        set(previous)
        & set(target)
    )

    if not common:
        available_dates = sorted(
            {
                date
                for date, _
                in files
            }
        )

        raise RuntimeError(
            f"{model_name} {year}: no common tiles. "
            f"Available dates: {available_dates}"
        )

    cached = {}
    output_crs = None

    for tile in common:
        previous_error = (
            read_error_sum(
                previous[
                    tile
                ]
            )
        )

        target_error = (
            read_error_sum(
                target[
                    tile
                ]
            )
        )

        if (
            previous_error.shape
            != target_error.shape
        ):
            raise ValueError(
                f"{model_name} {year} "
                f"tile {tile}: shape mismatch "
                f"{previous_error.shape} vs "
                f"{target_error.shape}"
            )

        with rasterio.open(
            target[
                tile
            ]
        ) as src:
            transform = (
                src.transform
            )

            output_crs = (
                src.crs
            )

        cached[
            tile
        ] = {
            "previous": (
                previous_error
            ),
            "target": (
                target_error
            ),
            "transform": (
                transform
            ),
        }

    print(
        f"{model_name} {year}: "
        f"{len(common)} common April/September "
        f"spatial pairs used for CHANGE DETECTION."
    )

    return {
        "common": common,
        "cached": cached,
        "crs": output_crs,
    }


# =============================================================================
# CONNECTED COMPONENTS / POLYGONIZATION
# =============================================================================

def filter_small_components(
    mask: np.ndarray,
    minimum: int = (
        MIN_COMPONENT_SIZE
    ),
) -> np.ndarray:
    labels, count = (
        ndimage.label(
            mask.astype(
                bool
            )
        )
    )

    if count == 0:
        return mask.astype(
            np.uint8
        )

    sizes = np.bincount(
        labels.ravel()
    )

    keep = (
        sizes >= minimum
    )

    keep[0] = False

    return keep[
        labels
    ].astype(
        np.uint8
    )


def polygons_from_mask(
    mask: np.ndarray,
    transform,
) -> list:
    polygons = []

    for (
        geometry,
        value,
    ) in shapes(
        mask.astype(
            np.uint8
        ),
        mask=mask.astype(
            bool
        ),
        transform=transform,
    ):
        if int(value) == 1:
            polygons.append(
                shape(
                    geometry
                )
            )

    return polygons


# =============================================================================
# FIXED-SUPPORT EVALUATION
# =============================================================================

def clean_gdf(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    invalid = (
        ~gdf.geometry.is_valid
    )

    if invalid.any():
        gdf.loc[
            invalid,
            "geometry",
        ] = (
            gdf.loc[
                invalid,
                "geometry",
            ]
            .buffer(0)
        )

    return gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
        & gdf.geometry.is_valid
    ].copy()


def read_layer(
    path: Path,
) -> gpd.GeoDataFrame:
    require_file(
        path,
        str(path),
    )

    gdf = gpd.read_file(
        path
    )

    if gdf.crs is None:
        raise ValueError(
            f"No CRS: {path}"
        )

    return clean_gdf(
        gdf
    )


def binary_raster(
    gdf: gpd.GeoDataFrame,
    transform,
) -> np.ndarray:
    geometries = [
        geom
        for geom
        in gdf.geometry
        if (
            geom is not None
            and not geom.is_empty
        )
    ]

    if not geometries:
        return np.zeros(
            (
                EVAL_HEIGHT,
                EVAL_WIDTH,
            ),
            dtype=np.uint8,
        )

    return rasterize(
        [
            (
                geom,
                1,
            )
            for geom
            in geometries
        ],
        out_shape=(
            EVAL_HEIGHT,
            EVAL_WIDTH,
        ),
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype=np.uint8,
    )


def evaluation_paths(
    year: int,
) -> dict:
    if year == 2021:
        return {
            "forest": (
                METRICS_ROOT
                / "Forest_Mask_2020.shp"
            ),
            "annotation": (
                METRICS_ROOT
                / "622_975_2021.shp"
            ),
            "support": (
                METRICS_ROOT
                / "anomaly_difference_20210912.shp"
            ),
        }

    if year == 2022:
        return {
            "forest": (
                METRICS_ROOT
                / "Forest_Mask_2021.shp"
            ),
            "annotation": (
                METRICS_ROOT
                / "622_975_2022.shp"
            ),
            "support": (
                METRICS_ROOT
                / "anomaly_difference_20220919.shp"
            ),
        }

    raise ValueError(
        year
    )


def prepare_evaluation_context(
    year: int,
) -> dict:
    paths = evaluation_paths(
        year
    )

    reference = read_layer(
        paths[
            "support"
        ]
    )

    forest = read_layer(
        paths[
            "forest"
        ]
    )

    annotation = read_layer(
        paths[
            "annotation"
        ]
    )

    crs = reference.crs

    forest = forest.to_crs(
        crs
    )

    annotation = annotation.to_crs(
        crs
    )

    (
        minx,
        miny,
        maxx,
        maxy,
    ) = reference.total_bounds

    transform = from_bounds(
        minx,
        miny,
        maxx,
        maxy,
        EVAL_WIDTH,
        EVAL_HEIGHT,
    )

    if hasattr(
        reference.geometry,
        "union_all",
    ):
        support_geometry = (
            reference.geometry
            .union_all()
        )
    else:
        support_geometry = (
            reference.geometry
            .unary_union
        )

    support_gdf = (
        gpd.GeoDataFrame(
            geometry=[
                support_geometry
            ],
            crs=crs,
        )
    )

    forest_clipped = gpd.clip(
        forest,
        support_gdf,
    )

    annotation_clipped = gpd.clip(
        annotation,
        support_gdf,
    )

    forest_clipped = clean_gdf(
        forest_clipped
    )

    annotation_clipped = clean_gdf(
        annotation_clipped
    )

    forest_raster = binary_raster(
        forest_clipped,
        transform,
    )

    annotation_raster = binary_raster(
        annotation_clipped,
        transform,
    )

    truth = (
        annotation_raster
        == 1
    ).astype(
        np.uint8
    )

    return {
        "crs": crs,
        "transform": transform,
        "forest_raster": (
            forest_raster
        ),
        "truth": truth,
    }


EVAL_CONTEXT = {
    year: prepare_evaluation_context(
        year
    )
    for year
    in YEAR_CONFIG
}


def evaluate_polygons(
    polygons: list,
    source_crs,
    year: int,
) -> dict:
    context = (
        EVAL_CONTEXT[
            year
        ]
    )

    if polygons:
        prediction_gdf = (
            gpd.GeoDataFrame(
                {
                    "class": (
                        [1]
                        * len(polygons)
                    )
                },
                geometry=polygons,
                crs=source_crs,
            )
        )

        prediction_gdf = (
            prediction_gdf
            .to_crs(
                context[
                    "crs"
                ]
            )
        )

    else:
        prediction_gdf = (
            gpd.GeoDataFrame(
                {
                    "class": (
                        pd.Series(
                            dtype=np.int16
                        )
                    )
                },
                geometry=gpd.GeoSeries(
                    [],
                    crs=context[
                        "crs"
                    ],
                ),
                crs=context[
                    "crs"
                ],
            )
        )

    prediction_raster = binary_raster(
        prediction_gdf,
        context[
            "transform"
        ],
    )

    prediction = (
        (
            prediction_raster
            == 1
        )
        & (
            context[
                "forest_raster"
            ]
            == 1
        )
    ).astype(
        np.uint8
    )

    truth = context[
        "truth"
    ]

    y_pred = prediction.ravel()
    y_true = truth.ravel()

    tp = int(
        np.sum(
            (
                y_true == 1
            )
            & (
                y_pred == 1
            )
        )
    )

    fp = int(
        np.sum(
            (
                y_true == 0
            )
            & (
                y_pred == 1
            )
        )
    )

    fn = int(
        np.sum(
            (
                y_true == 1
            )
            & (
                y_pred == 0
            )
        )
    )

    tn = int(
        np.sum(
            (
                y_true == 0
            )
            & (
                y_pred == 0
            )
        )
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "iou": float(
            jaccard_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
    }


# =============================================================================
# COMMON CHANGE DETECTION / METRIC COMPUTATION
# =============================================================================

def anomaly_masks_to_metrics(
    pair_data: dict,
    previous_masks: dict,
    target_masks: dict,
    year: int,
) -> tuple[
    dict,
    int,
]:
    polygons = []
    total_change_pixels = 0

    for tile in pair_data[
        "common"
    ]:
        previous_anomaly = (
            filter_small_components(
                previous_masks[
                    tile
                ]
            )
        )

        target_anomaly = (
            filter_small_components(
                target_masks[
                    tile
                ]
            )
        )

        change = (
            (
                previous_anomaly
                == 0
            )
            & (
                target_anomaly
                == 1
            )
        )

        change = (
            filter_small_components(
                change
            )
        )

        total_change_pixels += int(
            change.sum()
        )

        polygons.extend(
            polygons_from_mask(
                change,
                pair_data[
                    "cached"
                ][
                    tile
                ][
                    "transform"
                ],
            )
        )

    metrics = evaluate_polygons(
        polygons,
        pair_data[
            "crs"
        ],
        year,
    )

    return (
        metrics,
        total_change_pixels,
    )


# =============================================================================
# GMM POSTERIOR SWEEP USING GLOBAL 1519-TILE GMM
# =============================================================================

def run_gmm_threshold_sweep(
    model_name: str,
    year: int,
    pair_data: dict,
) -> list[dict]:
    calibration = (
        GLOBAL_CALIBRATION[
            model_name
        ]
    )

    gmm = calibration[
        "gmm"
    ]

    anomaly_component = (
        calibration[
            "anomaly_component"
        ]
    )

    means = calibration[
        "means"
    ]

    # Compute yearly pair probabilities once, but with the GLOBAL GMM.
    previous_probability = {}
    target_probability = {}

    for tile in pair_data[
        "common"
    ]:
        previous_error = (
            pair_data[
                "cached"
            ][
                tile
            ][
                "previous"
            ]
        )

        target_error = (
            pair_data[
                "cached"
            ][
                tile
            ][
                "target"
            ]
        )

        previous_probability[
            tile
        ] = (
            gmm.predict_proba(
                previous_error.reshape(
                    -1,
                    1,
                )
            )[
                :,
                anomaly_component,
            ]
            .reshape(
                previous_error.shape
            )
        )

        target_probability[
            tile
        ] = (
            gmm.predict_proba(
                target_error.reshape(
                    -1,
                    1,
                )
            )[
                :,
                anomaly_component,
            ]
            .reshape(
                target_error.shape
            )
        )

    results = []

    for threshold in THRESHOLDS:
        previous_masks = {
            tile: (
                previous_probability[
                    tile
                ]
                >= threshold
            )
            for tile
            in pair_data[
                "common"
            ]
        }

        target_masks = {
            tile: (
                target_probability[
                    tile
                ]
                >= threshold
            )
            for tile
            in pair_data[
                "common"
            ]
        }

        (
            metrics,
            change_pixels,
        ) = anomaly_masks_to_metrics(
            pair_data,
            previous_masks,
            target_masks,
            year,
        )

        row = {
            "method": (
                "GMM posterior - global 1519 tiles"
            ),
            "model": (
                model_name
            ),
            "year": (
                year
            ),
            "threshold": float(
                threshold
            ),
            "threshold_percent": float(
                threshold * 100.0
            ),
            "calibration_tiles": (
                EXPECTED_TEST_TILES
            ),
            "calibration_pixels": (
                calibration[
                    "metadata"
                ][
                    "calibration_pixel_count"
                ]
            ),
            "gmm_mean_0": float(
                means[
                    0
                ]
            ),
            "gmm_mean_1": float(
                means[
                    1
                ]
            ),
            "gmm_anomaly_component": (
                anomaly_component
            ),
            "change_pixels": (
                change_pixels
            ),
            **metrics,
        }

        results.append(
            row
        )

        print(
            f"GMM GLOBAL "
            f"{model_name:15s} "
            f"{year} "
            f"T={threshold * 100:.0f}% | "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"IoU={metrics['iou']:.4f} "
            f"Acc={metrics['accuracy']:.4f}"
        )

    return results


# =============================================================================
# DIRECT THRESHOLD SWEEP USING GLOBAL 1519-TILE ROBUST NORMALIZATION
# =============================================================================

def run_direct_threshold_sweep(
    model_name: str,
    year: int,
    pair_data: dict,
) -> list[dict]:
    calibration = (
        GLOBAL_CALIBRATION[
            model_name
        ]
    )

    robust_low = float(
        calibration[
            "robust_low"
        ]
    )

    robust_high = float(
        calibration[
            "robust_high"
        ]
    )

    robust_range = (
        robust_high
        - robust_low
    )

    if robust_range <= 0:
        raise RuntimeError(
            f"{model_name}: invalid global robust range."
        )

    previous_score = {}
    target_score = {}

    for tile in pair_data[
        "common"
    ]:
        previous_score[
            tile
        ] = np.clip(
            (
                pair_data[
                    "cached"
                ][
                    tile
                ][
                    "previous"
                ]
                - robust_low
            )
            / robust_range,
            0.0,
            1.0,
        )

        target_score[
            tile
        ] = np.clip(
            (
                pair_data[
                    "cached"
                ][
                    tile
                ][
                    "target"
                ]
                - robust_low
            )
            / robust_range,
            0.0,
            1.0,
        )

    results = []

    for threshold in THRESHOLDS:
        raw_error_threshold = (
            robust_low
            + threshold
            * robust_range
        )

        previous_masks = {
            tile: (
                previous_score[
                    tile
                ]
                >= threshold
            )
            for tile
            in pair_data[
                "common"
            ]
        }

        target_masks = {
            tile: (
                target_score[
                    tile
                ]
                >= threshold
            )
            for tile
            in pair_data[
                "common"
            ]
        }

        (
            metrics,
            change_pixels,
        ) = anomaly_masks_to_metrics(
            pair_data,
            previous_masks,
            target_masks,
            year,
        )

        row = {
            "method": (
                "Direct robust error - global 1519 tiles"
            ),
            "model": (
                model_name
            ),
            "year": (
                year
            ),
            "threshold": float(
                threshold
            ),
            "threshold_percent": float(
                threshold * 100.0
            ),
            "calibration_tiles": (
                EXPECTED_TEST_TILES
            ),
            "calibration_pixels": (
                calibration[
                    "metadata"
                ][
                    "calibration_pixel_count"
                ]
            ),
            "global_robust_low": (
                robust_low
            ),
            "global_robust_high": (
                robust_high
            ),
            "raw_error_threshold": float(
                raw_error_threshold
            ),
            "change_pixels": (
                change_pixels
            ),
            **metrics,
        }

        results.append(
            row
        )

        print(
            f"DIRECT GLOBAL "
            f"{model_name:15s} "
            f"{year} "
            f"T={threshold * 100:.0f}% "
            f"(raw={raw_error_threshold:.6f}) | "
            f"P={metrics['precision']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"F1={metrics['f1']:.4f} "
            f"IoU={metrics['iou']:.4f} "
            f"Acc={metrics['accuracy']:.4f}"
        )

    return results


# =============================================================================
# RUN SENSITIVITY ANALYSIS
# =============================================================================

all_gmm_results = []
all_direct_results = []

for model_name in MODELS:
    print(
        "\n"
        + "#" * 120
    )

    print(
        f"MODEL: {model_name.upper()}"
    )

    print(
        "#"
        * 120
    )

    for year in (
        2021,
        2022,
    ):
        print(
            "\n"
            + "=" * 120
        )

        print(
            f"{model_name.upper()} - {year}"
        )

        print(
            "=" * 120
        )

        pair_data = load_year_pair_data(
            model_name,
            year,
        )

        all_gmm_results.extend(
            run_gmm_threshold_sweep(
                model_name,
                year,
                pair_data,
            )
        )

        all_direct_results.extend(
            run_direct_threshold_sweep(
                model_name,
                year,
                pair_data,
            )
        )

        del pair_data

        gc.collect()


# =============================================================================
# SAVE RESULTS
# =============================================================================

gmm_df = pd.DataFrame(
    all_gmm_results
)

direct_df = pd.DataFrame(
    all_direct_results
)

combined_df = pd.concat(
    [
        gmm_df,
        direct_df,
    ],
    ignore_index=True,
    sort=False,
)

# Raw result files retain all calibration-specific columns.
gmm_csv = (
    OUTPUT_ROOT
    / "gmm_global1519_threshold_0p5pct_to_90pct.csv"
)

direct_csv = (
    OUTPUT_ROOT
    / "direct_global1519_threshold_0p5pct_to_90pct.csv"
)

combined_csv = (
    OUTPUT_ROOT
    / "all_global1519_threshold_0p5pct_to_90pct.csv"
)

gmm_df.to_csv(
    gmm_csv,
    index=False,
)

direct_df.to_csv(
    direct_csv,
    index=False,
)

combined_df.to_csv(
    combined_csv,
    index=False,
)

(
    OUTPUT_ROOT
    / "gmm_global1519_threshold_0p5pct_to_90pct.json"
).write_text(
    json.dumps(
        all_gmm_results,
        indent=2,
    ),
    encoding="utf-8",
)

(
    OUTPUT_ROOT
    / "direct_global1519_threshold_0p5pct_to_90pct.json"
).write_text(
    json.dumps(
        all_direct_results,
        indent=2,
    ),
    encoding="utf-8",
)


# =============================================================================
# CLEAN COMPLETE METRIC TABLES
# =============================================================================

def make_percent_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a clean publication-friendly table.

    One row = one technique/model/year/threshold combination.

    Precision, Recall, F1, IoU and Accuracy are converted from [0, 1]
    to percentages. TP, FP, FN and TN remain pixel counts.
    """
    columns = [
        "method",
        "model",
        "year",
        "threshold",
        "threshold_percent",
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
        "tp",
        "fp",
        "fn",
        "tn",
        "change_pixels",
    ]

    result = (
        dataframe[
            columns
        ]
        .copy()
    )

    # More readable technique names in the clean table.
    result["technique"] = result["method"].map(
        {
            "GMM posterior - global 1519 tiles": "GMM posterior",
            "Direct robust error - global 1519 tiles": "Direct threshold",
        }
    ).fillna(
        result["method"]
    )

    result["model_label"] = result["model"].map(
        MODEL_LABELS
    ).fillna(
        result["model"]
    )

    for metric in (
        "precision",
        "recall",
        "f1",
        "iou",
        "accuracy",
    ):
        result[metric] = (
            result[metric]
            * 100.0
        )

    result = result[
        [
            "technique",
            "model_label",
            "model",
            "year",
            "threshold",
            "threshold_percent",
            "precision",
            "recall",
            "f1",
            "iou",
            "accuracy",
            "tp",
            "fp",
            "fn",
            "tn",
            "change_pixels",
        ]
    ]

    result = result.sort_values(
        [
            "year",
            "technique",
            "model",
            "threshold",
        ]
    ).reset_index(
        drop=True
    )

    return result.round(
        {
            "threshold": 2,
            "precision": 2,
            "recall": 2,
            "f1": 2,
            "iou": 2,
            "accuracy": 2,
        }
    )


complete_results_table = make_percent_table(
    combined_df
)

gmm_results_table = make_percent_table(
    gmm_df
)

direct_results_table = make_percent_table(
    direct_df
)


# Save complete clean tables.
complete_table_csv = (
    OUTPUT_ROOT
    / "complete_metrics_all_models_years_techniques_0p5pct_to_90pct.csv"
)

gmm_table_csv = (
    OUTPUT_ROOT
    / "complete_metrics_gmm_0p5pct_to_90pct.csv"
)

direct_table_csv = (
    OUTPUT_ROOT
    / "complete_metrics_direct_0p5pct_to_90pct.csv"
)

complete_results_table.to_csv(
    complete_table_csv,
    index=False,
)

gmm_results_table.to_csv(
    gmm_table_csv,
    index=False,
)

direct_results_table.to_csv(
    direct_table_csv,
    index=False,
)

# Also save a separate clean table for each year.
for year in YEAR_CONFIG:
    year_table = complete_results_table[
        complete_results_table[
            "year"
        ]
        == year
    ].copy()

    year_table.to_csv(
        OUTPUT_ROOT
        / f"complete_metrics_{year}_0p5pct_to_90pct.csv",
        index=False,
    )


# Display all metrics in the terminal.
print(
    "\nCOMPLETE RESULTS - ALL MODELS / YEARS / TECHNIQUES:"
)

print(
    complete_results_table.to_string(
        index=False
    )
)


# =============================================================================
# IoU-ONLY PLOTTING
# =============================================================================

def format_threshold_label(value: float) -> str:
    """Format threshold percentages cleanly for the x-axis."""
    if np.isclose(value, 0.5):
        return "0.5"
    if np.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


# Keep the same visual identity for every model in ALL plots.
# Matplotlib's default color cycle is used; we only fix marker/linestyle so
# the model representation is consistent in both panels.
MODEL_STYLES = {
    "segformer": {"marker": "o", "linestyle": "-"},
    "unet": {"marker": "s", "linestyle": "-"},
    "pspnet": {"marker": "^", "linestyle": "-"},
    "deeplabv3plus": {"marker": "D", "linestyle": "-"},
}


def _plot_iou_on_axis(
    ax,
    dataframe: pd.DataFrame,
    year: int,
    panel_title: str,
) -> None:
    """Draw one yearly IoU panel on an existing axis."""
    subset = dataframe[
        dataframe["year"] == year
    ].copy()

    threshold_values = [
        float(v) * 100.0
        for v in THRESHOLDS
    ]
    x_positions = np.arange(
        len(threshold_values),
        dtype=float,
    )

    for model_name in MODELS:
        model_df = subset[
            subset["model"] == model_name
        ].copy()

        if model_df.empty:
            continue

        model_df = (
            model_df
            .assign(
                threshold_key=np.round(
                    model_df["threshold_percent"].astype(float),
                    6,
                )
            )
            .set_index("threshold_key")
        )

        valid_x = []
        y_values = []

        for xpos, threshold_pct in zip(
            x_positions,
            threshold_values,
        ):
            key = round(float(threshold_pct), 6)
            if key not in model_df.index:
                continue

            row = model_df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            valid_x.append(xpos)
            y_values.append(float(row["iou"]) * 100.0)

        style = MODEL_STYLES[model_name]
        ax.plot(
            valid_x,
            y_values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7.5,
            label=MODEL_LABELS[model_name],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [format_threshold_label(v) for v in threshold_values],
        rotation=45,
        ha="right",
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Threshold (%)", fontsize=16)
    ax.set_ylabel("IoU (%)", fontsize=16)
    ax.set_title(panel_title, fontsize=17, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.4, len(threshold_values) - 0.6)


def plot_yearly_two_panel_iou(
    gmm_dataframe: pd.DataFrame,
    direct_dataframe: pd.DataFrame,
    year: int,
) -> None:
    """
    Create one figure with two panels for a given year:
      (a) GMM posterior
      (b) Direct thresholding (DT)

    A single shared legend is used because model labels are identical.
    """
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20, 7.5),
        sharey=True,
    )

    _plot_iou_on_axis(
        axes[0],
        gmm_dataframe,
        year,
        "(a) GMM posterior",
    )

    _plot_iou_on_axis(
        axes[1],
        direct_dataframe,
        year,
        "(b) Direct thresholding (DT)",
    )

    # Only one y-label for the whole figure.
    axes[1].set_ylabel("")

    # One shared legend with the same model labeling for both panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(MODELS),
        fontsize=14,
        title_fontsize=15,
        frameon=True,
    )

    fig.suptitle(
        f"IoU sensitivity to anomaly threshold — {year}",
        fontsize=19,
        fontweight="bold",
        y=1.10,
    )

    fig.tight_layout(
        rect=[0.0, 0.0, 1.0, 0.93]
    )

    output = (
        OUTPUT_ROOT
        / f"gmm_vs_dt_iou_vs_threshold_{year}_two_panels.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print("Saved yearly two-panel IoU plot:", output)


# =============================================================================
# CREATE YEARLY TWO-PANEL IoU FIGURES
# =============================================================================

for year in (2021, 2022):
    plot_yearly_two_panel_iou(
        gmm_df,
        direct_df,
        year,
    )


# =============================================================================
# TWO-YEAR AVERAGE IoU PLOTS
# =============================================================================

def build_two_year_average_iou(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Average IoU across 2021 and 2022 for every model and threshold.
    Only model/threshold combinations containing both years are retained.
    """
    yearly = dataframe[
        [
            "model",
            "year",
            "threshold",
            "threshold_percent",
            "iou",
        ]
    ].copy()

    year_counts = (
        yearly
        .groupby(
            [
                "model",
                "threshold",
                "threshold_percent",
            ],
            as_index=False,
        )["year"]
        .nunique()
        .rename(columns={"year": "n_years"})
    )

    average_df = (
        yearly
        .groupby(
            [
                "model",
                "threshold",
                "threshold_percent",
            ],
            as_index=False,
        )["iou"]
        .mean()
        .rename(columns={"iou": "mean_iou_2021_2022"})
    )

    average_df = average_df.merge(
        year_counts,
        on=[
            "model",
            "threshold",
            "threshold_percent",
        ],
        how="left",
    )

    return average_df[
        average_df["n_years"] == 2
    ].copy()


def _plot_average_iou_on_axis(
    ax,
    average_df: pd.DataFrame,
    panel_title: str,
) -> None:
    """Draw one two-year-mean IoU panel on an existing axis."""
    threshold_values = [
        float(v) * 100.0
        for v in THRESHOLDS
    ]
    x_positions = np.arange(
        len(threshold_values),
        dtype=float,
    )

    for model_name in MODELS:
        model_df = average_df[
            average_df["model"] == model_name
        ].copy()

        if model_df.empty:
            continue

        model_df = (
            model_df
            .assign(
                threshold_key=np.round(
                    model_df["threshold_percent"].astype(float),
                    6,
                )
            )
            .set_index("threshold_key")
        )

        valid_x = []
        y_values = []

        for xpos, threshold_pct in zip(
            x_positions,
            threshold_values,
        ):
            key = round(float(threshold_pct), 6)
            if key not in model_df.index:
                continue

            row = model_df.loc[key]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            valid_x.append(xpos)
            y_values.append(
                float(row["mean_iou_2021_2022"]) * 100.0
            )

        style = MODEL_STYLES[model_name]
        ax.plot(
            valid_x,
            y_values,
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.8,
            markersize=7.5,
            label=MODEL_LABELS[model_name],
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [format_threshold_label(v) for v in threshold_values],
        rotation=45,
        ha="right",
        fontsize=13,
    )
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlabel("Threshold (%)", fontsize=16)
    ax.set_ylabel("Mean IoU (%)", fontsize=16)
    ax.set_title(panel_title, fontsize=17, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.set_xlim(-0.4, len(threshold_values) - 0.6)


def plot_average_two_panel_iou(
    gmm_dataframe: pd.DataFrame,
    direct_dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create ONE publication figure with two panels:
      (a) GMM posterior mean IoU over 2021 and 2022
      (b) DT mean IoU over 2021 and 2022

    Both panels use the same model labels and shared legend.
    """
    gmm_average_df = build_two_year_average_iou(
        gmm_dataframe
    )
    direct_average_df = build_two_year_average_iou(
        direct_dataframe
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(20, 7.5),
        sharey=True,
    )

    _plot_average_iou_on_axis(
        axes[0],
        gmm_average_df,
        "(a) GMM posterior",
    )

    _plot_average_iou_on_axis(
        axes[1],
        direct_average_df,
        "(b) Direct thresholding (DT)",
    )

    axes[1].set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Model",
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(MODELS),
        fontsize=14,
        title_fontsize=15,
        frameon=True,
    )

    fig.suptitle(
        "Mean IoU sensitivity to anomaly threshold (2021–2022)",
        fontsize=19,
        fontweight="bold",
        y=1.10,
    )

    fig.tight_layout(
        rect=[0.0, 0.0, 1.0, 0.93]
    )

    output = (
        OUTPUT_ROOT
        / "gmm_vs_dt_mean_iou_2021_2022_two_panels.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print("Saved two-panel 2-year mean IoU plot:", output)

    return gmm_average_df, direct_average_df


# Create one combined 2-year-average figure with two panels.
gmm_average_iou_df, direct_average_iou_df = plot_average_two_panel_iou(
    gmm_df,
    direct_df,
)

# Save the complete average-IoU curves as tables too.
gmm_average_iou_csv = (
    OUTPUT_ROOT
    / "gmm_mean_iou_2021_2022_all_thresholds.csv"
)

direct_average_iou_csv = (
    OUTPUT_ROOT
    / "direct_mean_iou_2021_2022_all_thresholds.csv"
)

gmm_average_iou_df.to_csv(
    gmm_average_iou_csv,
    index=False,
)

direct_average_iou_df.to_csv(
    direct_average_iou_csv,
    index=False,
)


# =============================================================================
# BEST THRESHOLD PER MODEL BASED ON MEAN IoU ACROSS BOTH YEARS
# =============================================================================

def best_threshold_summary(
    dataframe: pd.DataFrame,
    technique_label: str,
) -> pd.DataFrame:
    """
    For each model:
      1. compute mean IoU across 2021 and 2022 at every threshold;
      2. select the threshold with the highest 2-year mean IoU;
      3. report that mean together with the individual 2021 and 2022 IoUs.
    """
    average_df = build_two_year_average_iou(
        dataframe
    )

    rows = []

    for model_name in MODELS:
        model_average = average_df[
            average_df["model"] == model_name
        ].copy()

        if model_average.empty:
            continue

        # Best threshold according to the average IoU over both years.
        # idxmax returns the first occurrence in case of an exact tie.
        best_idx = model_average[
            "mean_iou_2021_2022"
        ].idxmax()

        best_row = model_average.loc[
            best_idx
        ]

        best_threshold = float(
            best_row["threshold"]
        )

        best_threshold_percent = float(
            best_row["threshold_percent"]
        )

        yearly_rows = dataframe[
            (dataframe["model"] == model_name)
            & np.isclose(
                dataframe["threshold"].astype(float),
                best_threshold,
            )
        ].copy()

        iou_2021_rows = yearly_rows[
            yearly_rows["year"] == 2021
        ]

        iou_2022_rows = yearly_rows[
            yearly_rows["year"] == 2022
        ]

        iou_2021 = (
            float(iou_2021_rows.iloc[0]["iou"])
            if not iou_2021_rows.empty
            else np.nan
        )

        iou_2022 = (
            float(iou_2022_rows.iloc[0]["iou"])
            if not iou_2022_rows.empty
            else np.nan
        )

        rows.append(
            {
                "technique": technique_label,
                "model": MODEL_LABELS[model_name],
                "model_key": model_name,
                "best_threshold": best_threshold,
                "best_threshold_percent": best_threshold_percent,
                "mean_iou_2021_2022": float(
                    best_row["mean_iou_2021_2022"]
                ),
                "iou_2021": iou_2021,
                "iou_2022": iou_2022,
            }
        )

    return pd.DataFrame(
        rows
    )


gmm_best_summary = best_threshold_summary(
    gmm_df,
    "GMM posterior",
)

direct_best_summary = best_threshold_summary(
    direct_df,
    "Direct normalized reconstruction error",
)

best_thresholds_df = pd.concat(
    [
        gmm_best_summary,
        direct_best_summary,
    ],
    ignore_index=True,
)

# Percentage version for easier reading in the final table.
best_thresholds_display = best_thresholds_df.copy()

for metric_column in (
    "mean_iou_2021_2022",
    "iou_2021",
    "iou_2022",
):
    best_thresholds_display[
        metric_column
    ] = (
        best_thresholds_display[
            metric_column
        ]
        * 100.0
    )

best_thresholds_display = (
    best_thresholds_display[
        [
            "technique",
            "model",
            "best_threshold",
            "best_threshold_percent",
            "mean_iou_2021_2022",
            "iou_2021",
            "iou_2022",
        ]
    ]
    .round(
        {
            "best_threshold": 4,
            "best_threshold_percent": 1,
            "mean_iou_2021_2022": 2,
            "iou_2021": 2,
            "iou_2022": 2,
        }
    )
)

best_thresholds_csv = (
    OUTPUT_ROOT
    / "best_threshold_per_model_based_on_2year_mean_iou.csv"
)

best_thresholds_display.to_csv(
    best_thresholds_csv,
    index=False,
)

print(
    "\n"
    + "=" * 120
)

print(
    "BEST THRESHOLD PER MODEL — SELECTED USING MEAN IoU OF 2021 AND 2022"
)

print(
    "=" * 120
)

print(
    best_thresholds_display.to_string(
        index=False
    )
)

print(
    "=" * 120
)

print(
    "Saved best-threshold table:",
    best_thresholds_csv,
)


# =============================================================================
# FINAL
# =============================================================================

print(
    "\n"
    + "=" * 120
)

print(
    "GLOBAL 1519-TILE THRESHOLD ANALYSIS COMPLETE"
)

print(
    "=" * 120
)

print(
    "GMM calibration: ALL 1519 test tiles contribute."
)

print(
    "Direct normalization: ALL 1519 test tiles contribute."
)

print(
    "49 common spatial pairs are used ONLY for "
    "the actual April -> September yearly change maps."
)

print(
    "Thresholds: 0.5%, 1%-10%, then 20%-90% in 10% steps"
)

print(
    "Thresholds:",
    [
        (
            f"{float(value) * 100:.1f}%"
            if np.isclose(float(value), 0.005)
            else f"{float(value) * 100:.0f}%"
        )
        for value
        in THRESHOLDS
    ],
)

print(
    "Metrics saved in tables: "
    "Precision, Recall, F1, IoU, Accuracy, "
    "TP, FP, FN, TN, change pixels."
)

print(
    "Plots generated: two-panel IoU figures for each year "
    "plus one two-panel 2-year-average IoU figure."
)

print(
    "Output folder:",
    OUTPUT_ROOT,
)

print(
    "GMM raw CSV:",
    gmm_csv,
)

print(
    "Direct raw CSV:",
    direct_csv,
)

print(
    "Combined raw CSV:",
    combined_csv,
)

print(
    "Complete clean metric table:",
    complete_table_csv,
)

print(
    "GMM 2-year mean IoU table:",
    gmm_average_iou_csv,
)

print(
    "Direct 2-year mean IoU table:",
    direct_average_iou_csv,
)

print(
    "Best threshold summary:",
    best_thresholds_csv,
)

print(
    "=" * 120
)