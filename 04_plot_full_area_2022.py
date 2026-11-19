#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FADAS (U-Net) — GLOBAL GMM posterior threshold = 2.0% — 2022

Creates TWO SEPARATE 6-panel figures:

1) COMPLETE STUDY AREA (all common 49 tiles)
2) SINGLE TILE 1024_1280

Panels:
    (a) Sentinel-1 VV/VH — 04/04/2022
    (b) Sentinel-1 VV/VH — 19/09/2022
    (c) GMM anomaly probability — 04/04/2022
    (d) GMM anomaly probability — 19/09/2022
    (e) Binary reference deforestation map — 2022
    (f) FADAS (U-Net) prediction — TP / FP / FN

IMPORTANT FIXES:
    - Pixel values are read with tifffile, exactly as in the earlier working
      visualization/training pipeline. Rasterio is used ONLY for georeferencing.
    - North arrow and scale bar are drawn in axes coordinates and therefore
      CANNOT change the map extent or shrink the Sentinel-1 image.

Detection protocol:
    - reconstruction model: U-Net
    - global 2-component GMM calibrated from ALL 1519 test error maps
    - anomaly component = GMM component with the larger mean error
    - posterior anomaly threshold = 0.02 (2%)
    - April anomaly mask and September anomaly mask are each filtered with
      minimum connected-component size = 100 pixels
    - deforestation = April NON-anomaly AND September anomaly
    - final change map is filtered again with minimum size = 100 pixels
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy import ndimage
from sklearn.mixture import GaussianMixture
from pyproj import CRS, Geod, Transformer
import tifffile


# =============================================================================
# 1. PATHS
# =============================================================================

# DATA_ROOT must point to the external dataset "forest" directory.
# This figure uses the Sentinel-1 TIFFs from DATA_ROOT/test/processed.
DATA_ROOT = Path("PATH/TO/DATASET/forest")

TEST_DIR = (
    DATA_ROOT
    / "test"
    / "processed"
)

# RESULTS_ROOT must point to the output directory created by the reconstruction
# training/evaluation script. This figure expects U-Net test error maps under:
#   RESULTS_ROOT/unet/test_maps/
RESULTS_ROOT = Path("PATH/TO/RESULTS/four_models_reconstruction_50epochs")

MODEL_NAME = "unet"
MODEL_LABEL = "FADAS (U-Net)"

ERROR_DIR = (
    RESULTS_ROOT
    / MODEL_NAME
    / "test_maps"
)

OUTPUT_DIR = (
    RESULTS_ROOT
    / "paper_figures_unet_gmm_2pct_2022"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FULL_FIGURE_PATH = (
    OUTPUT_DIR
    / "fadas_unet_gmm_2pct_complete_study_area_2022.png"
)

TILE_FIGURE_PATH = (
    OUTPUT_DIR
    / "fadas_unet_gmm_2pct_tile_1024_1280_2022.png"
)


# =============================================================================
# 2. EXPERIMENT CONFIGURATION
# =============================================================================

PREVIOUS_DATE = "20220404"
TARGET_DATE = "20220919"

GMM_THRESHOLD = 0.02

EXPECTED_TEST_TILES = 1519
PIXELS_PER_TILE_FOR_GLOBAL_CALIBRATION = 1024
GMM_SEED = 42

MIN_COMPONENT_SIZE = 100

TILE_TO_PLOT = (
    1024,
    1280,
)


# =============================================================================
# 3. COLORS
# =============================================================================

COLOR_OUTSIDE = np.array(
    [0.91, 0.91, 0.91],
    dtype=np.float32,
)

COLOR_BACKGROUND = np.array(
    [1.00, 1.00, 1.00],
    dtype=np.float32,
)

COLOR_REFERENCE = np.array(
    [0.843, 0.188, 0.153],   # #d73027
    dtype=np.float32,
)

COLOR_FP = np.array(
    [1.000, 0.843, 0.000],   # #ffd700
    dtype=np.float32,
)

COLOR_TP = np.array(
    [0.102, 0.596, 0.314],   # #1a9850
    dtype=np.float32,
)

COLOR_FN = np.array(
    [0.957, 0.427, 0.263],   # #f46d43
    dtype=np.float32,
)


# =============================================================================
# 4. FILE PARSING
# =============================================================================

DATE_RE = re.compile(
    r"_(20\d{6})T"
)

TILE_RE = re.compile(
    r"_db_(\d+)_(\d+)"
)


def get_date_and_tile(
    path: Path,
):
    """
    Extract:
        date = YYYYMMDD
        tile = (row_id, col_id)
    """
    name = Path(path).name

    date_match = DATE_RE.search(
        name
    )

    tile_match = TILE_RE.search(
        name
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


def index_files(
    folder: Path,
    error: bool = False,
) -> dict:
    """
    Return:
        {(date, (row, col)): path}
    """
    pattern = (
        "*_error.tif"
        if error
        else "*.tif"
    )

    result = {}

    for path in sorted(
        folder.glob(
            pattern
        )
    ):
        info = get_date_and_tile(
            path
        )

        if info is None:
            continue

        date, tile = info

        result[
            (
                date,
                tile,
            )
        ] = path

    return result


# =============================================================================
# 5. READ SENTINEL-1
# =============================================================================

def read_s1_pixels(
    path: Path,
) -> np.ndarray:
    """
    IMPORTANT:
    Read the actual two-channel Sentinel-1 pixel values with tifffile.

    Returns:
        image shape = (2, H, W)
    """
    image = np.asarray(
        tifffile.imread(
            path
        ),
        dtype=np.float32,
    )

    image = np.squeeze(
        image
    )

    if (
        image.ndim == 3
        and image.shape[0] == 2
    ):
        pass

    elif (
        image.ndim == 3
        and image.shape[-1] == 2
    ):
        image = np.moveaxis(
            image,
            -1,
            0,
        )

    else:
        raise ValueError(
            f"Unexpected Sentinel-1 shape {image.shape}: {path}"
        )

    return image


def read_georef(
    path: Path,
):
    """
    Rasterio is used only for georeferencing.
    """
    with rasterio.open(
        path
    ) as src:
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        height = src.height
        width = src.width

    return (
        transform,
        crs,
        bounds,
        height,
        width,
    )


# =============================================================================
# 6. READ RECONSTRUCTION ERROR
# =============================================================================

def read_error_sum(
    path: Path,
) -> np.ndarray:
    """
    Stored error TIFFs contain mean squared error over VV/VH.
    Multiply by 2 to preserve the summed VV+VH convention used by
    the threshold-sweep analysis.
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


# =============================================================================
# 7. METRICS DIRECTORY
# =============================================================================

def directory_has_required_metrics(
    directory: Path,
) -> bool:
    required = (
        "Forest_Mask_2021.shp",
        "622_975_2022.shp",
    )

    return all(
        (
            directory
            / name
        ).is_file()
        for name
        in required
    )


# Set this to the directory containing the reference vector data used by the
# paper. For this 2022 figure it must contain at least:
#   Forest_Mask_2021.shp
#   622_975_2022.shp
METRICS_ROOT = Path("PATH/TO/DATASET/metrics")

if not directory_has_required_metrics(METRICS_ROOT):
    raise FileNotFoundError(
        "METRICS_ROOT must contain Forest_Mask_2021.shp and 622_975_2022.shp. "
        f"Current METRICS_ROOT={METRICS_ROOT}"
    )


# =============================================================================
# 8. VECTOR HELPERS
# =============================================================================

def clean_gdf(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    gdf = gdf[
        gdf.geometry.notna()
        & ~gdf.geometry.is_empty
    ].copy()

    if len(gdf) == 0:
        return gdf

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


def read_vector(
    path: Path,
) -> gpd.GeoDataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            path
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


def rasterize_vector(
    gdf: gpd.GeoDataFrame,
    transform,
    shape_hw: tuple[int, int],
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
            shape_hw,
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
        out_shape=shape_hw,
        transform=transform,
        fill=0,
        default_value=1,
        all_touched=False,
        dtype=np.uint8,
    )


# =============================================================================
# 9. CONNECTED COMPONENT FILTER
# =============================================================================

def filter_small_components(
    mask: np.ndarray,
    minimum: int = MIN_COMPONENT_SIZE,
) -> np.ndarray:
    labels, count = ndimage.label(
        mask.astype(
            bool
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

    return (
        keep[
            labels
        ]
        .astype(
            np.uint8
        )
    )


# =============================================================================
# 10. GLOBAL GMM CALIBRATION — ALL 1519 TEST TILES
# =============================================================================

def build_global_calibration_sample(
    error_dir: Path,
) -> np.ndarray:
    files = sorted(
        error_dir.glob(
            "*_error.tif"
        )
    )

    if len(files) != EXPECTED_TEST_TILES:
        print(
            f"WARNING: expected {EXPECTED_TEST_TILES} "
            f"U-Net error maps, found {len(files)}."
        )

    if not files:
        raise FileNotFoundError(
            f"No error maps found in {error_dir}"
        )

    rng = np.random.default_rng(
        GMM_SEED
    )

    blocks = []

    print(
        "\nBuilding GLOBAL U-Net GMM calibration..."
    )

    for index, path in enumerate(
        files,
        start=1,
    ):
        error = read_error_sum(
            path
        )

        flat = error.ravel()

        number = min(
            PIXELS_PER_TILE_FOR_GLOBAL_CALIBRATION,
            flat.size,
        )

        indices = rng.choice(
            flat.size,
            size=number,
            replace=False,
        )

        blocks.append(
            flat[
                indices
            ].astype(
                np.float32,
                copy=False,
            )
        )

        if (
            index % 200 == 0
            or index == len(files)
        ):
            print(
                f"  calibration tiles: "
                f"{index}/{len(files)}"
            )

    calibration = np.concatenate(
        blocks
    )

    print(
        f"Calibration pixels: "
        f"{calibration.size:,}"
    )

    return calibration


def fit_global_gmm(
    error_dir: Path,
):
    calibration = (
        build_global_calibration_sample(
            error_dir
        )
    )

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

    print(
        "GMM means:",
        means.tolist(),
    )

    print(
        "Anomaly component:",
        anomaly_component,
    )

    del calibration
    gc.collect()

    return (
        gmm,
        anomaly_component,
    )


# =============================================================================
# 11. GMM PROBABILITY + CHANGE DETECTION
# =============================================================================

def probability_map(
    error: np.ndarray,
    gmm: GaussianMixture,
    anomaly_component: int,
) -> np.ndarray:
    probability = (
        gmm.predict_proba(
            error.reshape(
                -1,
                1,
            )
        )[
            :,
            anomaly_component,
        ]
        .reshape(
            error.shape
        )
        .astype(
            np.float32
        )
    )

    return probability


def change_from_probabilities(
    previous_probability: np.ndarray,
    target_probability: np.ndarray,
) -> np.ndarray:
    previous_anomaly = (
        previous_probability
        >= GMM_THRESHOLD
    ).astype(
        np.uint8
    )

    target_anomaly = (
        target_probability
        >= GMM_THRESHOLD
    ).astype(
        np.uint8
    )

    previous_anomaly = (
        filter_small_components(
            previous_anomaly
        )
    )

    target_anomaly = (
        filter_small_components(
            target_anomaly
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
    ).astype(
        np.uint8
    )

    change = (
        filter_small_components(
            change
        )
    )

    return change


# =============================================================================
# 12. SENTINEL-1 DISPLAY
# =============================================================================

def finite_percentiles(
    array: np.ndarray,
    low: float = 2.0,
    high: float = 98.0,
):
    valid = array[
        np.isfinite(
            array
        )
    ]

    if valid.size == 0:
        raise ValueError(
            "No finite values available for visualization."
        )

    p_low = float(
        np.percentile(
            valid,
            low,
        )
    )

    p_high = float(
        np.percentile(
            valid,
            high,
        )
    )

    if p_high <= p_low:
        p_high = (
            p_low
            + 1e-6
        )

    return (
        p_low,
        p_high,
    )


def normalize_with_limits(
    array: np.ndarray,
    low_value: float,
    high_value: float,
) -> np.ndarray:
    result = (
        array
        - low_value
    ) / (
        high_value
        - low_value
    )

    result = np.clip(
        result,
        0.0,
        1.0,
    )

    result[
        ~np.isfinite(
            result
        )
    ] = 0.0

    return result.astype(
        np.float32
    )


def make_s1_rgb_pair(
    previous: np.ndarray,
    target: np.ndarray,
):
    """
    Use IDENTICAL display limits for April and September.

    This prevents one date from appearing blank or being displayed on
    an incompatible stretch.

    Pseudo-RGB:
        R = VV
        G = (VV + VH) / 2
        B = VH
    """
    combined_vv = np.concatenate(
        [
            previous[0][
                np.isfinite(
                    previous[0]
                )
            ],
            target[0][
                np.isfinite(
                    target[0]
                )
            ],
        ]
    )

    combined_vh = np.concatenate(
        [
            previous[1][
                np.isfinite(
                    previous[1]
                )
            ],
            target[1][
                np.isfinite(
                    target[1]
                )
            ],
        ]
    )

    vv_low, vv_high = (
        finite_percentiles(
            combined_vv,
            2.0,
            98.0,
        )
    )

    vh_low, vh_high = (
        finite_percentiles(
            combined_vh,
            2.0,
            98.0,
        )
    )

    def convert(
        image: np.ndarray,
    ) -> np.ndarray:
        vv = normalize_with_limits(
            image[0],
            vv_low,
            vv_high,
        )

        vh = normalize_with_limits(
            image[1],
            vh_low,
            vh_high,
        )

        rgb = np.stack(
            [
                vv,
                (
                    vv
                    + vh
                )
                / 2.0,
                vh,
            ],
            axis=-1,
        )

        return np.clip(
            rgb,
            0.0,
            1.0,
        )

    return (
        convert(
            previous
        ),
        convert(
            target
        ),
    )


# =============================================================================
# 13. DISPLAY MAPS
# =============================================================================

def make_reference_rgb(
    reference_mask: np.ndarray,
    forest_mask: np.ndarray,
) -> np.ndarray:
    height, width = (
        reference_mask.shape
    )

    output = np.empty(
        (
            height,
            width,
            3,
        ),
        dtype=np.float32,
    )

    output[:] = (
        COLOR_BACKGROUND
    )

    output[
        forest_mask == 0
    ] = COLOR_OUTSIDE

    output[
        (
            forest_mask == 1
        )
        & (
            reference_mask == 1
        )
    ] = COLOR_REFERENCE

    return output


def make_confusion_rgb(
    reference_mask: np.ndarray,
    prediction_mask: np.ndarray,
    forest_mask: np.ndarray,
) -> np.ndarray:
    height, width = (
        reference_mask.shape
    )

    output = np.empty(
        (
            height,
            width,
            3,
        ),
        dtype=np.float32,
    )

    output[:] = (
        COLOR_BACKGROUND
    )

    output[
        forest_mask == 0
    ] = COLOR_OUTSIDE

    tp = (
        (forest_mask == 1)
        & (reference_mask == 1)
        & (prediction_mask == 1)
    )

    fp = (
        (forest_mask == 1)
        & (reference_mask == 0)
        & (prediction_mask == 1)
    )

    fn = (
        (forest_mask == 1)
        & (reference_mask == 1)
        & (prediction_mask == 0)
    )

    output[
        tp
    ] = COLOR_TP

    output[
        fp
    ] = COLOR_FP

    output[
        fn
    ] = COLOR_FN

    return output


# =============================================================================
# 14. FIGURE HELPERS
# =============================================================================

def add_north_arrow(
    ax,
):
    ax.annotate(
        "N",
        xy=(
            0.07,
            0.88,
        ),
        xytext=(
            0.07,
            0.74,
        ),
        xycoords="axes fraction",
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        arrowprops=dict(
            facecolor="black",
            edgecolor="black",
            width=2.5,
            headwidth=9,
            headlength=9,
        ),
    )


def add_scale_bar(
    ax,
    extent,
    crs,
):
    """
    Draw a geographically meaningful scale bar WITHOUT changing the image axis.

    The previous version drew the bar in data coordinates and assumed that the
    CRS units were meters. If the raster CRS was geographic (degrees), a value
    such as 1000 was interpreted as 1000 degrees, forcing Matplotlib to expand
    the x-axis and making the Sentinel-1 image appear blank.

    This corrected version:
        1. estimates the true map width in metres using pyproj/Geod;
        2. chooses a convenient scale-bar length;
        3. draws the bar in AXES-FRACTION coordinates (ax.transAxes), so it
           can never alter xlim/ylim or shrink the image.
    """
    left, right, bottom, top = extent

    crs_obj = CRS.from_user_input(crs)
    geod = Geod(ellps="WGS84")

    # Convert left/right midpoint coordinates to lon/lat for a robust distance
    # calculation regardless of whether the source CRS is projected or geographic.
    mid_y = 0.5 * (bottom + top)

    if crs_obj.is_geographic:
        lon1, lat1 = left, mid_y
        lon2, lat2 = right, mid_y
    else:
        transformer = Transformer.from_crs(
            crs_obj,
            CRS.from_epsg(4326),
            always_xy=True,
        )
        lon1, lat1 = transformer.transform(left, mid_y)
        lon2, lat2 = transformer.transform(right, mid_y)

    _, _, width_m = geod.inv(
        lon1,
        lat1,
        lon2,
        lat2,
    )

    width_km = abs(width_m) / 1000.0

    if not np.isfinite(width_km) or width_km <= 0:
        print("WARNING: could not compute scale-bar distance; skipping scale bar.")
        return

    # Choose a clean scale length close to 15--25% of the panel width.
    candidates_km = np.array(
        [
            0.1,
            0.2,
            0.5,
            1.0,
            2.0,
            5.0,
            10.0,
            20.0,
            50.0,
            100.0,
            200.0,
        ],
        dtype=float,
    )

    target_km = width_km * 0.22
    valid = candidates_km[candidates_km <= target_km]

    if valid.size:
        bar_km = float(valid[-1])
    else:
        bar_km = float(candidates_km[0])

    bar_fraction = bar_km / width_km
    bar_fraction = float(np.clip(bar_fraction, 0.08, 0.32))

    x0 = 0.08
    x1 = x0 + bar_fraction
    y = 0.075

    # Draw strictly in axes coordinates so no autoscaling occurs.
    ax.plot(
        [x0, x1],
        [y, y],
        transform=ax.transAxes,
        color="black",
        linewidth=4,
        solid_capstyle="butt",
        clip_on=False,
        zorder=20,
    )

    # Small vertical end ticks.
    tick_half = 0.012
    ax.plot(
        [x0, x0],
        [y - tick_half, y + tick_half],
        transform=ax.transAxes,
        color="black",
        linewidth=2.5,
        clip_on=False,
        zorder=20,
    )
    ax.plot(
        [x1, x1],
        [y - tick_half, y + tick_half],
        transform=ax.transAxes,
        color="black",
        linewidth=2.5,
        clip_on=False,
        zorder=20,
    )

    if bar_km < 1.0:
        label = f"{int(round(bar_km * 1000))} m"
    elif float(bar_km).is_integer():
        label = f"{int(bar_km)} km"
    else:
        label = f"{bar_km:g} km"

    ax.text(
        0.5 * (x0 + x1),
        y + 0.025,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="black",
        zorder=20,
    )

def common_legend():
    return [
        Patch(
            facecolor=COLOR_OUTSIDE,
            edgecolor="0.6",
            label="Non-forest / outside mask",
        ),
        Patch(
            facecolor=COLOR_REFERENCE,
            edgecolor="none",
            label="Reference deforestation",
        ),
        Patch(
            facecolor=COLOR_FP,
            edgecolor="none",
            label="False positive (FP)",
        ),
        Patch(
            facecolor=COLOR_TP,
            edgecolor="none",
            label="Correct prediction (TP)",
        ),
        Patch(
            facecolor=COLOR_FN,
            edgecolor="none",
            label="False negative (FN)",
        ),
    ]


def set_plot_fonts():
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 13,
        }
    )


def draw_six_panel_figure(
    previous_rgb: np.ndarray,
    target_rgb: np.ndarray,
    previous_probability: np.ndarray,
    target_probability: np.ndarray,
    reference_rgb: np.ndarray,
    confusion_rgb: np.ndarray,
    extent,
    crs,
    title: str,
    output_path: Path,
):
    set_plot_fonts()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(
            23,
            13,
        ),
    )

    axes = axes.ravel()

    # -------------------------------------------------------------------------
    # (a)
    # -------------------------------------------------------------------------
    axes[0].imshow(
        previous_rgb,
        extent=extent,
        origin="upper",
    )

    axes[0].set_title(
        "(a) Sentinel-1 VV/VH — 04/04/2022",
        fontsize=17,
    )

    add_north_arrow(
        axes[0]
    )

    add_scale_bar(
        axes[0],
        extent,
        crs,
    )

    # -------------------------------------------------------------------------
    # (b)
    # -------------------------------------------------------------------------
    axes[1].imshow(
        target_rgb,
        extent=extent,
        origin="upper",
    )

    axes[1].set_title(
        "(b) Sentinel-1 VV/VH — 19/09/2022",
        fontsize=17,
    )

    # -------------------------------------------------------------------------
    # (c)
    # -------------------------------------------------------------------------
    im_c = axes[2].imshow(
        previous_probability,
        extent=extent,
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    axes[2].set_title(
        "(c) GMM anomaly probability — 04/04/2022",
        fontsize=17,
    )

    cbar_c = fig.colorbar(
        im_c,
        ax=axes[2],
        fraction=0.046,
        pad=0.025,
    )

    cbar_c.set_label(
        "Probability",
        fontsize=14,
    )

    cbar_c.ax.tick_params(
        labelsize=13
    )

    # -------------------------------------------------------------------------
    # (d)
    # -------------------------------------------------------------------------
    im_d = axes[3].imshow(
        target_probability,
        extent=extent,
        origin="upper",
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    axes[3].set_title(
        "(d) GMM anomaly probability — 19/09/2022",
        fontsize=17,
    )

    cbar_d = fig.colorbar(
        im_d,
        ax=axes[3],
        fraction=0.046,
        pad=0.025,
    )

    cbar_d.set_label(
        "Probability",
        fontsize=14,
    )

    cbar_d.ax.tick_params(
        labelsize=13
    )

    # -------------------------------------------------------------------------
    # (e)
    # -------------------------------------------------------------------------
    axes[4].imshow(
        reference_rgb,
        extent=extent,
        origin="upper",
    )

    axes[4].set_title(
        "(e) Binary reference deforestation map — 2022",
        fontsize=17,
    )

    # -------------------------------------------------------------------------
    # (f)
    # -------------------------------------------------------------------------
    axes[5].imshow(
        confusion_rgb,
        extent=extent,
        origin="upper",
    )

    axes[5].set_title(
        "(f) FADAS (U-Net) prediction — GMM threshold = 2.0%",
        fontsize=17,
    )

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        title,
        fontsize=21,
        fontweight="bold",
        y=0.985,
    )

    fig.legend(
        handles=common_legend(),
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(
            0.5,
            0.018,
        ),
        columnspacing=1.8,
        handlelength=1.8,
    )

    fig.subplots_adjust(
        left=0.025,
        right=0.985,
        top=0.925,
        bottom=0.09,
        wspace=0.12,
        hspace=0.13,
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(
        fig
    )

    print(
        "\nSaved figure:",
        output_path,
    )


# =============================================================================
# 15. INDEX DATA
# =============================================================================

if not TEST_DIR.is_dir():
    raise FileNotFoundError(
        TEST_DIR
    )

if not ERROR_DIR.is_dir():
    raise FileNotFoundError(
        ERROR_DIR
    )

data_index = index_files(
    TEST_DIR,
    error=False,
)

error_index = index_files(
    ERROR_DIR,
    error=True,
)


# =============================================================================
# 16. COMMON 2022 TILES
# =============================================================================

previous_data_tiles = {
    tile
    for (
        date,
        tile,
    )
    in data_index
    if date == PREVIOUS_DATE
}

target_data_tiles = {
    tile
    for (
        date,
        tile,
    )
    in data_index
    if date == TARGET_DATE
}

previous_error_tiles = {
    tile
    for (
        date,
        tile,
    )
    in error_index
    if date == PREVIOUS_DATE
}

target_error_tiles = {
    tile
    for (
        date,
        tile,
    )
    in error_index
    if date == TARGET_DATE
}

common_tiles = sorted(
    previous_data_tiles
    & target_data_tiles
    & previous_error_tiles
    & target_error_tiles
)

print(
    "\nCommon 2022 April/September tiles:",
    len(
        common_tiles
    ),
)

if len(
    common_tiles
) != 49:
    print(
        "WARNING: expected 49 common tiles."
    )

if not common_tiles:
    raise RuntimeError(
        "No common 2022 tiles found."
    )

if TILE_TO_PLOT not in common_tiles:
    raise RuntimeError(
        f"Requested tile {TILE_TO_PLOT} is not in the common-tile set."
    )


# =============================================================================
# 17. FIT GLOBAL GMM
# =============================================================================

gmm, anomaly_component = (
    fit_global_gmm(
        ERROR_DIR
    )
)


# =============================================================================
# 18. MOSAIC GEOMETRY
# =============================================================================

rows = sorted(
    {
        row
        for row, _
        in common_tiles
    }
)

cols = sorted(
    {
        col
        for _, col
        in common_tiles
    }
)

row_lookup = {
    value: index
    for index, value
    in enumerate(
        rows
    )
}

col_lookup = {
    value: index
    for index, value
    in enumerate(
        cols
    )
}

sample_data_path = data_index[
    (
        PREVIOUS_DATE,
        common_tiles[0],
    )
]

sample_image = read_s1_pixels(
    sample_data_path
)

_, TILE_H, TILE_W = (
    sample_image.shape
)

MOSAIC_H = (
    len(rows)
    * TILE_H
)

MOSAIC_W = (
    len(cols)
    * TILE_W
)

print(
    "Tile size:",
    TILE_H,
    "x",
    TILE_W,
)

print(
    "Mosaic size:",
    MOSAIC_H,
    "x",
    MOSAIC_W,
)


# =============================================================================
# 19. CREATE MOSAICS
# =============================================================================

s1_previous = np.full(
    (
        2,
        MOSAIC_H,
        MOSAIC_W,
    ),
    np.nan,
    dtype=np.float32,
)

s1_target = np.full_like(
    s1_previous,
    np.nan,
)

probability_previous = np.full(
    (
        MOSAIC_H,
        MOSAIC_W,
    ),
    np.nan,
    dtype=np.float32,
)

probability_target = np.full_like(
    probability_previous,
    np.nan,
)

prediction_mosaic = np.zeros(
    (
        MOSAIC_H,
        MOSAIC_W,
    ),
    dtype=np.uint8,
)

bounds_list = []
mosaic_crs = None


# Keep data needed for the requested single tile.
selected_tile_data = None


# =============================================================================
# 20. BUILD FULL MOSAICS
# =============================================================================

print(
    "\nBuilding complete-study-area mosaics..."
)

for index, tile in enumerate(
    common_tiles,
    start=1,
):
    row, col = tile

    rr = row_lookup[
        row
    ]

    cc = col_lookup[
        col
    ]

    y0 = (
        rr
        * TILE_H
    )

    y1 = (
        y0
        + TILE_H
    )

    x0 = (
        cc
        * TILE_W
    )

    x1 = (
        x0
        + TILE_W
    )

    previous_source_path = data_index[
        (
            PREVIOUS_DATE,
            tile,
        )
    ]

    target_source_path = data_index[
        (
            TARGET_DATE,
            tile,
        )
    ]

    previous_error_path = error_index[
        (
            PREVIOUS_DATE,
            tile,
        )
    ]

    target_error_path = error_index[
        (
            TARGET_DATE,
            tile,
        )
    ]

    # -------------------------------------------------------------------------
    # Read Sentinel-1 correctly with tifffile.
    # -------------------------------------------------------------------------
    previous_s1_tile = read_s1_pixels(
        previous_source_path
    )

    target_s1_tile = read_s1_pixels(
        target_source_path
    )

    if (
        previous_s1_tile.shape
        != target_s1_tile.shape
    ):
        raise ValueError(
            f"Sentinel-1 shape mismatch for {tile}: "
            f"{previous_s1_tile.shape} vs {target_s1_tile.shape}"
        )

    # Diagnostic for the April image.
    if tile == TILE_TO_PLOT:
        print(
            "\nRequested tile Sentinel-1 diagnostics:"
        )

        for name, image in (
            (
                "04/04/2022",
                previous_s1_tile,
            ),
            (
                "19/09/2022",
                target_s1_tile,
            ),
        ):
            print(
                name,
                "| shape =",
                image.shape,
                "| VV min/max =",
                float(
                    np.nanmin(
                        image[0]
                    )
                ),
                float(
                    np.nanmax(
                        image[0]
                    )
                ),
                "| VH min/max =",
                float(
                    np.nanmin(
                        image[1]
                    )
                ),
                float(
                    np.nanmax(
                        image[1]
                    )
                ),
            )

    s1_previous[
        :,
        y0:y1,
        x0:x1,
    ] = previous_s1_tile

    s1_target[
        :,
        y0:y1,
        x0:x1,
    ] = target_s1_tile

    # -------------------------------------------------------------------------
    # Errors -> global GMM posterior probability.
    # -------------------------------------------------------------------------
    previous_error_tile = read_error_sum(
        previous_error_path
    )

    target_error_tile = read_error_sum(
        target_error_path
    )

    previous_probability_tile = probability_map(
        previous_error_tile,
        gmm,
        anomaly_component,
    )

    target_probability_tile = probability_map(
        target_error_tile,
        gmm,
        anomaly_component,
    )

    probability_previous[
        y0:y1,
        x0:x1,
    ] = previous_probability_tile

    probability_target[
        y0:y1,
        x0:x1,
    ] = target_probability_tile

    # -------------------------------------------------------------------------
    # Change map using the exact GMM 2% decision.
    # -------------------------------------------------------------------------
    prediction_tile = (
        change_from_probabilities(
            previous_probability_tile,
            target_probability_tile,
        )
    )

    prediction_mosaic[
        y0:y1,
        x0:x1,
    ] = prediction_tile

    # -------------------------------------------------------------------------
    # Georeferencing.
    # -------------------------------------------------------------------------
    (
        tile_transform,
        tile_crs,
        tile_bounds,
        _,
        _,
    ) = read_georef(
        target_error_path
    )

    bounds_list.append(
        tile_bounds
    )

    mosaic_crs = (
        tile_crs
    )

    if tile == TILE_TO_PLOT:
        selected_tile_data = {
            "previous_s1": (
                previous_s1_tile.copy()
            ),
            "target_s1": (
                target_s1_tile.copy()
            ),
            "previous_probability": (
                previous_probability_tile.copy()
            ),
            "target_probability": (
                target_probability_tile.copy()
            ),
            "prediction": (
                prediction_tile.copy()
            ),
            "transform": (
                tile_transform
            ),
            "crs": (
                tile_crs
            ),
            "bounds": (
                tile_bounds
            ),
            "y0": y0,
            "y1": y1,
            "x0": x0,
            "x1": x1,
        }

    if (
        index % 10 == 0
        or index == len(
            common_tiles
        )
    ):
        print(
            f"  {index}/{len(common_tiles)} tiles"
        )


if selected_tile_data is None:
    raise RuntimeError(
        "Requested tile data were not captured."
    )


# =============================================================================
# 21. COMPLETE STUDY-AREA EXTENT
# =============================================================================

left = min(
    bounds.left
    for bounds
    in bounds_list
)

right = max(
    bounds.right
    for bounds
    in bounds_list
)

bottom = min(
    bounds.bottom
    for bounds
    in bounds_list
)

top = max(
    bounds.top
    for bounds
    in bounds_list
)

full_extent = [
    left,
    right,
    bottom,
    top,
]

mosaic_transform = from_bounds(
    left,
    bottom,
    right,
    top,
    MOSAIC_W,
    MOSAIC_H,
)

print(
    "\nComplete-area extent:",
    full_extent,
)


# =============================================================================
# 22. READ REFERENCE / FOREST MASK
# =============================================================================

METRICS_ROOT = (
    find_metrics_root()
)

print(
    "Metrics root:",
    METRICS_ROOT,
)

forest = read_vector(
    METRICS_ROOT
    / "Forest_Mask_2021.shp"
)

reference = read_vector(
    METRICS_ROOT
    / "622_975_2022.shp"
)

forest = forest.to_crs(
    mosaic_crs
)

reference = reference.to_crs(
    mosaic_crs
)

forest_mask = rasterize_vector(
    forest,
    mosaic_transform,
    (
        MOSAIC_H,
        MOSAIC_W,
    ),
)

reference_mask = rasterize_vector(
    reference,
    mosaic_transform,
    (
        MOSAIC_H,
        MOSAIC_W,
    ),
)

# Restrict the reference to forest, consistently with evaluation.
reference_mask = (
    reference_mask
    * forest_mask
)

prediction_mosaic = (
    prediction_mosaic
    * forest_mask
)


# =============================================================================
# 23. COMPLETE-AREA RGB DISPLAY
# =============================================================================

full_rgb_previous, full_rgb_target = (
    make_s1_rgb_pair(
        s1_previous,
        s1_target,
    )
)

full_reference_rgb = (
    make_reference_rgb(
        reference_mask,
        forest_mask,
    )
)

full_confusion_rgb = (
    make_confusion_rgb(
        reference_mask,
        prediction_mosaic,
        forest_mask,
    )
)


# =============================================================================
# 24. COMPLETE-AREA METRICS — DISPLAY CHECK
# =============================================================================

full_tp = int(
    np.sum(
        (
            reference_mask == 1
        )
        & (
            prediction_mosaic == 1
        )
    )
)

full_fp = int(
    np.sum(
        (
            reference_mask == 0
        )
        & (
            prediction_mosaic == 1
        )
        & (
            forest_mask == 1
        )
    )
)

full_fn = int(
    np.sum(
        (
            reference_mask == 1
        )
        & (
            prediction_mosaic == 0
        )
    )
)

full_iou = (
    full_tp
    / (
        full_tp
        + full_fp
        + full_fn
    )
    if (
        full_tp
        + full_fp
        + full_fn
    ) > 0
    else 0.0
)

print(
    "\nComplete-area raster-display check:",
    f"TP={full_tp}",
    f"FP={full_fp}",
    f"FN={full_fn}",
    f"IoU={full_iou * 100:.2f}%",
)


# =============================================================================
# 25. SAVE COMPLETE STUDY-AREA FIGURE
# =============================================================================

draw_six_panel_figure(
    previous_rgb=full_rgb_previous,
    target_rgb=full_rgb_target,
    previous_probability=probability_previous,
    target_probability=probability_target,
    reference_rgb=full_reference_rgb,
    confusion_rgb=full_confusion_rgb,
    extent=full_extent,
    crs=mosaic_crs,
    title=(
        "FADAS (U-Net) — GMM threshold = 2.0% — "
        "complete study area"
    ),
    output_path=FULL_FIGURE_PATH,
)


# =============================================================================
# 26. SINGLE TILE MASKS
# =============================================================================

tile_y0 = selected_tile_data[
    "y0"
]

tile_y1 = selected_tile_data[
    "y1"
]

tile_x0 = selected_tile_data[
    "x0"
]

tile_x1 = selected_tile_data[
    "x1"
]

tile_forest_mask = forest_mask[
    tile_y0:tile_y1,
    tile_x0:tile_x1,
].copy()

tile_reference_mask = reference_mask[
    tile_y0:tile_y1,
    tile_x0:tile_x1,
].copy()

tile_prediction_mask = prediction_mosaic[
    tile_y0:tile_y1,
    tile_x0:tile_x1,
].copy()


# =============================================================================
# 27. SINGLE TILE RGB
# =============================================================================

tile_rgb_previous, tile_rgb_target = (
    make_s1_rgb_pair(
        selected_tile_data[
            "previous_s1"
        ],
        selected_tile_data[
            "target_s1"
        ],
    )
)

tile_reference_rgb = (
    make_reference_rgb(
        tile_reference_mask,
        tile_forest_mask,
    )
)

tile_confusion_rgb = (
    make_confusion_rgb(
        tile_reference_mask,
        tile_prediction_mask,
        tile_forest_mask,
    )
)

tile_bounds = selected_tile_data[
    "bounds"
]

tile_extent = [
    tile_bounds.left,
    tile_bounds.right,
    tile_bounds.bottom,
    tile_bounds.top,
]


# =============================================================================
# 28. SINGLE-TILE METRICS — DISPLAY CHECK
# =============================================================================

tile_tp = int(
    np.sum(
        (
            tile_reference_mask == 1
        )
        & (
            tile_prediction_mask == 1
        )
    )
)

tile_fp = int(
    np.sum(
        (
            tile_reference_mask == 0
        )
        & (
            tile_prediction_mask == 1
        )
        & (
            tile_forest_mask == 1
        )
    )
)

tile_fn = int(
    np.sum(
        (
            tile_reference_mask == 1
        )
        & (
            tile_prediction_mask == 0
        )
    )
)

tile_iou = (
    tile_tp
    / (
        tile_tp
        + tile_fp
        + tile_fn
    )
    if (
        tile_tp
        + tile_fp
        + tile_fn
    ) > 0
    else 0.0
)

print(
    "\nTile 1024_1280 raster-display check:",
    f"TP={tile_tp}",
    f"FP={tile_fp}",
    f"FN={tile_fn}",
    f"IoU={tile_iou * 100:.2f}%",
)


# =============================================================================
# 29. SAVE SINGLE-TILE FIGURE
# =============================================================================

draw_six_panel_figure(
    previous_rgb=tile_rgb_previous,
    target_rgb=tile_rgb_target,
    previous_probability=selected_tile_data[
        "previous_probability"
    ],
    target_probability=selected_tile_data[
        "target_probability"
    ],
    reference_rgb=tile_reference_rgb,
    confusion_rgb=tile_confusion_rgb,
    extent=tile_extent,
    crs=selected_tile_data["crs"],
    title=(
        "FADAS (U-Net) — GMM threshold = 2.0% — "
        "tile 1024_1280"
    ),
    output_path=TILE_FIGURE_PATH,
)


# =============================================================================
# 30. FINAL
# =============================================================================

print(
    "\n"
    + "=" * 100
)

print(
    "DONE"
)

print(
    "=" * 100
)

print(
    "Model:",
    MODEL_LABEL,
)

print(
    "Technique: Global GMM posterior"
)

print(
    "Threshold:",
    f"{GMM_THRESHOLD * 100:.1f}%",
)

print(
    "Dates:",
    PREVIOUS_DATE,
    "->",
    TARGET_DATE,
)

print(
    "Full study-area figure:",
    FULL_FIGURE_PATH,
)

print(
    "Tile figure:",
    TILE_FIGURE_PATH,
)

print(
    "=" * 100
)
