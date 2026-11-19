#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Verify that the spatial locations used for the yearly deforestation evaluation
are contained in the same reconstruction test dataset.

The reconstruction test set contains many date-specific TIFF observations.
The yearly deforestation evaluation uses 49 fixed spatial locations and two
dates per year. This utility verifies the exact date/tile pairs.

Path:
    TEST_DIR must point to the SAME test/processed directory used for the
    1,519-image reconstruction evaluation.
"""

from pathlib import Path
import re
import pandas as pd

# External dataset test directory used by the reconstruction experiment.
TEST_DIR = Path("PATH/TO/DATASET/forest/test/processed")

YEAR_CONFIG = {
    2021: {"previous_date": "20210409", "target_date": "20210912"},
    2022: {"previous_date": "20220404", "target_date": "20220919"},
}

DATE_RE = re.compile(r"(20\d{6})T?")
TILE_PATTERNS = [
    re.compile(r"_db_(\d+)_(\d+)"),
    re.compile(r"_(\d+)_(\d+)(?:\.tif|_)$", re.IGNORECASE),
]


def parse_file(path: Path):
    name = path.name
    dm = DATE_RE.search(name)
    if dm is None:
        return None

    tile = None
    for pattern in TILE_PATTERNS:
        tm = pattern.search(name)
        if tm is not None:
            tile = (int(tm.group(1)), int(tm.group(2)))
            break

    if tile is None:
        return None

    return {"path": path, "date": dm.group(1), "tile": tile}


if not TEST_DIR.is_dir():
    raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")

all_tiffs = sorted(
    list(TEST_DIR.rglob("*.tif"))
    + list(TEST_DIR.rglob("*.tiff"))
)

records = []
unparsed = []

for path in all_tiffs:
    item = parse_file(path)
    (records if item is not None else unparsed).append(item if item is not None else path)

lookup = {}
for record in records:
    lookup.setdefault((record["date"], record["tile"]), []).append(record["path"])

all_test_tiles = {record["tile"] for record in records}

print("Reconstruction TIFF files:", len(all_tiffs))
print("Successfully parsed:", len(records))
print("Unparsed:", len(unparsed))
print("Unique spatial locations:", len(all_test_tiles))

summary = []

for year, cfg in YEAR_CONFIG.items():
    prev = cfg["previous_date"]
    target = cfg["target_date"]

    prev_tiles = {tile for date, tile in lookup if date == prev}
    target_tiles = {tile for date, tile in lookup if date == target}
    common = sorted(prev_tiles & target_tiles)

    missing = [
        tile for tile in common
        if tile not in all_test_tiles
    ]

    complete = [
        tile for tile in common
        if (prev, tile) in lookup and (target, tile) in lookup
    ]

    print("\n" + "=" * 80)
    print("YEAR:", year)
    print("=" * 80)
    print("Previous-date tiles:", len(prev_tiles))
    print("Target-date tiles:", len(target_tiles))
    print("Common spatial tiles:", len(common))
    print("Inside reconstruction test set:", len(common) - len(missing))
    print("Complete date pairs:", len(complete))
    print("Missing:", len(missing))

    summary.append({
        "year": year,
        "previous_date": prev,
        "target_date": target,
        "previous_tiles": len(prev_tiles),
        "target_tiles": len(target_tiles),
        "common_tiles": len(common),
        "inside_reconstruction_test": len(common) - len(missing),
        "complete_pairs": len(complete),
        "missing": len(missing),
    })

summary_df = pd.DataFrame(summary)

print("\nFINAL SUMMARY")
print(summary_df.to_string(index=False))

all_ok = all(
    row["common_tiles"] == row["inside_reconstruction_test"]
    and row["common_tiles"] == row["complete_pairs"]
    for row in summary
)

print("\nSTRICT VERIFICATION:", "PASS" if all_ok else "FAIL")
