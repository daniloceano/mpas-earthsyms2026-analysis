#!/usr/bin/env python
"""Preprocess the raw Porto-Ilha LiDAR (LPI) spreadsheet into a small,
git-trackable CSV — the raw file is an 87 MB export, gitignored
(data/*.xlsx; see data/README.md), so every other LPI script reads the
processed CSV produced here instead of touching the spreadsheet directly.

Source: data/lidar_porto_ilha_5niveis_final_final.xlsx, sheet
"lidar_porto_ilha_screened_local" (the QC-screened production sheet; the
workbook's other sheets are derived/exploratory summaries not used here).

Decisions made while preprocessing (see also data/README.md):
  - Timestamps are labeled "Timestamp (UTC-03)" (local time); converted to
    UTC by adding 3h, to match the MPAS xtime convention used everywhere
    else in this repo.
  - The sheet has two blocks of "Dir <height>m [deg]" columns (raw columns
    7-12 and 25-30) that disagree by exactly 180 deg for ~2-3% of rows. A
    cached formula fragment surviving in one cell ("=Z953-180") indicates
    the second (rightmost) block is a corrected column, so it is kept and
    the first block is dropped.
  - Dropped the "offset" column: only seen taking values {None, 21}, with
    no documentation in the sheet of what it means.
  - "50m A" / "26m A" columns kept as heights 50 m / 26 m — the "A" suffix
    is unexplained in the source; values look like ordinary speed/direction
    readings, not a separate adjusted duplicate.
  - Uses raw openpyxl (not pandas.read_excel) to read the workbook: this
    machine's pandas/openpyxl combination in some envs raises a version
    conflict on read_excel, and iterating cells directly sidesteps it.

    python scripts/exploratory/preprocess_lpi_lidar.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import openpyxl
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from mpas_analysis import config as cfg  # noqa: E402

RAW_PATH = cfg.REPO_ROOT / "data" / "lidar_porto_ilha_5niveis_final_final.xlsx"
OUT_PATH = cfg.REPO_ROOT / "data" / "LPI_processed.csv.gz"
SHEET = "lidar_porto_ilha_screened_local"

# raw column name -> (output name, whether raw is a value we keep as-is)
KEEP_FIRST_BLOCK = {
    "Timestamp (UTC-03)": "time_local",
    "Spd 200m [m/s]": "spd_200",
    "Spd 150m [m/s]": "spd_150",
    "Spd 100m [m/s]": "spd_100",
    "Spd 50m A [m/s]": "spd_50",
    "Spd 26m A [m/s]": "spd_26",
    "Spd 10m [m/s]": "spd_10",
    "Tmp 10m [°C]": "temp_10",
    "Pres 10m [mbar]": "pres_10",
    "RH 10m [%]": "rh_10",
    "TI 200m": "ti_200",
    "TI 150m": "ti_150",
    "TI 100m": "ti_100",
    "TI 50m A": "ti_50",
    "TI 26m A": "ti_26",
    "wind_speed_std_dev_m_s_200m [m/s]": "wspd_std_200",
    "wind_speed_std_dev_m_s_150m [m/s]": "wspd_std_150",
    "wind_speed_std_dev_m_s_100m [m/s]": "wspd_std_100",
}
# Second occurrence of each of these names = the offset-corrected direction.
DIR_NAMES = [
    "Dir 200m [°]", "Dir 150m [°]", "Dir 100m [°]",
    "Dir 50m A [°]", "Dir 26m A [°]", "Dir 10m [°]",
]
DIR_OUT = ["dir_200", "dir_150", "dir_100", "dir_50", "dir_26", "dir_10"]


def main() -> int:
    if not RAW_PATH.exists():
        print(f"missing raw file: {RAW_PATH}", file=sys.stderr)
        return 1

    print(f"reading {RAW_PATH.name} (sheet={SHEET})...")
    wb = openpyxl.load_workbook(RAW_PATH, read_only=True, data_only=True)
    ws = wb[SHEET]

    row_iter = ws.iter_rows(values_only=True)
    header = next(row_iter)

    # Index of the *second* occurrence of each Dir column name.
    dir_second_idx = {}
    for name in DIR_NAMES:
        idxs = [i for i, h in enumerate(header) if h == name]
        if len(idxs) != 2:
            raise SystemExit(
                f"expected exactly 2 occurrences of column {name!r}, found "
                f"{len(idxs)} — sheet layout may have changed.")
        dir_second_idx[name] = idxs[1]

    first_block_idx = {}
    for name in KEEP_FIRST_BLOCK:
        if name not in header:
            raise SystemExit(f"expected column not found: {name!r}")
        first_block_idx[name] = header.index(name)

    out_cols = (["time_local"] + list(KEEP_FIRST_BLOCK.values())[1:] + DIR_OUT)
    records: list[list] = []
    for row in row_iter:
        rec = [row[first_block_idx[name]] for name in KEEP_FIRST_BLOCK]
        rec += [row[dir_second_idx[name]] for name in DIR_NAMES]
        records.append(rec)
    wb.close()

    df = pd.DataFrame.from_records(records, columns=out_cols)
    df["time"] = pd.to_datetime(df["time_local"]) + pd.Timedelta(hours=3)
    df = df.drop(columns=["time_local"])
    df = df[["time"] + [c for c in df.columns if c != "time"]]
    df = df.sort_values("time").reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False, compression="gzip")
    print(f"wrote {OUT_PATH} ({len(df)} records, "
          f"{df['time'].min()} to {df['time'].max()} UTC)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
