# data/

In-situ observations used to validate the `meqbr_05km` simulations. The MPAS
raw output itself is **not** here — it is read in place from `data_root`
(see `config/paths.local.yaml`); this folder is only for external
observational data small enough to version.

Site metadata (coordinates, instrument, variables, period) is tracked
centrally in `config/simulations.yaml` under `observations:` — read it via
`mpas_analysis.config.load_config().observation("P0")` rather than
hardcoding coordinates in analysis scripts.

## P0_LIDAR_matrix.mat

Floating-LiDAR wind/met profiler, site **P0** (-2.694107, -42.554807 —
Maranhão coast, within the `meqbr_05km` domain).

- **Format**: MATLAB v5 struct (field `L`), one row per 10-min record.
- **Period**: 2021-11-09 to 2021-12-13 (4831 records) — fully inside
  `sim_2021`'s analysis window (2021-11-01 to 2021-12-01), so this is the
  simulation to compare it against.
- **Heights**: 20 levels, 40-260 m AGL (`L.heights`, same for every record).
- **Key fields**: `mtime` (MATLAB datenum), `wspeed`/`wdir`/`vertspeed`
  (per height), `temp`/`press`/`humid` (single sensor level), `TI`, `CNR`,
  `avail` (per-height data-completeness percentage for that 10-min bin —
  quicklook scripts mask bins below a threshold on this field).
- **Known issue**: the met sensor (`temp`/`press`/`humid`) drops out after the
  first ~140 records (~23 h) and reports an exact-0 fill value for the rest of
  the deployment — 0 hPa/0 degC is unphysical, not a real reading. The LiDAR
  wind fields (`wspeed`/`wdir`/...) are unaffected and stay valid throughout.
  `quicklook_lidar_p0.py` masks `temp`/`press` on `press <= 0`.
- **Provenance**: received as a pre-processed `.mat` export from the
  instrument operator (file timestamp 2022-10-21); no raw download script —
  document any reprocessing here if the file is ever regenerated.

Read with `scipy.io.loadmat(path, simplify_cells=True)["L"]`. MATLAB datenum
converts to a pandas timestamp via:
`pd.Timestamp("1970-01-01") + pd.to_timedelta(mtime - 719529, unit="D")`.

See `scripts/exploratory/quicklook_lidar_p0.py` for a worked example.

## LPI_processed.csv.gz

Fixed LiDAR, site **LPI** ("Porto-Ilha", -4.8789425, -37.1478801 — Rio
Grande do Norte coast, within the `meqbr_05km` domain). Nearest MPAS cell is
ocean, ~1.8 km away.

- **Raw source**: `data/lidar_porto_ilha_5niveis_final_final.xlsx` (~87 MB,
  **gitignored** — `data/*.xlsx`), sheet `lidar_porto_ilha_screened_local`
  (the workbook's other sheets are derived/exploratory summaries, not used).
  Preprocessed into this tracked CSV by
  `scripts/exploratory/preprocess_lpi_lidar.py`; rerun that script if the raw
  file is replaced.
- **Period**: 2022-06-23 to 2025-06-06 (151,486 records at 10-min
  resolution). Only **`sim_2022`** (2022-10-01 to 2022-11-01) falls inside
  this span — **no overlap with `sim_2021`** at all, unlike P0. Coverage in
  October 2022 is >99% non-null at every height.
- **Heights**: 200, 150, 100, 50, 26, 10 m — the first four line up almost
  exactly with the model's near-surface layers (50/100/150/200 m, like P0);
  26 m and 10 m fall between the model's 12.5 m and 50 m levels with no
  clean match (also true of P0's 40 m channel).
- **Preprocessing decisions** (see the script's docstring for full detail):
  - Raw timestamps are labeled `Timestamp (UTC-03)` (local time); converted
    to UTC (+3h) to match MPAS `xtime`.
  - The raw sheet has **two** blocks of `Dir <height>m` columns that
    disagree by exactly 180 deg for ~2-3% of rows. Kept the second
    (rightmost) block — a leftover cached formula fragment in one cell
    (`=Z953-180`) indicates it is the corrected column — and dropped the
    first.
  - Dropped the `offset` column (values only ever `{None, 21}`, undocumented
    in the sheet, not physically interpretable on its own).
  - `RH 10m` is null for the first ~2 months of the deployment (mid-2022)
    but populated later — kept as-is (`rh_10`), not dropped, since it isn't
    uniformly empty.

Read with `pandas.read_csv(path, parse_dates=["time"])`; `time` is already UTC.
