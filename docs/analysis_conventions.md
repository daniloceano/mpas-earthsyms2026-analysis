# Analysis conventions

Conventions every script and notebook in this repo must follow. They encode
MPAS-specific behavior that is easy to get wrong.

## Time

- Use `xtime` (string `YYYY-MM-DD_hh:mm:ss`) as the authoritative valid time of
  each record; decode it to timestamps rather than trusting file order alone
  (though for these runs filenames are also chronological). Helper:
  `mpas_analysis.io.read_timestamps`.

## Geography

- `latCell` / `lonCell` are in **radians** — convert to degrees before any
  geographic plotting. Helper: `mpas_analysis.io.cell_lonlat_degrees` (also
  wraps longitude to (−180, 180]).

## Vertical grid

- Derive layer-center heights as the midpoint of adjacent `zgrid` interfaces
  (`mpas_analysis.vertical.layer_center_heights`). `zgrid` and `w` are on
  interfaces; winds/theta/rho/pressure are at centers.
- For height-specific extraction (e.g. 50 m wind), compute center heights above
  ground (`heights_above_ground`) and pick the nearest level
  (`nearest_level_index`). **Do not assume a fixed numeric vertical index across
  files without validating the derived heights.**
- **1-based vs 0-based indexing:** the source README describes the ~50 m level
  as "level 2" (MPAS/Fortran is 1-based). In 0-based numpy/xarray that is index
  **1**. `config/simulations.yaml` records both (`turbine_level_index_fortran`,
  `turbine_level_index_python`); code uses the Python one and still validates the
  derived height.

## Precipitation

- Total accumulated precipitation is `rainnc + rainc`.
- Hourly precipitation is the difference between consecutive accumulated values.
- Accumulation **started during spin-up** (full integration start), not at the
  retained analysis boundary, and is not reset — the first retained record
  already carries spin-up accumulation, so always difference.

## Masks

- `landmask` (static, 1 = land, 0 = ocean) and `xland` (time-varying, 1 = land
  incl. sea-ice, 2 = ocean) use **different encodings** — keep them distinct and
  do not mix their conventions.

## Memory

- Never silently load a full simulation into memory. Read only the variables and
  time slices needed; use lazy/chunked access (`open_history(..., chunks=...)`)
  for anything spanning many files or 3D fields.
