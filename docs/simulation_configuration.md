# Simulation configuration reference

Technical reference for the two MPAS-Atmosphere regional simulations analysed
for the EarthSyms 2026 poster. The authoritative, full description lives with
the simulations themselves:

- `runs/meqbr_05km/README.md` (under the MPAS-Research tree; see `data_root` in
  `config/paths.local.yaml`) — full setup, output variable table, and
  operational notes.
- Per-run archived `namelist.atmosphere`, `namelist.init_atmosphere`,
  `streams.atmosphere`, `streams.init_atmosphere` inside each dated run
  directory — the exact configuration used.

This file summarizes what the analysis code needs; it does not duplicate the
source README. Structured values are in `config/simulations.yaml`.

## Mesh and domain (shared)

- Mesh `meqbr_05km`: variable-resolution, quasi-uniform at ~4.6 km mean cell
  spacing in the region of interest; `nCells = 76813`; `config_len_disp = 5000 m`.
- Domain (cell extent): lat −7.7° to 6.7°, lon −55.0° to −32.7° (Brazilian
  equatorial margin, onshore + offshore).
- Shared static/decomposition files (`meqbr_05km.static.nc`,
  `meqbr_05km.graph.info.part.80`) live in the parent run directory and are
  used by both simulations.

## Vertical grid (shared)

- 55 layers (`nVertLevels`), 56 interfaces (`nVertLevelsP1`); hybrid coordinate,
  model top at 30 km, terrain-following and smoothed near the surface.
- Interface heights AGL are set explicitly (`zeta_wind.txt`): 0, 25, 75, 125,
  175, 225, … m, i.e. 50 m spacing above the first (0–25 m) layer.
- **Staggering (critical for extraction):** `w` and `zgrid` are on interfaces;
  horizontal wind (`uReconstructZonal`/`uReconstructMeridional`), `theta`,
  `rho`, `pressure` are at layer centers (midpoints of adjacent interfaces).
- The interface set was designed so that **layer-center index 2 falls at exactly
  50 m above local terrain** — the turbine-relevant sampling height. Centers
  then continue at 100, 150, 200 m. Extraction code must derive center heights
  from `zgrid` and validate the index, not assume it.

## Forcing (shared method)

- Initial + 6-hourly lateral boundary conditions from ERA5 reanalysis
  (`config_fg_interval = 21600 s`).
- SST: NOAA OISST v2.1 daily interpolated to a surface-update file, but SST
  updating was **disabled** in both runs (`config_sst_update = false`); SST is
  held at the initial ERA5 skin temperature.

## Physics / dynamics (identical in both runs)

- `config_physics_suite = mesoscale_reference` (WSM6, RRTMG LW/SW, YSU PBL,
  Monin–Obukhov surface layer, Tiedtke convection); Noah-MP land surface.
- Time step 30 s; RK3 split-explicit dynamics; `2d_smagorinsky` horizontal
  mixing; limited-area lateral boundaries applied; 80 MPI ranks.

## Periods and output

Hourly `history.*.nc` output (single `output` stream, `wind_energy` variable
list). Only the retained **analysis-period** history is kept per run.

| Sim | Analysis period | Spin-up (discarded) | Full integration | Notes |
|---|---|---|---|---|
| `sim_2021` | 2021-11-01 → 2021-12-01 | 2021-10-21 → 2021-10-31 | 2021-10-21 → 2021-12-01 | fresh ERA5 init 2021-10-21 |
| `sim_2022` | 2022-10-01 → 2022-11-01 | 2022-09-21 → 2022-09-30 | 2022-09-21 → 2022-11-01 | ERA5 init 2022-09-21; **restarted** at 2022-10-01 before continuing |

- Retained history: 721 hourly records (2021) and 745 (2022), inclusive of both
  endpoints.
- **Accumulated precipitation (`rainnc`, `rainc`) counts from the full
  integration start (spin-up), not the analysis boundary**, and is never reset —
  compute rates by differencing consecutive hourly values.
