# Scientific Notes — mpas-earthsyms2026-analysis

## Research Questions

1. How well does the MPAS-Atmosphere regional hindcast (mesh `meqbr_05km`,
   ~4.6 km spacing) reproduce observed near-surface/hub-height wind speed at
   two in-situ LiDAR sites (P0, floating; LPI, fixed) along the Brazilian
   equatorial margin, across the vertical levels relevant to offshore wind
   resource assessment (50–250 m)?
2. Does the model reproduce the sea-breeze circulation (thermally driven,
   diurnal) near the coast, in terms of the timing, vertical extent, and
   structure of the onshore/offshore flow reversal?

## Physical / Statistical Framework

- Model-observation agreement is summarized with the classical pair
  (bias, RMSE, Pearson R) plus the Takacs (1985) mean-square-error
  decomposition into dissipative and dispersive components
  (`mpas_analysis.verification.mse_decomposition`), such that
  $\mathrm{MSE}_{total} = \mathrm{bias}^2 + \mathrm{MSE}_{diss} + \mathrm{MSE}_{disp}$
  exactly.
- The sea-breeze moment is identified per site as the time of maximum 2 m
  (or near-surface, ~100 m target height) temperature on the windiest day of
  the analysis window (`mpas_analysis.seabreeze.find_moments`), i.e. the
  moment sea-breeze forcing is expected to be strongest.

## Datasets and Variables

### Model
- MPAS-Atmosphere regional hindcast, mesh `meqbr_05km` (n_cells = 76813,
  mean spacing ~4.6 km), hourly history output (`history.*.nc`).
- Two runs, same mesh/physics, different periods: `sim_2021`
  (2021-11-01–2021-12-01, overlaps P0) and `sim_2022`
  (2022-10-01–2022-11-01, overlaps LPI).
- Vertical grid: 55 layer centers / 56 interfaces, terrain-following,
  spaced so a layer center sits ~50 m above local terrain (turbine-relevant).
  Layer-center heights AGL are derived per cell from `zgrid` in
  `{mesh}.init.nc` via `mpas_analysis.vertical.layer_center_heights`.
- **Terrain elevation used in figures is the model's own terrain**, not an
  external DEM: `fig02_xsection_seabreeze.py` reads `zgrid` from
  `{mesh_name}.init.nc` for the cells sampled along the transect and takes
  the lowest interface, `ter_cols = zgrid_full[0]`, as the terrain height at
  each transect column. There is no SRTM/ASTER or other externally-sourced
  topography anywhere in this figure — the "topography" is exactly the
  elevation MPAS solved its dynamics on.

### Observations
- **P0** (floating LiDAR): lat −2.694107, lon −42.554807; `.mat` file
  (`data/P0_LIDAR_matrix.mat`); native temporal resolution 10 min; heights
  40–260 m (20 channels, non-uniform spacing above 200 m: 220/240/260 m);
  coverage 2021-11-09 to 2021-12-13, fully inside `sim_2021`'s window; each
  record has a data-availability flag (`avail`, percent).
- **LPI** (fixed LiDAR, Porto-Ilha): lat −4.8789425, lon −37.1478801;
  preprocessed CSV (`data/LPI_processed.csv.gz`); native temporal resolution
  10 min; heights 10/26/50/100/150/200 m; coverage 2022-06-23 to
  2025-06-06, overlapping only `sim_2022`.

## Methodology

### Model–observation pairing for the wind-speed scatterplots (fig03, fig04)

- Model side: hourly wind speed at the single MPAS cell nearest each site
  (`uReconstructZonal`, `uReconstructMeridional` combined), read from the
  full history record of the overlapping simulation. Vertical levels used
  are the model's native terrain-following layer centers, matched to a
  target height AGL (`model_level_index`, tolerance 1 m) rather than
  interpolated to a fixed height grid.
- Observation side: LiDAR record nearest in time to each model hour, kept
  at native 10-min resolution (no resampling/averaging to hourly — a single
  closest 10-min bin represents each model hour).
- **Temporal pairing rule**: `pandas.merge_asof(..., direction="nearest",
  tolerance=pd.Timedelta(minutes=5))` — i.e. an explicit ±5 minute
  tolerance (`--tolerance-min`, default 5.0), not exact-timestamp
  coincidence. Model/observation pairs with no LiDAR record within that
  window are dropped (`dropna`), so the pooled sample is implicitly
  restricted to hours with LiDAR data available within 5 minutes.
- **Vertical-level correspondence**:
  - P0 vs model: 50/100/150/200 m match an LiDAR channel exactly; the
    model's ~250 m layer has no exact P0 channel, so it is compared against
    the **mean of the 240 m and 260 m LiDAR channels**
    (`LEVEL_MATCHES` in `fig03_scatter_wspeed_p0_all_levels.py`). This
    correspondence (which model layer sits near which LiDAR channel(s)) was
    established beforehand with
    `scripts/exploratory/list_model_levels_at_p0.py`.
  - LPI vs model: 50/100/150/200 m all match exactly; no averaging needed
    (`scripts/exploratory/list_model_levels_at_p0.py --site LPI`).
  - All five (P0) / four (LPI) matched levels are pooled into a single
    density scatter (`gaussian_kde`-colored) rather than shown per level,
    with per-level N reported in the figure caption for traceability.
- **Quality control on P0 observations**: LiDAR wind-speed records are
  masked to NaN wherever `avail < --min-avail` (default 50%) *before*
  temporal pairing (`fig03_scatter_wspeed_p0_all_levels.py`). LPI has no
  equivalent flag applied in `fig04`.
- Fit statistics reported per figure: N, Pearson R, RMSE, bias, and the
  Takacs (1985) MSE dissipative/dispersive decomposition, computed on the
  pooled (all-levels) sample.

### Sea-breeze cross-section (fig02)

- Transect: fixed-longitude, meridional line through the site
  (`site["lon"]`), sampled at 400 points across ±2° latitude by default,
  snapped to nearest MPAS cells via a `cKDTree` nearest-neighbor query on
  cell-center lat/lon; consecutive duplicate cells collapsed
  (`sample_meridional_transect`).
- Fields (potential temperature, pressure, zonal/meridional/vertical wind)
  are read from the single history file matching the identified sea-breeze
  (Tmax) moment, at the transect's cells only.
- Terrain: as noted above, read directly from `{mesh}.init.nc` `zgrid`
  (model terrain, not an external DEM).

## Assumptions

- **5-minute pairing tolerance** (fig03/fig04): treats a LiDAR record up to
  5 minutes away from a model hour as representative of that hour. Not
  validated against a sensitivity sweep of the tolerance value in this
  repo; chosen as a value much smaller than the sea-breeze/diurnal
  timescale being evaluated.
- **No temporal resampling/interpolation** of the 10-min LiDAR series to
  hourly: the nearest 10-min sample stands in for the full hour, rather
  than an hourly mean or a temporally interpolated value.
- **250 m model level at P0 represented by the mean of the 240/260 m LiDAR
  channels**: assumes near-linear wind-speed variation with height across
  that ~20 m gap, adequate at the ~250 m target but not validated
  separately.
- **Single nearest ocean cell** represents each site (no spatial averaging
  or interpolation across neighboring cells); a land-cell match at the
  nearest-cell step triggers a warning but is not automatically corrected.
- **P0 data-availability threshold** of 50% is a fixed default, not tuned
  per level or per analysis window.

## Results and Interpretation

*(to be filled in as poster figures are finalized and interpreted)*

## Caveats and Limitations

- The pooled (all-levels) scatter in fig03/fig04 mixes different physical
  regimes (near-surface vs upper hub-height levels) into one fit; per-level
  N is reported in the caption but per-level R/RMSE/bias are not broken out
  separately in these figures.
- LPI has no data-availability/quality flag equivalent to P0's `avail`
  applied in `fig04`.
- Terrain shown in fig02 is the coarse ~4.6 km MPAS terrain, not a
  high-resolution DEM — small-scale coastal terrain features finer than the
  mesh spacing are not resolved.

## Next Steps

- Consider a sensitivity check on the ±5 minute pairing tolerance.
- Consider reporting per-level fit statistics alongside the pooled ones.

## References

- Takacs, L. L. (1985). A Two-Step Scheme for the Advection Equation with
  Minimized Dissipation and Dispersion Errors. *Monthly Weather Review*,
  113(6), 1050–1065.
