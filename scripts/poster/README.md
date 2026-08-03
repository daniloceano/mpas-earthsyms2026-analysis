# Poster figure scripts

Deterministic scripts that produce the **final** EarthSyms 2026 poster figures.
A poster script is only written once the corresponding analysis has actually
been selected from exploration.

| Script | Figure |
|---|---|
| `fig01_domain_topo_sst.py` | Domain topography + initial SST, native mesh (all observation sites marked) |
| `fig02_scatter_wspeed_p0_all_levels.py` | MPAS vs P0 LiDAR wind speed, all levels pooled |
| `fig03_map_p0_nearest_cell.py` | Native-mesh zoom on P0 + comparison cell (also used as banner) |
| `fig04_scatter_wspeed_lpi_all_levels.py` | MPAS vs LPI LiDAR wind speed, all levels pooled |
| `fig05_map_lpi_nearest_cell.py` | Native-mesh zoom on LPI + comparison cell |
| `fig06_xsection_seabreeze.py` | Meridional cross-section of temperature + wind at sea-breeze (Tmax) moment; `--site P0` or `--site LPI` |
| `fig07_domain_with_zoom_insets.py` | Domain terrain/SST (left) + P0 and LPI zoom panels stacked on right; coloured bounding boxes link each zoom to its location on the domain map |

## Promotion process (exploration → poster)

1. Explore freely in `scripts/exploratory/` or `notebooks/`, writing outputs to
   `figures/exploratory/`.
2. When an analysis is chosen for the poster, reimplement it here as a single
   self-contained script that:
   - reads all settings from `config/` or command-line arguments (no hidden
     notebook state);
   - is deterministic — same inputs produce the same output;
   - has a short module docstring naming the scientific quantity and output file;
   - writes to `figures/poster/` with a stable, descriptive filename;
   - records the simulation period and key processing choices in the figure or
     adjacent metadata.
3. Follow `docs/analysis_conventions.md` (time, radians→degrees, vertical
   staggering, precipitation differencing, mask encodings).

Generated poster figures are gitignored by default; add an approved final figure
deliberately with `git add -f`.
