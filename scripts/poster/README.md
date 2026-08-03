# Poster figure scripts

Deterministic scripts that produce the **final** EarthSyms 2026 poster figures.
A poster script is only written once the corresponding analysis has actually
been selected from exploration.

Numbered in poster reading order (row 1 left→right, then row 2 left→right):

| Script | Figure |
|---|---|
| `fig01_domain_with_zoom_insets.py` | Domain terrain/SST, full-figure map; P0 and LPI zoom panels drawn as insets *inside* the map (top-left/top-right), each linked to its zoom extent by a coloured box + connector line |
| `fig02_xsection_seabreeze.py` | Meridional cross-section of temperature + wind at sea-breeze (Tmax) moment; `--site P0` or `--site LPI` |
| `fig03_scatter_wspeed_p0_all_levels.py` | MPAS vs P0 LiDAR wind speed, all levels pooled |
| `fig04_scatter_wspeed_lpi_all_levels.py` | MPAS vs LPI LiDAR wind speed, all levels pooled |

Poster layout: row 1 is fig01 + fig02 side by side (fig01 ≈ 44%, fig02 ≈ 51%
of the row width to match their native aspect ratios at equal height); row 2
is fig03 + fig04 side by side (~45%/45%, both square).

`fig01_domain_with_zoom_insets.py` replaces three older scripts (separate
domain overview, P0 zoom map, and LPI zoom map) that have been deleted; it
combines the domain overview and both site zooms into a single panel.

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
