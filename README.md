# mpas-earthsyms2026-analysis

Analysis of two regional MPAS-Atmosphere hindcasts over the Brazilian equatorial
margin, for an **EarthSyms 2026** conference poster. The region is of offshore
wind-resource interest; the runs cover periods chosen to compare against
available wind measurements.

This is an initial, compact poster project: exploratory analysis of two existing
simulations, with a clean path to promote selected analyses into final poster
figures. Raw simulation data are large and **external** — they are read in place,
never copied into this repository.

## The two simulations

Shared mesh (`meqbr_05km`, ~4.6 km, `nCells = 76813`) and model configuration;
they differ only in period. Both keep hourly `history.*.nc` over their retained
analysis window.

| Key | Analysis period | Notes |
|---|---|---|
| `sim_2021` | 2021-11-01 → 2021-12-01 | fresh ERA5 init 2021-10-21; 10-day spin-up discarded |
| `sim_2022` | 2022-10-01 → 2022-11-01 | ERA5 init 2022-09-21; restarted at 2022-10-01 |

Full technical reference: [`docs/simulation_configuration.md`](docs/simulation_configuration.md).
Analysis conventions: [`docs/analysis_conventions.md`](docs/analysis_conventions.md).

## Repository layout

```
config/     paths (example + local) and simulation metadata (simulations.yaml)
docs/       simulation configuration reference and analysis conventions
src/mpas_analysis/   config/path handling, lazy history I/O, vertical-grid helpers
scripts/exploratory/ inventory_outputs.py, quicklook_wind.py
scripts/poster/      final deterministic poster-figure scripts (added later)
figures/exploratory/ throwaway diagnostic figures (gitignored)
figures/poster/      final poster figures (gitignored unless approved)
notebooks/           scratch exploration
```

## Environment setup

```bash
conda env create -f environment.yml
conda activate mpas-earthsyms2026
```

## External-path configuration

Raw MPAS output is external. Configure where it lives once:

```bash
cp config/paths.example.yaml config/paths.local.yaml
# edit config/paths.local.yaml: set data_root to the meqbr_05km run directory
```

`config/paths.local.yaml` is gitignored. `config/simulations.yaml` (committed)
holds the scientific metadata. Any script also accepts `--data-root` to override
without editing the file. Optionally, put local symlinks under `.local-data/`
(also gitignored); scripts never require them.

## Running the scripts

```bash
# Inventory both runs + validate paths (files, timestamps, coverage, variables):
python scripts/exploratory/inventory_outputs.py

# Minimal wind quick-look (10 m vs ~50 m), first 24 h of a run:
python scripts/exploratory/quicklook_wind.py --sim sim_2021 --hours 24
```

`inventory_outputs.py` doubles as the path-validation command: it reports any
missing run directory or required variable and exits non-zero on problems.

## Exploratory vs poster figures

- **Exploratory** (`figures/exploratory/`): quick, descriptive, not
  publication-ready; produced by `scripts/exploratory/` and notebooks.
- **Poster** (`figures/poster/`): final figures from deterministic scripts in
  `scripts/poster/`. See [`scripts/poster/README.md`](scripts/poster/README.md)
  for the promotion process. Poster scripts are only written once an analysis is
  actually selected.

## Reproducibility

Analyses read configuration from `config/` and simulation provenance from the
archived namelists/streams referenced in `docs/simulation_configuration.md`.
Poster scripts are deterministic (settings from config/CLI, no notebook state)
and record period and processing choices with each figure.

## Data availability

Raw simulation data (NetCDF history, mesh, static, LBC, restart,
initialization, animations) are **not** version controlled and are not stored
here. They remain in the external simulation directories and are accessed in
place via `config/paths.local.yaml`.
