# Notebooks

Scratch space for exploratory analysis. Notebooks may be messy and interactive.

Rules of thumb:
- Import the shared helpers (`mpas_analysis.config`, `.io`, `.vertical`) rather
  than re-deriving paths or vertical grids inline.
- Write exploratory outputs to `figures/exploratory/`.
- Do not treat a notebook as a reproducible deliverable. Once an analysis is
  chosen for the poster, promote it to a deterministic script under
  `scripts/poster/` (see `scripts/poster/README.md`).
- Checkpoints (`.ipynb_checkpoints/`) are gitignored. Keep committed notebooks
  small; clear bulky outputs before committing.
