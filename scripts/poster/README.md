# Poster figure scripts

Deterministic scripts that produce the **final** EarthSyms 2026 poster figures.
Empty for now by design — no poster script is written until the corresponding
analysis has actually been selected from exploration.

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
