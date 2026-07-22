"""Centralized configuration and path handling.

One place resolves everything machine-specific: it merges the version-controlled
scientific metadata (``config/simulations.yaml``) with the local, gitignored
paths (``config/paths.local.yaml``), applying optional command-line overrides.
Nothing else in the codebase should hardcode filesystem locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

# repo_root/src/mpas_analysis/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"
SIMULATIONS_YAML = CONFIG_DIR / "simulations.yaml"
PATHS_LOCAL_YAML = CONFIG_DIR / "paths.local.yaml"
PATHS_EXAMPLE_YAML = CONFIG_DIR / "paths.example.yaml"


@dataclass(frozen=True)
class Simulation:
    """One MPAS run: where its retained history lives, plus period metadata."""

    key: str
    label: str
    history_dir: Path
    analysis_start: date
    analysis_end: date
    spinup_start: date
    integration_start: date
    integration_end: date
    era5_init: date
    restart: dict | None
    expected_history_records: int

    @property
    def restarted(self) -> bool:
        return self.restart is not None


@dataclass(frozen=True)
class Config:
    data_root: Path
    simulations: dict[str, Simulation]
    mesh: dict = field(default_factory=dict)
    vertical: dict = field(default_factory=dict)
    output: dict = field(default_factory=dict)

    def simulation(self, key: str) -> Simulation:
        try:
            return self.simulations[key]
        except KeyError:
            known = ", ".join(self.simulations)
            raise KeyError(f"unknown simulation {key!r}; known: {known}") from None

    @property
    def required_variables(self) -> list[str]:
        return list(self.output.get("required_variables", []))

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty == everything OK)."""
        problems: list[str] = []
        if not self.data_root.is_dir():
            problems.append(f"data_root does not exist: {self.data_root}")
        for sim in self.simulations.values():
            if not sim.history_dir.is_dir():
                problems.append(
                    f"[{sim.key}] history_dir missing: {sim.history_dir}"
                )
            elif not any(sim.history_dir.glob("history.*.nc")):
                problems.append(
                    f"[{sim.key}] no history.*.nc files in {sim.history_dir}"
                )
        return problems


def _load_yaml(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    *,
    paths_file: Path | None = None,
    sims_file: Path | None = None,
    data_root: str | Path | None = None,
    run_paths: dict[str, str | Path] | None = None,
) -> Config:
    """Build a :class:`Config`.

    Precedence for ``data_root`` and per-run paths: explicit argument >
    ``paths.local.yaml`` (or ``paths_file``). ``data_root`` bypasses the need for
    a local paths file entirely (useful with a ``--data-root`` CLI override).
    """
    sims_file = sims_file or SIMULATIONS_YAML
    sims_doc = _load_yaml(sims_file)

    paths_doc: dict = {}
    if data_root is None:
        paths_file = paths_file or PATHS_LOCAL_YAML
        if not paths_file.exists():
            raise FileNotFoundError(
                f"missing {paths_file}. Copy the template and edit it:\n"
                f"  cp {PATHS_EXAMPLE_YAML.relative_to(REPO_ROOT)} "
                f"{paths_file.relative_to(REPO_ROOT)}\n"
                "or pass --data-root."
            )
        paths_doc = _load_yaml(paths_file)
        data_root = paths_doc.get("data_root")
        if not data_root:
            raise ValueError(f"'data_root' not set in {paths_file}")

    data_root = Path(data_root).expanduser()
    overrides = dict(paths_doc.get("runs") or {})
    overrides.update({k: v for k, v in (run_paths or {}).items() if v})

    simulations: dict[str, Simulation] = {}
    for key, meta in (sims_doc.get("simulations") or {}).items():
        override = overrides.get(key)
        history_dir = (
            Path(override).expanduser()
            if override
            else data_root / meta["history_subdir"]
        )
        simulations[key] = Simulation(
            key=key,
            label=meta["label"],
            history_dir=history_dir,
            analysis_start=meta["analysis_start"],
            analysis_end=meta["analysis_end"],
            spinup_start=meta["spinup_start"],
            integration_start=meta["integration_start"],
            integration_end=meta["integration_end"],
            era5_init=meta["era5_init"],
            restart=meta.get("restart"),
            expected_history_records=meta.get("expected_history_records", 0),
        )

    return Config(
        data_root=data_root,
        simulations=simulations,
        mesh=sims_doc.get("mesh", {}),
        vertical=sims_doc.get("vertical", {}),
        output=sims_doc.get("output", {}),
    )
