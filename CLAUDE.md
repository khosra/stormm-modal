# stormm-modal

Modal GPU harness that validates explicit-solvent (PME) molecular dynamics in
[Psivant/stormm](https://github.com/Psivant/stormm).

## What this repo is for

Answering one question with evidence: can STORMM actually run explicit-solvent MD
on a GPU — one simulation, and then 10+ small ones concurrently on a single card?

This repo contains only the harness. STORMM itself is cloned from its public GitHub
URL at image-build time and pinned by commit in `runs.toml`. Nothing here forks or
patches upstream.

## Layout

- `runs.toml` — all configuration: pinned commit, GPU arch, systems, MD settings, phases.
  Prefer editing this over hardcoding values in Python.
- `stormm_modal.py` — Modal app: image definition, `smoke`/`run_dynamics` remote
  functions, and `smoke_test`/`phase_b`/`phase_c` local entrypoints.
- `results/` — gitignored; populated by `modal volume get`.

## Conventions

- **Use `uv` and `toml` throughout.** Dependencies in `pyproject.toml`, run via
  `uv run modal ...`. Configuration in TOML, not Python literals or JSON.
- Modal **>= 1.0** API only (`modal.App`, `gpu="L40S"` as a string,
  `image.add_local_*` not `modal.Mount`, `@modal.fastapi_endpoint`). The client
  here is 1.5.5; introspect signatures with `inspect.signature` rather than
  trusting recalled API shapes.
- Global scope runs **remotely as well as locally**. The `@app.function` decorators
  need config at import time (GPU type, volume name), so `runs.toml` is read at module
  level *and* shipped into the image via `.add_local_file("runs.toml", "/root/runs.toml")`
  — remotely `__file__` is `/root/stormm_modal.py`, so `HERE` resolves to `/root`.
  Omitting that file is a `FileNotFoundError` at container import, not at call time.
- Beyond that, per-run values are resolved in `local_entrypoint`s and passed to remote
  functions as plain dicts, so remote functions never re-read config themselves.

## Running

```
uv run modal run stormm_modal.py::smoke_test          # Phase A: build + verify
uv run modal run stormm_modal.py::phase_b             # 1 sim on 1 GPU (JAC/DHFR)
uv run modal run stormm_modal.py::phase_c             # 10+ sims on 1 GPU
uv run modal volume get stormm-runs / ./results       # pull artifacts down
```

The first invocation triggers the STORMM build (47 CUDA translation units, single
GPU arch). Subsequent runs reuse the cached image layer.

## STORMM facts worth not re-deriving

Verified against commit `84e97db`:

- Explicit solvent runs through `apps/Dyna/src/simulator.cpp:642-677`, which builds
  `CellGrid` + `PMIGrid` + `ConvolutionManager` for `ORTHORHOMBIC`/`TRICLINIC` cells
  in both `DOUBLE` and `SINGLE` precision.
- Solvated test systems ship in the repo: `jac` (DHFR, 23,558 atoms, 61.6 A cube),
  `ubiquitin` (3,105), `tip3p` (768), `tip4p` (1,024, has virtual sites).
- `apps/Dyna/test/{JacTest,WaterTest,AminoAcidTest}.sh` are the canonical input-deck
  examples. The `-n` flag on a `-sys` block is how you request N concurrent replicas;
  STORMM's `PhaseSpaceSynthesis` runs them all in one process on one GPU.
- In `&pppm`, supplying `cut` + `dsum_tol` while omitting `ew_coeff` makes STORMM
  derive the Ewald coefficient itself and avoids the conflict branch at
  `src/Namelists/nml_pppm.cpp:84`. Defaults: `cut` 8.0, `dsum_tol` 1.0e-5,
  `mesh_ticks` 4, `order` 5.
- `src/Potential/cellgrid.h:109-110`: `minimum_cell_width` 2.5 A, `maximum_cell_width`
  12.0 A, enforced by `CellGrid::checkViability`. Small boxes at a 9 A cutoff are
  marginal — tip3p at 23.4 A is the stress case.

See `REQUIREMENTS.md` for the ToposBio-specific gap analysis (a99SB-disp, GAFF2,
TIP4P virtual sites, REST2, equilibration) — that is the document to hand to Andre.

### Known limitations (all confirmed in source)

- **No energy minimization under PBC** (`simulator.cpp:297`, `:323`, `:356`). Runs must
  start from equilibrated coordinates.
- **No NPT.** `BarostatKind` exists in `src/Trajectory/barostat.h` and `nml_dynamics.h`
  exposes MC-barostat keywords, but nothing in `src/MolecularMechanics/` consumes a
  barostat. NVE/NVT only.
- **No REMD under PBC** (`simulator.cpp:606`).
- **Periodic dynamics is GPU-only**; the CPU `dynamics.stormm` binary handles only
  `UnitCellType::NONE`.
- README warns H100 (arch 90) is "not well tested" — hence targeting L40S (arch 89).
