# stormm-modal

Does [Psivant/stormm](https://github.com/Psivant/stormm) actually run explicit-solvent
molecular dynamics on a GPU — one simulation, and 10+ small ones at once?

**Yes to both, with caveats worth knowing before you build on it.**

Measured on one NVIDIA L40S via [Modal](https://modal.com), STORMM commit `84e97db`,
CUDA 12.4.1, built for `sm_89`.

## Results

### Phase A — build and regression suite

`sm_89` native SASS confirmed in `libstormm` (not a PTX JIT fallback). **8/8 of
STORMM's own relevant test suites pass on the L40S** in 141 s, including `test_pme`
(50.5 s), `test_neighbor_list`, `test_hpc_dynamics` and `test_generalized_born`.
Full image build from a clean clone: **409 s**.

### Phase B — one explicit-solvent simulation

JAC / DHFR, 23,558 atoms, 61.6 Å cube, full PME, 9 Å cutoff, 10,000 steps at 1 fs:

| Metric | Value |
|---|---|
| Throughput | **296.7 ns/day** at 1 fs |
| NVE total-energy drift | **−0.021 kcal/mol/ns/atom** over 9.5 ps |
| Single vs double precision, step 0 | agree to **~9 significant figures** |

That last row is the strongest correctness evidence here. `PrecisionModel::SINGLE`
and `DOUBLE` select entirely different `CellGrid` template instantiations and
different kernels (`hpc_ffpme` vs `hpc_ddpme`), yet initial electrostatic energies
match at 347865.733 vs 347865.734 kcal/mol.

### Phase C — many small simulations on one GPU

10,000 steps at 1 fs each, all replicas concurrent in a single process on a single
GPU. Times are STORMM's own `Run, Dynamics` timer, excluding setup and I/O.

| System | Atoms | Replicas | Dynamics (s) | ns/day each | **Aggregate ns/day** | Scaling vs n=1 |
|---|---|---|---|---|---|---|
| tip3p | 768 | 1 | 1.990 | 434.1 | 434 | — |
| tip3p | 768 | **16** | 2.180 | 396.3 | **6,342** | **14.6×** |
| ubiquitin | 3,105 | 1 | 2.600 | 332.3 | 332 | — |
| ubiquitin | 3,105 | **12** | 3.857 | 224.0 | **2,688** | **8.1×** |
| tip4p | 1,024 | 1 | 2.419 | 357.1 | 357 | — |
| tip4p | 1,024 | **12** | 2.961 | 291.8 | **3,501** | **9.8×** |
| jac (DHFR) | 23,558 | 1 | 2.912 | 296.7 | 297 | — |

Running 16 small solvated systems costs **9.6% more wall time than running one**.
That is the payoff of STORMM's `PhaseSpaceSynthesis` design, and it is the reason
to care about this engine for batch data generation.

Replica independence was verified rather than assumed: all 12/16 replicas end at
distinct total energies (minimum pairwise gap 0.047 kcal/mol for tip3p), so these
are genuinely separate trajectories, not duplicated work.

**TIP4P works**, including its off-atom virtual site, under PME. That matters
because a99SB-disp is parameterized against TIP4P-D.

## Caveats

One is a bug, and it fails silently. See [REQUIREMENTS.md](REQUIREMENTS.md) for the
full evidence and the ToposBio-specific gap analysis.

- **The Langevin thermostat (`ntt=3`) is an energy sink.** It applies friction with
  no compensating stochastic force and quenches systems toward 0 K — JAC fell from
  ~300 K to ~37 K in 9.5 ps. The run still exits 0 and writes a complete trajectory.
  **Use `ntt=2` (Andersen)**, which is verified working here.
- No energy minimization under periodic boundaries.
- No NPT — there is no barostat in the integrator.
- No REMD under PBC, and no REST2 (no lambda/solute-scaling infrastructure exists).

Together the middle two mean STORMM is a **production-sampling engine**, not a
whole-pipeline replacement: prepare and equilibrate elsewhere, then hand STORMM
equilibrated coordinates for the sampling where the per-GPU throughput pays.

## Running it

```
uv run modal run stormm_modal.py::smoke_test     # build + regression suites
uv run modal run stormm_modal.py::phase_b        # 1 explicit-solvent sim (JAC)
uv run modal run stormm_modal.py::phase_c        # 12-16 concurrent replicas
uv run modal run stormm_modal.py::phase_gb       # implicit vs explicit cost
uv run modal run stormm_modal.py::thermostat_probe   # the Langevin diagnostic
uv run modal volume get stormm-runs / ./results  # pull artifacts
```

All configuration is in [runs.toml](runs.toml). STORMM is cloned from its public
GitHub URL at image-build time and pinned by commit; nothing here forks it.

`analyze.py` parses STORMM's Matlab-syntax diagnostics, including the column-blocked
matrix format used for multi-system runs.
