# ToposBio requirements vs. STORMM capability

Gap analysis for moving the Tsunami data-generation workload (IDRs + small
molecules) from implicit to explicit solvent on STORMM.

Everything below is read from STORMM source at commit `84e97db` unless a row says
"measured", in which case it came from a run in this repo. **Source reading is not
a substitute for running the real system** — treat unmeasured rows as "the machinery
is present", not "this works".

Audience: Andre, who built Tsunami. Context: currently OpenMM + a99SB-disp
(Robustelli) + GAFF2, implicit solvent GB-Neck2 today, explicit solvent wanted.

## Verdict summary

| Requirement | Status | Blocker? |
|---|---|---|
| Explicit solvent / PME | Implemented, GPU-only | No |
| GB-Neck2 implicit (igb=8) | Implemented | No |
| Many small systems per GPU | Core design (`PhaseSpaceSynthesis`, `-n` flag) | No |
| TIP4P virtual sites under PME | Machinery present, needs measurement | No, pending |
| Off-diagonal LJ / NBFIX (a99SB-disp) | Implemented | No |
| GAFF2 ligands | Standard prmtop, nothing special needed | No |
| Energy minimization under PBC | **Not implemented** | Yes — prep stays upstream |
| NPT / barostat | **Not implemented** | Yes — prep stays upstream |
| REST2 | **Not implemented**, three ways | Yes |
| T-REMD under PBC (the Tsunami ladder: 300/316/332/350 K) | **Not implemented** — isolated boundaries only | Yes — removes the campaign's sampling method |
| Langevin thermostat (`ntt=3`) | **Broken** — friction with no random force, quenches to 0 K | Use `ntt=2` (Andersen) |

## Detail

### Works

- **Explicit solvent (PME).** `apps/Dyna/src/simulator.cpp:642-677` builds
  `CellGrid` + `PMIGrid` + `ConvolutionManager` for `ORTHORHOMBIC`/`TRICLINIC`
  cells, in both single and double precision. Landed via the `PeriodicDynamics`
  branch; this is recent code.
- **GB-Neck2.** `ImplicitSolventModel::NECK_GB_II` is igb=8
  (`src/Topology/atomgraph_enumerators.h:147`) — the model behind the current
  Tsunami implicit runs. `apps/Dyna/test/AminoAcidTest.sh` drives exactly this
  pattern: three peptide systems at `-n 8`/`-n 20`/`-n 20`, concurrent on one GPU.
- **Many systems per GPU.** This is STORMM's whole design point, not a bolt-on.
  One process holds N systems in a `PhaseSpaceSynthesis` and steps them together.
- **Off-diagonal Lennard-Jones.** a99SB-disp modifies protein-water dispersion,
  which cannot be expressed by Lorentz-Berthelot mixing. STORMM reads
  `LENNARD_JONES_ACOEF`/`BCOEF` from the prmtop as *essential* sections into a
  pair-indexed matrix (`src/Topology/atomgraph_detailers.cpp:986`) and has
  `inferCombiningRule()` to detect non-standard mixing. So NBFIX-style terms
  survive the topology round trip.
- **Force fields generally.** STORMM consumes Amber prmtop. a99SB-disp and GAFF2
  are upstream concerns — build topologies with tleap/OpenMM as today and hand
  STORMM the prmtop. The engine only needs the functional forms, which it has.

### Needs measurement

- **TIP4P virtual sites under PME.** ToposBio uses TIP4P, and a99SB-disp is
  parameterized against TIP4P-D, so the off-atom M-site is a hard prerequisite.
  Machinery is present and correctly placed: placement is fused into the valence
  kernel's integration section (`src/Potential/valence_potential.cui:2553`), which
  every PME valence TU (`hpc_valence_potential_{fpme,dpme,fpmedual}.cu`) includes,
  and `src/Structure/virtual_site_transmission.cui` returns virtual-site forces to
  frame atoms. Frame types cover FLEX_2/FIXED_2/FLEX_3/FIXED_3/FAD_3 and more.
  The `tip4p` system in `runs.toml` tests this.

### Confirmed bug: the Langevin thermostat is an energy sink

**`ntt=3` (Langevin) does not thermostat. It applies friction with no compensating
stochastic force, quenching the system toward 0 K.** Measured on an L40S, STORMM
`84e97db`.

Phase B: JAC/DHFR cooled from ~300 K to ~37 K in 9.5 ps, kinetic energy decaying
monotonically 9937 -> 1.7 kcal/mol at about 0.91 ps^-1, which matches
`default_langevin_frequency` (0.001 fs^-1 = 1 ps^-1).

A one-knob-at-a-time probe on ubiquitin (4000 steps each) isolates it:

| Thermostat | KE(0) -> KE(end) | Behaviour |
|---|---|---|
| NVE, `ntt=0` | 1666 -> 2851 | holds, plateaus |
| Langevin, `ntt=3`, default gamma | 1675 -> 118 | collapses |
| Langevin, `gamma_ln=1.0` | 745 -> 0.1 | collapses faster |
| Langevin, `gamma_ln=0.01` | 1660 -> 2.0 | collapses |
| Andersen, `ntt=2` | 1676 -> 2777 | holds, plateaus |
| Berendsen, `ntt=1` | 1676 -> 2834 | holds, plateaus |

Two things make this conclusive rather than suggestive. The collapse rate scales
monotonically with `gamma_ln`, exactly as friction without fluctuation-dissipation
predicts. And it reproduces with `rigid_geom off`, so it is not an interaction with
the constraint solver. NVE, Andersen and Berendsen all converge to the same ~2800
plateau, so the integrator and the PME forces are fine — this is specific to Langevin.

Ruled out as causes, by reading source: the `tevo_start`/`tevo_end` defaults of 0
correctly yield a constant `temp0` target (the `>=` branch in
`src/Trajectory/thermostat.tpp:106` fires immediately, no degenerate division), and
the RNG cache defaults are sane (`tcache_depth` 1, seed 1329440765, config "single").

**Workaround: use `ntt=2` (Andersen).** It is a legitimate canonical thermostat.
Berendsen also holds energy but does not sample the canonical ensemble correctly
and should not be used for production IDR sampling.

**This is worth reporting upstream to Psivant.** Langevin is the default choice for
most production MD and the failure is silent — the run exits 0, writes a trajectory,
and looks superficially fine. Anyone who does not check the kinetic energy would
publish a frozen trajectory.

### Blockers

- **No energy minimization under periodic boundaries.** `rtErr` at
  `apps/Dyna/src/simulator.cpp:297`, `:323`, `:356`.
- **No NPT.** `BarostatKind` is declared (`src/Trajectory/barostat.h`) and
  `nml_dynamics.h` exposes MC-barostat keywords, but nothing in
  `src/MolecularMechanics/` consumes a barostat. NVE/NVT only.

  *Consequence for the workflow:* STORMM cannot do solvate → minimize →
  NPT-equilibrate. It needs an already-equilibrated box at the right density.
  Practically this makes STORMM a **production-sampling engine**, not a
  replacement for the whole pipeline — keep OpenMM/Amber for prep and
  equilibration, hand STORMM equilibrated coordinates for the sampling where
  the per-GPU throughput actually pays.

- **REST2 is not available.** Three independent reasons, any one sufficient:
  1. REMD does not run under periodic boundaries at all —
     `apps/Dyna/src/simulator.cpp:606` raises "Replica Exchange molecular dynamics
     is not yet operational for periodic boundary conditions."
  2. The `&remd` namelist advertises a "Hamiltonian" type, but
     `ExchangeNexus::getHamiltonian()` (`src/Sampling/exchange_nexus.cpp:89-101`)
     just returns kinetic + potential energy per replica. That is a total energy,
     not a lambda-scaled Hamiltonian, and the function carries an unfinished
     `// TODO` about index correspondence. It is not REST2 and not currently
     anything close to it.
  3. There is no lambda / solute-scaling infrastructure anywhere in the tree.
     `src/Potential/soft_core_potentials.h` has no lambda parameter, and nothing
     matches `rest2`/`solute_scaling`.

  REST2 needs per-replica scaling of solute-solute and solute-solvent terms, which
  would mean new parameter plumbing through the synthesis and the non-bonded
  kernels. This is a feature request to the STORMM authors, not a configuration
  problem. Worth raising with them directly if REST2 is load-bearing for Tsunami.

- **Lengthy equilibration** is a consequence of the two items above rather than a
  separate gap: equilibration has to happen in whatever engine does prep.

## When PME actually became usable

Relevant because the Tsunami campaign ran on an earlier STORMM and the question came
up of whether explicit solvent had been available all along. It had not. Three
separate things landed at three separate times, and only the last one makes an
explicit-solvent trajectory possible.

| What | When | Commit |
|---|---|---|
| PME *library components* (`CellGrid`, `PMIGrid`, `pme_util`) | 2023-10-12 | `bff1ba3` |
| `ConvolutionManager` created, "Begin to add periodic dynamics" | 2024-09-22 | `50ef66a` |
| Dyna app gains a periodic path (`CellGrid` only) | 2024-12-19 | `c1bf07d` |
| **`PMIGrid` + `ConvolutionManager` wired into the MD loop** | **2026-08-31** | **`c76bffc`** |

`c76bffc` is the second-to-last commit in the repository, inside the current release
(v0.3.0, tagged 2026-08-31 at `84e97db`). Release dates: v0.2.0 on 2025-05-23,
v0.3.0 on 2026-08-31, 465 days apart.

The trap is that v0.2.0 *looks* like it supports explicit solvent. Its Dyna app has a
`UnitCellType::ORTHORHOMBIC`/`TRICLINIC` branch that constructs a `CellGrid` and calls
a templated `dynamics()`. But that call passes only the cell grid:

    v0.2.0   dynamics(&poly_ps, &cg, &edyn, &tst, poly_ag, lem, dyncon, preccon, pmecon)
    v0.3.0   dynamics(&poly_ps, &cg, &pmig, &cvol, &edyn, &tst, poly_ag, lem, dyncon, ...)

v0.2.0's GPU dynamics driver contains **zero** references to `PMIGrid` across its
`.h`, `.cu` and `.tpp`. There is no reciprocal-space Ewald sum in the MD loop — it
had a periodic neighbour list, not particle-mesh Ewald.

**To check any given STORMM tree:**

    git log -1 --format='%h %ad' --date=short
    grep -c PMIGrid src/MolecularMechanics/hpc_dynamics.*   # 0 = no PME in the MD loop

This also explains the quality gradient observed in testing: the PME kernels had
roughly three years to mature and their unit tests pass, while the integration around
them is days old — which is where the Langevin thermostat bug lives.

### What the Tsunami container pinned

`ToposBio/stormm` (4 commits, last pushed 2025-07-17) is Andre's Modal container for
the campaign. Its Dockerfile does:

    RUN git clone https://github.com/psivant/stormm.git

No tag, no commit, no `--branch`. The STORMM version is therefore whatever `main` was
**on the day the image was built**. Built July 2025, that is v0.2.0-era main:
no PME in the MD loop. Explicit solvent was not available to that campaign.

Vintage barely matters here, because **`main` was frozen for fourteen months**: it sat
at `3eeccbd` (2025-06-18) until `84e97db` (2026-08-31). Any clone of
`psivant/stormm` main taken between those dates — July 2025 or July 2026 alike —
produced the same tree, with zero `PMIGrid` references in the dynamics driver. So no
Tsunami-era build of the public repo could do explicit solvent.

For the rebuild, pin the commit. `runs.toml` in this repo does.

## What the Tsunami campaign actually ran

Read directly from `tsunami-sims-v2b`, `/results/p000/p000000/metadata.json`:

    ladder_temps_K      [300.0, 315.818, 332.47, 350.0]    n_rungs 4
    remd_swap_steps     10000
    n_steps             10000000        dt_fs 2.0      -> 20 ns per replica
    ntwx                10000                          -> 1000 frames
    igb                 5               gbsa 0
    restrain_radius_a   66.62           restrain_radius_base_a 40.0
    transfer 0          ligand_transfer 0

Note **`igb = 5`** (OBC-GB-II), not `igb = 8` (GB-Neck2). Anything downstream that
assumes GB-Neck2 — model cards, dataset documentation, the `constants.py` comment in
`tp2-partstruct` — is describing a different solvent model than the data was generated
with. Worth correcting at the source.

Layout is `results/pNNN/pNNNNNN/` with `T{300,316,332,350}.{nc,csv}`, `metadata.json`
and `topology.pdb` per system, plus a `results/_done` marker.

### The data is sound — the Langevin concern does not apply to it

Each `T*.csv` records `inst_temp_K` per frame, so this is checkable directly rather
than by proxy:

| Rung | Setpoint | Measured mean | sd |
|---|---|---|---|
| T300 | 300 K | **298.8 K** | 13.4 |
| T350 | 350 K | **348.1 K** | 21.5 |

No drift (first decile 300.8 K, last decile 297.6 K), and all four `walker_id` values
appear in each rung's file, so replica exchange was genuinely swapping. **The Tsunami
trajectories are correctly thermostatted.** The Langevin energy-sink bug recorded above
was measured on v0.3.0 and does not describe this dataset.

## Andre already tried explicit solvent, in April 2026, and it crashed

`stormm-pme-benchmark-artifacts` (2026-04-02/03) holds ~20 timestamped runs of the
androgen receptor under PME. Run `20260403_062328` is a replica sweep, and all three
arms died inside the reciprocal-space code:

| Replicas | Failure |
|---|---|
| 1 | `runGpuReciprocalPmeStep :: Unable to release reciprocal-space energy scratch on the GPU` |
| 8 | `hpc_fft :: (executeCuFFTForward) cuFFT call failed with status CUFFT_EXEC_FAILED` |
| 32 | `hpc_fft :: (executeCuFFTBackward) cuFFT call failed with status CUFFT_EXEC_FAILED` |

This is almost certainly *why* Tsunami is an implicit-solvent dataset: explicit solvent
was attempted first and did not survive the FFT.

**Those symbols do not exist in Psivant/stormm.** `runGpuReciprocalPmeStep`,
`executeCuFFTForward` and `executeCuFFTBackward` appear in no branch — not `main`,
`DSCDev`, `ImprovedCLI` or `temp` — and the public tree's FFT sources are named
`hpc_fft_stage.cu` with several files still carrying `_wip` suffixes. So the STORMM
Andre benchmarked in April came from somewhere other than the public repository.
**Ask him where that build came from**; it changes what "upgrading STORMM" even means.

### The good news

The failure mode he hit is exactly what now works. On v0.3.0, PME with cuFFT ran
cleanly at 1, 12 and 16 replicas, conserved energy to −0.021 kcal/mol/ns/atom, and
agreed between single and double precision to ~9 significant figures. Whatever broke
the reciprocal-space step in April is not present in the current release.

## Pilot: 10 real Tsunami systems in explicit solvent

Ran end to end on one L40S. ff14SB + GAFF2 + TIP3P (first pass; a99SB-disp + TIP4P-D
still to source). Ligand parameters reused from `tsunami-ligands` — the campaign's own
`.mol2` + `.frcmod`, with the docked pose transferred by atom name — so the chemistry
matches the implicit runs. tleap solvates, OpenMM minimizes/heats/NPT-equilibrates,
STORMM does production with PME and `ntt=2`.

**All 10 succeeded.** Every system equilibrated to 297.7–301.8 K in OpenMM, then ran in
STORMM with post-thermalization energy drift of **+0.0072 to +0.0235 kcal/mol/ns/atom**
(mean +0.014), comparable to the JAC NVE reference of −0.021. Temperature held steady
within each run (sd 0.6–2.4 K).

| System | Atoms | ns/day | Drift (kcal/mol/ns/atom) |
|---|---|---|---|
| p000003 | 31,272 | 320.0 | +0.0139 |
| p000005 | 33,773 | 281.4 | +0.0072 |
| p000000 | 42,727 | 265.8 | +0.0235 |
| p000007 | 45,987 | 257.1 | +0.0168 |
| p000001 | 46,137 | 290.9 | +0.0144 |
| p000002 | 60,595 | 226.2 | +0.0177 |
| p000004 | 63,118 | 211.2 | +0.0147 |
| p000008 | 88,220 | 179.6 | +0.0130 |
| p000006 | 102,206 | 158.5 | +0.0132 |
| p000009 | 152,342 | 113.2 | +0.0096 |

Mean 66,637 atoms and 230 ns/day per system.

### The cost of a like-for-like rebuild

The solutes are 472–848 atoms. Solvated they are **31k–152k atoms**, because extended
IDR conformations (43–83 Å across) force large boxes — volume scales as extent³ while
the solute does not. This is the dominant cost, and it is far worse than the 3.28×
implicit-to-explicit ratio measured on a compact small molecule.

    97,000 systems x 4 rungs x 20 ns = 7,760,000 ns
    at 230 ns/day/GPU  ->  33,700 GPU-days  =  92 GPU-years

That is not a schedulable number, so a full like-for-like rebuild is off the table and
the question becomes which axis to cut:

| Scope | GPU-days |
|---|---|
| Full: 97k x 4 rungs x 20 ns | 33,700 |
| Single temperature, 20 ns | 8,420 |
| Single temperature, 5 ns | 2,105 |
| 10k subset, 1 temp, 5 ns | 217 |
| 1k subset, 1 temp, 20 ns | 87 |

Three things make this less bleak than it looks:

1. **The 4 rungs may be moot anyway.** STORMM has no REMD under periodic boundaries, so
   the ladder is unavailable regardless — which removes a 4× factor but also removes the
   enhanced sampling that motivated it.
2. **Boxes are oversized.** `solvateBox ... iso` builds a cube that contains the solute
   under any rotation. Aligning principal axes first, or dropping `iso`, should cut atom
   counts substantially — plausibly ~2×, which roughly doubles throughput.
3. **Batching is untested at this size.** STORMM ran 16 small replicas for +9.6% wall
   time, but these systems are 30–150k atoms and only a few fit per card. The pilot ran
   one system per GPU; measuring 2–4 concurrent is the obvious next experiment.

### Known gap in the pilot

ParmEd wrote positions only, so STORMM started production without velocities and spent
the first ~250 steps rethermalizing. Harmless for a throughput measurement, and the
reported drift excludes it, but for production the equilibrated velocities should be
carried across so trajectories continue rather than restart.

## The sampling method, not just the solvent model, is at risk

The `tsunami-sims-v2b` volume stores trajectories at **300 / 316 / 332 / 350 K** per
system. That is a replica-exchange temperature ladder, not four unrelated runs, and it
means the campaign's sampling depended on T-REMD.

STORMM's REMD is implemented **only for isolated boundary conditions**. The branch in
`apps/Dyna/src/simulator.cpp` handles `UnitCellType::NONE` and then, for
`ORTHORHOMBIC`/`TRICLINIC`, raises:

    "Replica Exchange molecular dynamics is not yet operational for periodic
     boundary conditions."

So REMD worked for the original implicit-solvent campaign precisely because those
systems were isolated. **Moving to explicit solvent removes it.** This is a larger
obstacle than the solvent model itself: the rebuild does not merely swap GB for PME,
it loses the enhanced sampling the dataset was constructed around, and IDR conformational
sampling is exactly the case where that hurts most.

Options, none free:

1. Wait for / request periodic REMD upstream from Psivant. Combined with the REST2 gap,
   this is the single highest-value ask.
2. Run explicit-solvent plain MD at multiple fixed temperatures and accept the loss of
   exchange. Cheapest, and STORMM's per-GPU batch throughput makes brute force more
   viable than it would be elsewhere, but it is not equivalent sampling.
3. Keep enhanced sampling in OpenMM and use STORMM only where plain MD throughput wins.

Worth resolving before committing to a rebuild plan, because it affects whether STORMM
is the right engine for this workload at all — as opposed to being the right engine for
the production-sampling stage of it.

## Next investigation: topos-md

The authoritative statement of what a production run actually needs is `topos-md`,
the primary ToposBio MD repo, where Malhar configures the OpenMM runs. Reading the
OpenMM setup there and diffing it against what STORMM's `&dynamics`/`&pppm`/
`&solvent` namelists can express is the way to find the remaining gaps, rather
than guessing from STORMM's side as this document has done so far.

Known to look for, from the discussion that produced this file:

- **Warmup / heating protocol.** Almost certainly a gap. STORMM has `tevo_start`
  and `tevo_end` for a temperature ramp within a run, but no PBC minimization and
  no NPT means the usual minimize → heat → density-equilibrate sequence cannot
  happen in STORMM at all.
- Thermostat and integrator choice, and whether STORMM's set matches.
- Constraint scheme and timestep (`rigid_geom`, hydrogen mass repartitioning).
- Restraint usage during equilibration.
- Reporting cadence and trajectory format expected by downstream analysis.

Note: `topos-md` is private and could not be read from this session — enumerating
org repositories was blocked by the sandbox. Needs either access or a paste of the
relevant OpenMM setup code.

## Deferred / not yet investigated

- Trajectory output formats and whether they round-trip into the existing Tsunami
  analysis (STORMM writes Amber ASCII/`.crd`; NetCDF is available but off by
  default, `-DSTORMM_INCLUDE_NETCDF`).
- Hydrogen mass repartitioning / 4 fs timesteps.
- Restraints under PBC (`&restraint` exists; untested here for periodic systems).
- Whether STORMM's thermostat set matches what the current production runs use.
