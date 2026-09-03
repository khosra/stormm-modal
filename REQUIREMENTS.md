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
