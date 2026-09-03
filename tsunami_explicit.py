"""Re-run Tsunami systems in explicit solvent.

The original campaign ran implicit solvent (igb=5) with 4-rung T-REMD. This pilot
takes real systems from `tsunami-sims-v2b` and runs them with explicit water and PME,
to measure what that costs and prove the pipeline end to end.

First-pass force fields: ff14SB (protein) + GAFF2 (ligand) + TIP3P (water). The target
is a99SB-disp + TIP4P-D, which is a real dependency to source rather than a flag; the
engine-side prerequisites for it (off-diagonal LJ, virtual sites under PME) are already
verified in REQUIREMENTS.md.

Division of labour is forced by STORMM's limits: it cannot minimize or run NPT under
periodic boundaries, so solvation and equilibration happen in tleap + OpenMM, and
STORMM does production sampling only.

    uv run modal run tsunami_explicit.py::pilot
"""

import json
import pathlib

import modal

from stormm_modal import (
    STORMM_SRC, STORMM_BUILD, DYNA, stormm_image, results_vol, load_config,
)

app = modal.App("tsunami-explicit")
prep_vol = modal.Volume.from_name("tsunami-explicit-prep", create_if_missing=True)

# This module imports stormm_modal at module scope, and module scope runs in the
# container too, so the sibling module has to travel with the image. Modal >= 1.0
# does not bundle local modules implicitly.
stormm_image = stormm_image.add_local_python_source("stormm_modal")

# AmberTools gives tleap for solvation; OpenMM does the minimize/NPT that STORMM cannot.
prep_image = (
    modal.Image.micromamba(python_version="3.11")
    .micromamba_install(
        "ambertools=23.6", "openmm=8.1.1", "parmed=4.2.2", "numpy<2",
        channels=["conda-forge"],
    )
    .add_local_python_source("stormm_modal")
)

PAD_A = 10.0          # solvent padding beyond the solute, per side
EQUIL_NVT_PS = 20.0
EQUIL_NPT_PS = 80.0


def retype_mol2_coords(mol2_text: str, pdb_text: str) -> str:
    """Put the docked pose's coordinates onto the parameterized ligand.

    The .mol2 in tsunami-ligands carries GAFF types and charges but the *seed*
    conformer's geometry. The pose we want lives in the system's topology.pdb. Atom
    names are shared between the two, so they can be matched by name.
    """
    pose = {}
    for l in pdb_text.splitlines():
        if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() == "LIG":
            pose[l[12:16].strip()] = (float(l[30:38]), float(l[38:46]), float(l[46:54]))

    lines = mol2_text.splitlines()
    i = lines.index("@<TRIPOS>ATOM")
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("@<TRIPOS>"))
    out, missing = list(lines[: i + 1]), []
    for row in lines[i + 1:j]:
        p = row.split()
        name = p[1]
        if name in pose:
            x, y, z = pose[name]
            p[2], p[3], p[4] = f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"
        else:
            missing.append(name)
        out.append(" ".join(p))
    if missing:
        raise ValueError(f"ligand atoms absent from topology.pdb: {missing[:5]}")
    out.extend(lines[j:])
    return "\n".join(out) + "\n"


@app.function(image=prep_image, gpu="L40S", timeout=5400,
              volumes={"/prep": prep_vol}, cpu=4.0, memory=16384)
def prep_and_equilibrate(pair_id: str, pdb_text: str, mol2_text: str,
                         frcmod_text: str, net_charge: int) -> dict:
    """Solvate with tleap, then minimize + heat + NPT-equilibrate with OpenMM.

    Returns a summary; writes prmtop and equilibrated coordinates to the prep volume.
    STORMM can do neither of these steps under periodic boundaries, which is why they
    live here.
    """
    import subprocess
    import time

    work = pathlib.Path("/tmp") / pair_id
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    # Protein-only PDB; the ligand comes back in via its own mol2 so it keeps the
    # GAFF types and AM1-BCC charges that were computed for the original campaign.
    prot = [l for l in pdb_text.splitlines()
            if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() != "LIG"]
    (work / "prot.pdb").write_text("\n".join(prot) + "\nEND\n")
    (work / "lig.mol2").write_text(retype_mol2_coords(mol2_text, pdb_text))
    (work / "lig.frcmod").write_text(frcmod_text)

    ion = "Na+" if net_charge < 0 else "Cl-"
    n_ion = abs(int(net_charge))
    addions = f"addIons2 sys {ion} {n_ion}\n" if n_ion else ""

    (work / "tleap.in").write_text(f"""source leaprc.protein.ff14SB
source leaprc.gaff2
source leaprc.water.tip3p
loadamberparams {work}/lig.frcmod
LIG = loadmol2 {work}/lig.mol2
prot = loadpdb {work}/prot.pdb
sys = combine {{ prot LIG }}
{addions}solvateBox sys TIP3PBOX {PAD_A} iso
addIonsRand sys Na+ 0 Cl- 0
saveAmberParm sys {work}/system.prmtop {work}/system.inpcrd
quit
""")
    tl = subprocess.run(["tleap", "-f", str(work / "tleap.in")],
                        cwd=work, capture_output=True, text=True, timeout=1800)
    if not (work / "system.prmtop").exists():
        return {"pair_id": pair_id, "ok": False, "stage": "tleap",
                "log": (tl.stdout + tl.stderr)[-4000:]}
    t_leap = time.monotonic() - t0

    # ---- OpenMM: minimize, heat, NPT ----
    import openmm as mm
    import openmm.app as app_mm
    import openmm.unit as unit

    prm = app_mm.AmberPrmtopFile(str(work / "system.prmtop"))
    crd = app_mm.AmberInpcrdFile(str(work / "system.inpcrd"))
    n_atoms = prm.topology.getNumAtoms()

    system = prm.createSystem(nonbondedMethod=app_mm.PME,
                              nonbondedCutoff=9.0 * unit.angstrom,
                              constraints=app_mm.HBonds,
                              rigidWater=True)
    integrator = mm.LangevinMiddleIntegrator(300 * unit.kelvin, 1 / unit.picosecond,
                                             2.0 * unit.femtosecond)
    sim = app_mm.Simulation(prm.topology, system, integrator,
                            mm.Platform.getPlatformByName("CUDA"))
    sim.context.setPositions(crd.positions)
    if crd.boxVectors is not None:
        sim.context.setPeriodicBoxVectors(*crd.boxVectors)

    t1 = time.monotonic()
    sim.minimizeEnergy(maxIterations=5000)
    sim.context.setVelocitiesToTemperature(100 * unit.kelvin)
    sim.step(int(EQUIL_NVT_PS * 500))                     # 2 fs steps

    system.addForce(mm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 25))
    sim.context.reinitialize(preserveState=True)
    sim.step(int(EQUIL_NPT_PS * 500))
    t_equil = time.monotonic() - t1

    st = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=True)
    box = st.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.angstrom)
    ke = st.getKineticEnergy().value_in_unit(unit.kilocalorie_per_mole)
    dof = 3 * n_atoms - system.getNumConstraints() - 3
    temp = 2 * ke / (dof * 0.0019872041)

    import parmed
    struct = parmed.load_file(str(work / "system.prmtop"),
                             xyz=st.getPositions(asNumpy=True).value_in_unit(unit.angstrom))
    struct.box = [box[0][0], box[1][1], box[2][2], 90.0, 90.0, 90.0]
    dest = pathlib.Path("/prep") / pair_id
    dest.mkdir(parents=True, exist_ok=True)
    struct.save(str(dest / "system.prmtop"), overwrite=True)
    struct.save(str(dest / "equil.rst7"), overwrite=True, format="rst7")
    prep_vol.commit()

    return {"pair_id": pair_id, "ok": True, "n_atoms": n_atoms,
            "box_a": [round(float(box[i][i]), 2) for i in range(3)],
            "equil_temp_K": round(float(temp), 1),
            "tleap_s": round(t_leap, 1), "equil_s": round(t_equil, 1)}


@app.function(image=stormm_image, gpu="L40S", timeout=3600,
              volumes={"/prep": prep_vol, "/results": results_vol})
def stormm_production(pair_id: str, nstlim: int = 5000, replicas: int = 1) -> dict:
    """Production explicit-solvent MD in STORMM, from the equilibrated box."""
    import json
    import shutil
    import subprocess
    import time

    src = pathlib.Path("/prep") / pair_id
    prm, rst = src / "system.prmtop", src / "equil.rst7"
    if not prm.exists() or not rst.exists():
        return {"run_name": pair_id, "returncode": -99, "wall_seconds": 0.0,
                "peak_gpu_mib": 0, "stdout_tail": "prep artifacts missing",
                "stderr_tail": "", "timed_out": False}

    work = pathlib.Path("/tmp/prod") / pair_id
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # ntt=2 (Andersen), NOT 3: STORMM's Langevin thermostat applies friction with no
    # compensating stochastic force. See REQUIREMENTS.md.
    deck = f"""&files
  -sys {{ -p {prm}
         -c {rst}
         -label TSU -n {replicas} }}
  -o diagnostics.m
  -x traj.crd
  x_kind AMBER_CRD
&end

&dynamics
  nstlim = {nstlim},  ntpr = {max(nstlim // 20, 1)},  ntwx = {nstlim},  dt = 2.0,
  cut = 9.0,
  ntt = 2,
  rigid_geom on,
  temperature = {{ tempi 300.0, temp0 300.0, -label TSU }},
&end

&pppm
  theme electrostatic,
  order 5,
  cut 9.0,
  dsum_tol 1.0e-5,
  mesh_ticks 4,
&end

&precision
  nonbonded single,
  valence single,
&end

&report
  syntax = Matlab,
  energy total,
  energy electrostatic,
  energy vdw,
  energy kinetic,
  ascii_salvage STARS,
&end
"""
    (work / "md.in").write_text(deck)
    t0 = time.monotonic()
    proc = subprocess.run([DYNA, "-O", "-i", "md.in", "-except", "warn"],
                          cwd=work, capture_output=True, text=True, timeout=3000)
    wall = time.monotonic() - t0
    (work / "stdout.txt").write_text(proc.stdout)
    (work / "stderr.txt").write_text(proc.stderr)

    summary = {"run_name": f"explicit-{pair_id}", "returncode": proc.returncode,
               "timed_out": False, "wall_seconds": round(wall, 2), "peak_gpu_mib": 0,
               "nstlim": nstlim, "replicas": replicas,
               "stdout_tail": proc.stdout[-6000:], "stderr_tail": proc.stderr[-6000:],
               "artifacts": sorted(p.name for p in work.iterdir())}
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    dest = pathlib.Path("/results") / f"explicit-{pair_id}"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(work, dest)
    results_vol.commit()
    return summary


@app.local_entrypoint()
def pilot(n: int = 10, nstlim: int = 5000,
          data_dir: str = "/private/tmp/claude-501/-Users-amir-Documents-Github-stormm/3c6943e3-f0fa-43bf-a831-f09531240bfa/scratchpad"):
    """Solvate, equilibrate and run the first n Tsunami systems in explicit solvent."""
    tsu = pathlib.Path(data_dir) / "tsu10"
    lig = pathlib.Path(data_dir) / "ligmatch"

    jobs = []
    for p in sorted(tsu.glob("p*.pdb"))[:n]:
        pid = p.stem
        m, f, j = lig / f"{pid}.mol2", lig / f"{pid}.frcmod", lig / f"{pid}.meta.json"
        if not m.exists():
            print(f"  skip {pid}: no matched ligand")
            continue
        meta = json.loads(j.read_text()) if j.exists() else {}
        jobs.append((pid, p.read_text(), m.read_text(),
                     f.read_text() if f.exists() else "",
                     int(meta.get("net_charge", 0))))

    print(f"preparing {len(jobs)} systems (tleap solvate + OpenMM minimize/NVT/NPT)...")
    preps = list(prep_and_equilibrate.starmap(jobs))
    print(f"\n{'system':<10} {'ok':>4} {'atoms':>8} {'box (A)':<22} {'T after equil':>14} {'tleap_s':>8} {'equil_s':>8}")
    print("-" * 82)
    good = []
    for r in preps:
        if r.get("ok"):
            good.append(r["pair_id"])
            print(f"{r['pair_id']:<10} {'yes':>4} {r['n_atoms']:>8,} "
                  f"{str(r['box_a']):<22} {r['equil_temp_K']:>13.1f}K "
                  f"{r['tleap_s']:>8.1f} {r['equil_s']:>8.1f}")
        else:
            print(f"{r['pair_id']:<10} {'NO':>4}  failed at {r.get('stage')}")
            print("      " + r.get("log", "")[-400:].replace("\n", "\n      "))

    if not good:
        print("\nno systems prepared; stopping before production")
        return

    print(f"\nrunning STORMM explicit-solvent production on {len(good)} systems...")
    runs = list(stormm_production.starmap([(g, nstlim, 1) for g in good]))
    atoms = {r["pair_id"]: r["n_atoms"] for r in preps if r.get("ok")}
    print(f"\n{'system':<10} {'rc':>4} {'atoms':>8} {'wall_s':>8} {'ns/day':>9}")
    print("-" * 46)
    ps = nstlim * 2.0 / 1000.0
    for r in runs:
        pid = r["run_name"].replace("explicit-", "")
        nsday = (ps / 1000.0) / (r["wall_seconds"] / 86400.0) if r["wall_seconds"] else 0
        print(f"{pid:<10} {r['returncode']:>4} {atoms.get(pid,0):>8,} "
              f"{r['wall_seconds']:>8.1f} {nsday:>9.1f}")
        if r["returncode"] != 0:
            print("   " + r["stdout_tail"][-500:].replace("\n", "\n   "))
