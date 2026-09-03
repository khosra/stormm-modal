"""Validate STORMM explicit-solvent (PME) molecular dynamics on a Modal GPU.

The question this harness answers: can https://github.com/Psivant/stormm actually
run explicit-solvent MD on a GPU, one simulation at a time and then 10+ small ones
concurrently?

STORMM's design point is the PhaseSpaceSynthesis, which holds many systems at once
so a single process saturates one GPU with many independent trajectories. The `-n`
flag on a `-sys` block is how you ask for that, so "10+ small simulations on a GPU"
is a single `dynamics.stormm.cuda` invocation, not 10 of them.

Usage:
    uv run modal run stormm_modal.py::smoke
    uv run modal run stormm_modal.py::phase_b
    uv run modal run stormm_modal.py::phase_c
"""

import pathlib
import tomllib

import modal

HERE = pathlib.Path(__file__).parent


def load_config() -> dict:
    """Read runs.toml. Local-side only; remote functions receive plain dicts."""
    return tomllib.loads((HERE / "runs.toml").read_text())


_CFG = load_config()
_BUILD = _CFG["build"]
_MODAL = _CFG["modal"]

STORMM_SRC = "/app/stormm"
STORMM_BUILD = "/app/stormmbuild"
DYNA = f"{STORMM_BUILD}/apps/Dyna/dynamics.stormm.cuda"

app = modal.App(_MODAL["app"])
results_vol = modal.Volume.from_name(_MODAL["volume"], create_if_missing=True)

# Build STORMM from a pinned commit rather than Image.from_dockerfile("Dockerfile"),
# so the GPU arch can be narrowed to a single target and the source is reproducible
# without editing anything upstream. Note that cmake configure clones PocketFFT from
# gitlab.mpcdf.mpg.de, so the builder needs outbound network (it has it).
#
# make_jobs is capped deliberately. STORMM's CUDA translation units are individually
# memory-hungry and Image.run_commands exposes no cpu/memory knobs, so an unbounded
# -j risks an OOM kill on the builder.
stormm_image = (
    modal.Image.from_registry(_BUILD["cuda_image"], add_python="3.12")
    .apt_install("git", "cmake", "g++", "libeigen3-dev", "ca-certificates")
    # These must be set BEFORE cmake configures, because cmake bakes CXXFLAGS into
    # its cache. STORMM's plain-C++ translation units include cuda_runtime.h and are
    # compiled by g++, not nvcc, so without the CUDA include path on CXXFLAGS the
    # build dies at src/Accelerator/hybrid.h. The repo's own Dockerfile:28-30 sets
    # exactly these; dropping them is what broke the first build attempt.
    .env(
        {
            "CUDADIR": "/usr/local/cuda",
            "CUDACXX": "/usr/local/cuda/bin/nvcc",
            "CXXFLAGS": "-I/usr/local/cuda/include",
            "LDFLAGS": "-L/usr/local/cuda/lib64",
        }
    )
    .run_commands(
        f"git clone {_BUILD['repo']} {STORMM_SRC}",
        f"cd {STORMM_SRC} && git checkout {_BUILD['commit']}",
        (
            f"cmake -S {STORMM_SRC} -B {STORMM_BUILD}"
            " -DCMAKE_BUILD_TYPE=RELEASE"
            " -DSTORMM_ENABLE_CUDA=YES"
            " -DSTORMM_ENABLE_RDKIT=NO"
            f" -DCUSTOM_GPU_ARCH={_BUILD['gpu_arch']}"
            f" -DCUSTOM_NVCC_THREADS={_BUILD['nvcc_threads']}"
            " -DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include"
            " -DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64"
        ),
        f"cmake --build {STORMM_BUILD} -j {_BUILD['make_jobs']}",
    )
    .env(
        {
            "STORMM_HOME": STORMM_SRC,
            "STORMM_SOURCE": STORMM_SRC,
            "STORMM_BUILD": STORMM_BUILD,
            "STORMM_VERBOSE": "COMPACT",
        }
    )
    # Module-level code runs remotely too, and the decorators below need the config
    # at import time (gpu=, volume names), so runs.toml has to exist in the container
    # before this module is imported. Remotely __file__ is /root/stormm_modal.py, so
    # HERE resolves to /root and this lands exactly where load_config() looks.
    # add_local_file is applied at startup and does not invalidate the build layers.
    .add_local_file("runs.toml", "/root/runs.toml")
)


# A second image built from STORMM main as of 2025-07-17 -- what Andre's unpinned
# `git clone psivant/stormm` would have produced when the Tsunami container was last
# built. Its only purpose is to establish whether the Langevin energy-sink bug
# predates the ~97k-system campaign, which code archaeology could not settle cleanly.
LEGACY_SRC = "/app/stormm_legacy"
LEGACY_BUILD = "/app/stormmbuild_legacy"
LEGACY_DYNA = f"{LEGACY_BUILD}/apps/Dyna/dynamics.stormm.cuda"

stormm_legacy_image = (
    modal.Image.from_registry(_BUILD["cuda_image"], add_python="3.12")
    .apt_install("git", "cmake", "g++", "libeigen3-dev", "ca-certificates")
    .env({"CUDADIR": "/usr/local/cuda", "CUDACXX": "/usr/local/cuda/bin/nvcc",
          "CXXFLAGS": "-I/usr/local/cuda/include", "LDFLAGS": "-L/usr/local/cuda/lib64"})
    .run_commands(
        f"git clone {_BUILD['repo']} {LEGACY_SRC}",
        f"cd {LEGACY_SRC} && git checkout {_BUILD['legacy_commit']}",
        (
            f"cmake -S {LEGACY_SRC} -B {LEGACY_BUILD}"
            " -DCMAKE_BUILD_TYPE=RELEASE -DSTORMM_ENABLE_CUDA=YES -DSTORMM_ENABLE_RDKIT=NO"
            f" -DCUSTOM_GPU_ARCH={_BUILD['gpu_arch']}"
            f" -DCUSTOM_NVCC_THREADS={_BUILD['nvcc_threads']}"
            " -DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include"
            " -DCMAKE_EXE_LINKER_FLAGS=-L/usr/local/cuda/lib64"
        ),
        f"cmake --build {LEGACY_BUILD} -j {_BUILD['make_jobs']}",
    )
    .env({"STORMM_HOME": LEGACY_SRC, "STORMM_SOURCE": LEGACY_SRC,
          "STORMM_BUILD": LEGACY_BUILD, "STORMM_VERBOSE": "COMPACT"})
    .add_local_file("runs.toml", "/root/runs.toml")
)


# ---------------------------------------------------------------------------
# Input deck generation
# ---------------------------------------------------------------------------

ENERGY_TERMS = [
    "total", "bond", "angle", "dihedral", "electrostatic",
    "vdw", "elec_14", "vdw_14", "kinetic",
]


def build_deck(
    system: dict,
    dynamics: dict,
    pppm: dict,
    nstlim: int,
    ntpr: int,
    ntwx: int,
    replicas: int,
    thermostat: int,
    precision: str,
    igb: int = 0,
) -> str:
    """Render a STORMM namelist input deck.

    Two shapes, following the two driver scripts that ship with STORMM:

    - Periodic / explicit solvent, after apps/Dyna/test/JacTest.sh: a real-space
      `cut` in &dynamics plus an explicit &pppm block, so the Ewald settings are
      recorded in the run rather than left to defaults.
    - Isolated / implicit solvent, after apps/Dyna/test/AminoAcidTest.sh: no cutoff
      and no &pppm, with a &solvent block naming the Generalized Born model.

    igb=8 is NECK_GB_II, i.e. GB-Neck2 (src/Topology/atomgraph_enumerators.h:147).
    """
    periodic = bool(system.get("periodic", True))
    if periodic and igb:
        raise ValueError(
            f"{system['label']}: Generalized Born is an isolated-boundary model and "
            "cannot be combined with a periodic unit cell"
        )

    label = system["label"]
    top = f"{STORMM_SRC}/{system['topology']}"
    crd = f"{STORMM_SRC}/{system['coordinates']}"

    temp_line = ""
    if thermostat != 0:
        temp_line = f"  temperature = {{ tempi 300.0, temp0 300.0, -label {label} }},\n"

    if periodic:
        cut_line = f"  cut = {dynamics['cut']},\n"
        # 1e-05 is not a form the Amber-style namelist reader is guaranteed to take.
        dsum_tol_str = f"{pppm['dsum_tol']:.1e}".replace("e-0", "e-")
        solvent_block = f"""&pppm
  theme {pppm['theme']},
  order {pppm['order']},
  cut {dynamics['cut']},
  dsum_tol {dsum_tol_str},
  mesh_ticks {pppm['mesh_ticks']},
&end
"""
    else:
        cut_line = ""
        solvent_block = f"""&solvent
  igb = {igb},
&end
"""

    energy_lines = "\n".join(f"  energy {term}," for term in ENERGY_TERMS)

    return f"""&files
  -sys {{ -p {top}
         -c {crd}
         -label {label} -n {replicas} }}
  -o diagnostics.m
  -x traj.crd
  x_kind AMBER_CRD
&end

&dynamics
  nstlim = {nstlim},  ntpr = {ntpr},  ntwx = {ntwx},  dt = {dynamics['dt']},
{cut_line}  ntt = {thermostat},
  rigid_geom {dynamics['rigid_geom']},
{temp_line}&end

{solvent_block}
&precision
  nonbonded {precision},
  valence {precision},
&end

&report
  syntax = Matlab,
{energy_lines}
  ascii_salvage STARS,
&end
"""


# ---------------------------------------------------------------------------
# Remote functions
# ---------------------------------------------------------------------------


@app.function(image=stormm_image, gpu=_MODAL["gpu"], timeout=3600)
def smoke(ctest_regex: str = "") -> dict:
    """Phase A: confirm the binary links, targets the right SASS, and passes tests."""
    import subprocess

    def run(cmd, **kw):
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-8000:],
            "stderr": proc.stderr[-4000:],
        }

    out = {}
    out["nvidia_smi"] = run("nvidia-smi")
    # Confirm real sm_89 SASS is present rather than a silent PTX JIT fallback.
    # The device code is NOT in the executable -- dynamics.stormm.cuda is a ~250 KB
    # driver that links against the CUDA object library, so cuobjdump has to be
    # pointed at the library or it reports "does not contain device code".
    out["libs"] = run(f"find {STORMM_BUILD} -name '*.so*' -o -name '*.a' | head -20")
    out["sass"] = run(
        "for f in $(find %s -name 'libstormm*' | head -3); do"
        "  echo \"== $f\";"
        "  /usr/local/cuda/bin/cuobjdump --list-elf \"$f\" 2>&1 | head -6;"
        "done" % STORMM_BUILD
    )
    out["binaries"] = run(f"ls -la {STORMM_BUILD}/apps/*/")
    out["help"] = run(f"{DYNA} --help 2>&1 | head -40")
    # Enumerate the test suite first; we do not yet know STORMM's ctest naming.
    out["ctest_list"] = run(f"cd {STORMM_BUILD} && ctest -N")
    if ctest_regex:
        out["ctest_run"] = run(
            f"cd {STORMM_BUILD} && ctest -R '{ctest_regex}' --output-on-failure",
            timeout=1800,
        )
    return out


@app.function(
    image=stormm_image,
    gpu=_MODAL["gpu"],
    timeout=3600,
    volumes={"/results": results_vol},
)
def run_dynamics(run_name: str, deck: str, run_timeout: int = 3000) -> dict:
    """Run one dynamics.stormm.cuda invocation and archive everything it produced."""
    import json
    import shutil
    import subprocess
    import threading
    import time

    work = pathlib.Path("/tmp/run") / run_name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "md.in").write_text(deck)

    # Sample GPU memory so the report can quote a high-water mark.
    peak_mib = 0
    stop = threading.Event()

    def poll():
        nonlocal peak_mib
        while not stop.is_set():
            try:
                p = subprocess.run(
                    "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                    shell=True, capture_output=True, text=True, timeout=10,
                )
                if p.returncode == 0 and p.stdout.strip():
                    peak_mib = max(peak_mib, int(p.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            stop.wait(2.0)

    watcher = threading.Thread(target=poll, daemon=True)
    watcher.start()

    t0 = time.monotonic()
    timed_out = False
    try:
        proc = subprocess.run(
            [DYNA, "-O", "-i", "md.in", "-except", "warn"],
            cwd=work, capture_output=True, text=True, timeout=run_timeout,
        )
        rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = -1
        stdout = (exc.stdout or b"").decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = (exc.stderr or b"").decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
    wall = time.monotonic() - t0
    stop.set()
    watcher.join(timeout=5)

    (work / "stdout.txt").write_text(stdout)
    (work / "stderr.txt").write_text(stderr)

    summary = {
        "run_name": run_name,
        "returncode": rc,
        "timed_out": timed_out,
        "wall_seconds": round(wall, 2),
        "peak_gpu_mib": peak_mib,
        "stdout_tail": stdout[-6000:],
        "stderr_tail": stderr[-6000:],
        "artifacts": sorted(p.name for p in work.iterdir()),
    }
    (work / "summary.json").write_text(json.dumps(summary, indent=2))

    dest = pathlib.Path("/results") / run_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(work, dest)
    results_vol.commit()

    return summary


# ---------------------------------------------------------------------------
# Local entrypoints
# ---------------------------------------------------------------------------


def _report(results: list) -> None:
    print("\n" + "=" * 78)
    print(f"{'run':<34} {'rc':>4} {'wall_s':>9} {'peak_MiB':>9}")
    print("-" * 78)
    for r in results:
        print(
            f"{r['run_name']:<34} {r['returncode']:>4} "
            f"{r['wall_seconds']:>9.1f} {r['peak_gpu_mib']:>9}"
        )
    print("=" * 78)
    for r in results:
        if r["returncode"] != 0:
            print(f"\n--- {r['run_name']} FAILED (rc={r['returncode']}) ---")
            print(r["stdout_tail"][-3000:])
            print(r["stderr_tail"][-2000:])


@app.local_entrypoint()
def smoke_test(ctest_regex: str = ""):
    """Phase A. Pass --ctest-regex to also run a subset of the test suite."""
    out = smoke.remote(ctest_regex=ctest_regex)
    for key, res in out.items():
        print(f"\n{'=' * 78}\n## {key}  (rc={res['returncode']})\n{'=' * 78}")
        print(res["stdout"])
        if res["stderr"].strip():
            print("--- stderr ---")
            print(res["stderr"])


@app.local_entrypoint()
def phase_b(nstlim: int = 0):
    """One explicit-solvent simulation on one GPU: JAC / DHFR, 23,558 atoms."""
    cfg = load_config()
    ph = cfg["phases"]["b"]
    system = cfg["systems"][ph["system"]]
    steps = nstlim or ph["nstlim"]

    jobs = []
    # NVE is the sharp test: with no thermostat absorbing error, total-energy drift
    # is a direct readout of whether the PME forces are consistent with the energy.
    jobs.append(("jac-nve-single", build_deck(
        system, cfg["dynamics"], cfg["pppm"], steps, ph["ntpr"], ph["ntwx"],
        replicas=1, thermostat=0, precision="single")))
    # NVT confirms the Langevin thermostat path holds temperature under PBC.
    jobs.append(("jac-nvt-single", build_deck(
        system, cfg["dynamics"], cfg["pppm"], steps, ph["ntpr"], ph["ntwx"],
        replicas=1, thermostat=3, precision="single")))
    # A shorter double-precision run exercises the other CellGrid template
    # instantiation in simulator.cpp; agreement between the two is the strongest
    # single piece of evidence that the PME path is correct rather than merely alive.
    jobs.append(("jac-nve-double", build_deck(
        system, cfg["dynamics"], cfg["pppm"], max(steps // 10, 500), ph["ntpr"],
        ph["ntwx"], replicas=1, thermostat=0, precision="double")))

    results = list(run_dynamics.starmap([(n, d) for n, d in jobs]))
    _report(results)


@app.local_entrypoint()
def phase_c(nstlim: int = 0):
    """10+ small explicit-solvent simulations concurrently on one GPU."""
    cfg = load_config()
    ph = cfg["phases"]["c"]
    steps = nstlim or ph["nstlim"]

    jobs = []
    for entry in ph["matrix"]:
        system = cfg["systems"][entry["system"]]
        n = entry["replicas"]
        # All replicas in one -sys block share a label, so a per-replica starting
        # temperature is not expressible here. Langevin (ntt=3) gives each replica
        # its own stochastic forces, so identical starting coordinates still
        # decorrelate; we verify that from the per-replica energies in the output.
        jobs.append((
            f"{entry['system']}-n{n}",
            build_deck(system, cfg["dynamics"], cfg["pppm"], steps, ph["ntpr"],
                       ph["ntwx"], replicas=n,
                       thermostat=cfg["dynamics"]["thermostat"], precision="single"),
        ))

    results = list(run_dynamics.starmap([(n, d) for n, d in jobs]))
    _report(results)

    print("\nConcurrency payoff (wall time for N replicas vs 1):")
    by_name = {r["run_name"]: r for r in results}
    for entry in ph["matrix"]:
        n = entry["replicas"]
        if n == 1:
            continue
        one = by_name.get(f"{entry['system']}-n1")
        many = by_name.get(f"{entry['system']}-n{n}")
        if one and many and one["returncode"] == 0 and many["returncode"] == 0:
            speedup = (one["wall_seconds"] * n) / many["wall_seconds"]
            print(
                f"  {entry['system']:<12} n=1: {one['wall_seconds']:7.1f}s   "
                f"n={n}: {many['wall_seconds']:7.1f}s   "
                f"effective speedup vs serial: {speedup:5.2f}x"
            )


@app.local_entrypoint()
def phase_gb(nstlim: int = 0):
    """Implicit (GB-Neck2) vs explicit solvent cost on identical hardware.

    Motivated by the Tsunami workload: IDR + small-molecule data generation that
    runs in implicit solvent today and wants to move to explicit. symmetry_C1 and
    symmetry_C1_in_water are the same solute, so that pair gives a clean ratio.
    """
    cfg = load_config()
    ph = cfg["phases"]["gb"]
    steps = nstlim or ph["nstlim"]
    n = ph["replicas"]

    jobs = []
    for entry in ph["matrix"]:
        system = cfg["systems"][entry["system"]]
        igb = entry["igb"]
        tag = f"gb{igb}" if igb else "explicit"
        jobs.append((
            f"{entry['system']}-{tag}-n{n}",
            build_deck(system, cfg["dynamics"], cfg["pppm"], steps, ph["ntpr"],
                       ph["ntwx"], replicas=n,
                       thermostat=cfg["dynamics"]["thermostat"], precision="single",
                       igb=igb),
        ))

    results = list(run_dynamics.starmap([(name, deck) for name, deck in jobs]))
    _report(results)

    by_name = {r["run_name"]: r for r in results}
    dry = by_name.get(f"symmetry_C1-gb8-n{n}")
    wet = by_name.get(f"symmetry_C1_in_water-explicit-n{n}")
    if dry and wet and dry["returncode"] == 0 and wet["returncode"] == 0:
        print(
            f"\nSame-solute cost of explicit solvent (symmetry_C1, {n} replicas):\n"
            f"  implicit GB-Neck2 (22 atoms):   {dry['wall_seconds']:7.1f}s\n"
            f"  explicit water   (496 atoms):   {wet['wall_seconds']:7.1f}s\n"
            f"  explicit / implicit:            {wet['wall_seconds'] / dry['wall_seconds']:6.2f}x"
        )


@app.local_entrypoint()
def thermostat_probe(nstlim: int = 4000):
    """Diagnose why ntt=3 drains kinetic energy instead of holding temperature.

    The Phase B NVT run cooled JAC from ~300 K to ~37 K in 9.5 ps, decaying
    monotonically at roughly the default gamma_ln. That is the signature of
    friction applied without a compensating stochastic force. The evolution
    window and RNG cache defaults were both ruled out by reading source, so this
    varies one knob at a time to find which one the behaviour actually tracks.
    """
    cfg = load_config()
    system = cfg["systems"]["ubiquitin"]   # small enough to iterate quickly
    dyn = dict(cfg["dynamics"])
    ppp = cfg["pppm"]
    base = dict(nstlim=nstlim, ntpr=200, ntwx=nstlim, replicas=1, precision="single")

    def deck(**over):
        extra = over.pop("extra", "")
        d = build_deck(system, dyn, ppp, base["nstlim"], base["ntpr"], base["ntwx"],
                       base["replicas"], over.pop("thermostat"), base["precision"])
        if extra:
            d = d.replace("&end\n\n&pppm", extra + "&end\n\n&pppm", 1)
        return d

    variants = [
        ("probe-nve",              deck(thermostat=0)),
        ("probe-langevin-default", deck(thermostat=3)),
        ("probe-langevin-gamma1",  deck(thermostat=3, extra="  gamma_ln = 1.0,\n")),
        ("probe-langevin-gamma01", deck(thermostat=3, extra="  gamma_ln = 0.01,\n")),
        ("probe-andersen",         deck(thermostat=2)),
        ("probe-berendsen",        deck(thermostat=1)),
        ("probe-langevin-norigid", deck(thermostat=3).replace("rigid_geom on", "rigid_geom off")),
    ]
    results = list(run_dynamics.starmap([(n, d) for n, d in variants]))
    _report(results)


@app.local_entrypoint()
def implicit_thermostat_probe(nstlim: int = 20000):
    """Does the Langevin energy-sink bug also affect IMPLICIT-solvent runs?

    This matters beyond STORMM itself. The Tsunami dataset (~97k systems) was
    generated with `ntt = 3` and `igb = 8` on isolated systems. The Phase B
    finding was measured on periodic systems only, so it does not transfer
    automatically. If Langevin also drains energy under isolated boundary
    conditions, that dataset consists of quenched trajectories.

    Reproduces Andre's deck shape: heat from tempi 100 K to temp0 300 K across an
    evolution window, GB-Neck2 implicit solvent, isolated boundaries.
    """
    cfg = load_config()
    system = cfg["systems"]["gly_arg"]
    dyn = dict(cfg["dynamics"])
    ppp = cfg["pppm"]

    def deck(ntt, ramp=True):
        d = build_deck(system, dyn, ppp, nstlim, max(nstlim // 20, 1), nstlim,
                       replicas=4, thermostat=ntt, precision="single", igb=8)
        if ramp and ntt != 0:
            # Andre's shape: ramp 100 K -> 300 K over an explicit evolution window.
            d = d.replace(
                "  temperature = { tempi 300.0, temp0 300.0, -label GlyArg },\n",
                "  temperature = { tempi 100.0, temp0 300.0, -label GlyArg },\n"
                f"  tevo_start = {nstlim // 8}, tevo_end = {nstlim * 3 // 8},\n"
                "  tcache_depth 1,\n")
        return d

    variants = [
        ("imp-nve",                deck(0)),
        ("imp-langevin-ramp",      deck(3)),           # <- Andre's configuration
        ("imp-andersen-ramp",      deck(2)),
        ("imp-langevin-flat",      deck(3, ramp=False)),
        ("imp-andersen-flat",      deck(2, ramp=False)),
    ]
    results = list(run_dynamics.starmap([(n, d) for n, d in variants]))
    _report(results)


@app.function(image=stormm_legacy_image, gpu=_MODAL["gpu"], timeout=3600,
              volumes={"/results": results_vol})
def run_dynamics_legacy(run_name: str, deck: str, run_timeout: int = 3000) -> dict:
    """Same as run_dynamics, against the July-2025 STORMM build."""
    import json, shutil, subprocess, time
    work = pathlib.Path("/tmp/run") / run_name
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    (work / "md.in").write_text(deck)
    t0 = time.monotonic()
    proc = subprocess.run([LEGACY_DYNA, "-O", "-i", "md.in", "-except", "warn"],
                          cwd=work, capture_output=True, text=True, timeout=run_timeout)
    wall = time.monotonic() - t0
    (work / "stdout.txt").write_text(proc.stdout)
    (work / "stderr.txt").write_text(proc.stderr)
    summary = {"run_name": run_name, "returncode": proc.returncode, "timed_out": False,
               "wall_seconds": round(wall, 2), "peak_gpu_mib": 0,
               "stdout_tail": proc.stdout[-6000:], "stderr_tail": proc.stderr[-6000:],
               "artifacts": sorted(p.name for p in work.iterdir())}
    (work / "summary.json").write_text(json.dumps(summary, indent=2))
    dest = pathlib.Path("/results") / run_name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(work, dest)
    results_vol.commit()
    return summary


@app.local_entrypoint()
def legacy_thermostat_probe(nstlim: int = 20000):
    """Did the Langevin energy-sink bug exist when Tsunami was generated?

    Runs the implicit-solvent thermostat comparison against STORMM main as of
    2025-07-17. If Langevin plateaus well below NVE and Andersen here too, the
    campaign's ~97k systems were sampled at the wrong temperature.
    """
    cfg = load_config()
    system = cfg["systems"]["gly_arg"]
    dyn, ppp = dict(cfg["dynamics"]), cfg["pppm"]

    def deck(ntt):
        d = build_deck(system, dyn, ppp, nstlim, max(nstlim // 20, 1), nstlim,
                       replicas=4, thermostat=ntt, precision="single", igb=8)
        if ntt != 0:
            d = d.replace(
                "  temperature = { tempi 300.0, temp0 300.0, -label GlyArg },\n",
                "  temperature = { tempi 100.0, temp0 300.0, -label GlyArg },\n"
                f"  tevo_start = {nstlim // 8}, tevo_end = {nstlim * 3 // 8},\n"
                "  tcache_depth 1,\n")
        return d

    variants = [("legacy-imp-nve", deck(0)),
                ("legacy-imp-langevin", deck(3)),
                ("legacy-imp-andersen", deck(2))]
    results = list(run_dynamics_legacy.starmap([(n, d) for n, d in variants]))
    _report(results)
