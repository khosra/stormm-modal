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
    temperatures: list | None = None,
) -> str:
    """Render a STORMM namelist input deck.

    Follows the shape of apps/Dyna/test/JacTest.sh, with an explicit &pppm block
    so the Ewald settings are recorded in the run rather than left to defaults.
    """
    label = system["label"]
    top = f"{STORMM_SRC}/{system['topology']}"
    crd = f"{STORMM_SRC}/{system['coordinates']}"

    temp_lines = ""
    if thermostat != 0:
        if temperatures:
            # Distinct starting temperatures per replica label break the degeneracy
            # of replicas that all start from the same coordinates, which is how we
            # confirm they are genuinely independent trajectories. This per-label
            # form follows apps/Dyna/test/AminoAcidTest.sh.
            for lbl, tempi, temp0 in temperatures:
                temp_lines += (
                    f"  temperature = {{ tempi {tempi:.1f}, temp0 {temp0:.1f},"
                    f" -label {lbl} }},\n"
                )
        else:
            temp_lines = (
                f"  temperature = {{ tempi 300.0, temp0 300.0, -label {label} }},\n"
            )

    energy_lines = "\n".join(f"  energy {term}," for term in ENERGY_TERMS)
    # 1e-05 is not a form the Amber-style namelist reader is guaranteed to take.
    dsum_tol_str = f"{pppm['dsum_tol']:.1e}".replace("e-0", "e-")

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
  cut = {dynamics['cut']},
  ntt = {thermostat},
  rigid_geom {dynamics['rigid_geom']},
{temp_lines}&end

&pppm
  theme {pppm['theme']},
  order {pppm['order']},
  cut {dynamics['cut']},
  dsum_tol {dsum_tol_str},
  mesh_ticks {pppm['mesh_ticks']},
&end

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
    # Confirm real sm_89 SASS is embedded rather than a silent PTX JIT fallback.
    out["sass"] = run(
        f"/usr/local/cuda/bin/cuobjdump --list-elf {DYNA} | head -20"
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
                       ph["ntwx"], replicas=n, thermostat=3, precision="single"),
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
