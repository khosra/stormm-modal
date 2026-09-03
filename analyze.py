"""Parse STORMM Matlab-syntax diagnostics, single- or multi-system.

Single-system runs emit `stormm_x = [ step value ];`. Multi-system runs emit a
preallocated matrix filled by column blocks:
    stormm_x = zeros(nrow, ncol);
    stormm_x(:, 1:11) = [ ... ];
    stormm_x(:, 12:17) = [ ... ];
Column 1 is the step number; the rest are one column per replica.
"""
import re
import pathlib

KB = 0.0019872041  # kcal/mol/K


def parse(path):
    txt = pathlib.Path(path).read_text()
    out = {}

    # Multi-system: preallocated matrix filled in column blocks.
    for m in re.finditer(r"^(stormm_\w+) = zeros\((\d+), (\d+)\);", txt, re.M):
        name, nrow, ncol = m.group(1), int(m.group(2)), int(m.group(3))
        mat = [[None] * ncol for _ in range(nrow)]
        for b in re.finditer(
            rf"^{re.escape(name)}\(:, (\d+):(\d+)\) = \[\n(.*?)^\];", txt, re.S | re.M
        ):
            c0, c1, body = int(b.group(1)), int(b.group(2)), b.group(3)
            for r, line in enumerate(body.strip().splitlines()):
                vals = [float(x) for x in line.split()]
                for k, v in enumerate(vals):
                    if c0 - 1 + k < ncol:
                        mat[r][c0 - 1 + k] = v
        out[name] = mat

    # Single-system: plain two-column list.
    for m in re.finditer(r"^(stormm_\w+) = \[\n(.*?)^\];", txt, re.S | re.M):
        name = m.group(1)
        if name in out:
            continue
        out[name] = [[float(x) for x in line.split()]
                     for line in m.group(2).strip().splitlines()]
    return out


def replicas(mat):
    """Split a parsed matrix into (steps, [per-replica series])."""
    steps = [row[0] for row in mat]
    ncol = len(mat[0])
    cols = [[row[c] for row in mat] for c in range(1, ncol)]
    return steps, cols


if __name__ == "__main__":
    import sys
    for run in sys.argv[1:]:
        d = parse(f"results/{run}/diagnostics.m")
        key = "stormm_total_energy"
        if key not in d:
            print(f"{run}: no {key}")
            continue
        steps, cols = replicas(d[key])
        n = len(cols)
        first = [c[0] for c in cols]
        last = [c[-1] for c in cols]
        print(f"\n===== {run}: {n} replica(s), {len(steps)} report points "
              f"(steps {int(steps[0])}..{int(steps[-1])}) =====")
        print("  start: " + " ".join(f"{v:.3f}" for v in first[:5]) + (" ..." if n > 5 else ""))
        print("  end:   " + " ".join(f"{v:.3f}" for v in last[:5]) + (" ..." if n > 5 else ""))
        if n > 1:
            s0, sN = max(first) - min(first), max(last) - min(last)
            print(f"  spread across replicas:  start {s0:.6f}   end {sN:.4f}")
            print("  -> " + ("INDEPENDENT trajectories (diverged)" if sN > 1e-3
                             else "IDENTICAL -- replicas are not diverging"))
        ke = d.get("stormm_kinetic")
        if ke:
            _, kcols = replicas(ke)
            kf = [c[0] for c in kcols]
            kl = [c[-1] for c in kcols]
            print(f"  kinetic: mean start {sum(kf)/len(kf):.1f} -> mean end {sum(kl)/len(kl):.1f}"
                  f"  (ratio {sum(kl)/sum(kf):.2f})")
