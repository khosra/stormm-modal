"""Match Tsunami systems to their ligands, and retrieve the pre-built GAFF parameters.

The topology.pdb for each system carries the ligand as a LIG residue with atom names
(C1, C2, O1, ...). The tsunami-ligands volume holds, per InChIKey, a .mol2 with the
same atom names plus a .frcmod -- i.e. antechamber/parmchk2 has already been run.
Reusing those beats re-deriving chemistry from coordinates.

There is no pair_id -> InChIKey mapping in the dataset or the preprocessed index, and
listing order is not the mapping (it matches only 5/10). So this fingerprints each
ligand by (molecular formula, ordered atom-name tuple) and scans the volume.

Reads tsunami-ligands; writes only to this project's own volume.
"""

import json
import pathlib

import modal

app = modal.App("tsunami-ligand-match")
ligands_vol = modal.Volume.from_name("tsunami-ligands")
out_vol = modal.Volume.from_name("stormm-runs", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12")


def elem_of(gaff_type: str) -> str:
    """GAFF atom type -> element. Two-letter halogens must be tested first."""
    t = gaff_type.lower()
    if t.startswith("cl"):
        return "Cl"
    if t.startswith("br"):
        return "Br"
    return {"c": "C", "h": "H", "n": "N", "o": "O", "s": "S",
            "f": "F", "p": "P", "i": "I"}.get(t[0], t[0].upper())


def mol2_fingerprint(text: str):
    lines = text.splitlines()
    i = lines.index("@<TRIPOS>ATOM")
    j = next(k for k in range(i + 1, len(lines)) if lines[k].startswith("@<TRIPOS>"))
    names, counts = [], {}
    for row in lines[i + 1:j]:
        p = row.split()
        names.append(p[1])
        e = elem_of(p[5])
        counts[e] = counts.get(e, 0) + 1
    formula = "".join(f"{k}{v}" for k, v in sorted(counts.items()))
    return formula, tuple(names)


@app.function(image=image, volumes={"/ligands": ligands_vol, "/out": out_vol},
              cpu=4.0, timeout=1800)
def find_ligands(targets: list) -> dict:
    """targets: [{pair_id, formula, names:[...]}] -> {pair_id: {inchikey, mol2, frcmod}}"""
    # Reads go over a network volume mount, so this is latency-bound, not CPU-bound:
    # a serial walk of ~10k ligands did not finish in 14 minutes. Fan out with threads.
    from concurrent.futures import ThreadPoolExecutor

    want = {(t["formula"], tuple(t["names"])): t["pair_id"] for t in targets}
    root = pathlib.Path("/ligands/ligands")
    dirs = [d for d in root.iterdir() if d.is_dir()]

    def probe(d):
        m = d / f"{d.name}.mol2"
        try:
            fp = mol2_fingerprint(m.read_text())
        except Exception:
            return None
        pid = want.get(fp)
        if pid is None:
            return None
        frc = d / f"{d.name}.frcmod"
        meta = d / "meta.json"
        return pid, {
            "inchikey": d.name,
            "mol2": m.read_text(),
            "frcmod": frc.read_text() if frc.exists() else "",
            "meta": json.loads(meta.read_text()) if meta.exists() else {},
        }

    found, scanned = {}, 0
    with ThreadPoolExecutor(max_workers=64) as ex:
        for res in ex.map(probe, dirs):
            scanned += 1
            if res is not None:
                found[res[0]] = res[1]
    return {"scanned": scanned, "total_dirs": len(dirs), "found": found}


@app.local_entrypoint()
def main(pdb_dir: str = "/private/tmp/claude-501/-Users-amir-Documents-Github-stormm/3c6943e3-f0fa-43bf-a831-f09531240bfa/scratchpad/tsu10"):
    import collections
    targets = []
    for p in sorted(pathlib.Path(pdb_dir).glob("p*.pdb")):
        names, el = [], collections.Counter()
        for l in p.read_text().splitlines():
            if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() == "LIG":
                names.append(l[12:16].strip())
                el[(l[76:78].strip() or l[12:16].strip()[0]).capitalize()] += 1
        targets.append({"pair_id": p.stem,
                        "formula": "".join(f"{k}{v}" for k, v in sorted(el.items())),
                        "names": names})
    res = find_ligands.remote(targets)
    found = res["found"]
    print(f"scanned {res['scanned']} ligand dirs; matched {len(found)}/{len(targets)}")
    out = pathlib.Path(pdb_dir).parent / "ligmatch"
    out.mkdir(exist_ok=True)
    for pid, d in sorted(found.items()):
        (out / f"{pid}.mol2").write_text(d["mol2"])
        (out / f"{pid}.frcmod").write_text(d["frcmod"])
        (out / f"{pid}.meta.json").write_text(json.dumps(d["meta"], indent=2))
        print(f"  {pid}  {d['inchikey']}  charge={d['meta'].get('net_charge')}  "
              f"frcmod={'yes' if d['frcmod'] else 'MISSING'}")
    missing = [t["pair_id"] for t in targets if t["pair_id"] not in found]
    if missing:
        print("  unmatched:", ", ".join(missing))
