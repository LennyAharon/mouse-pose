#!/usr/bin/env python
"""
Write MANIFEST.json for the current data version, or diff it against another version.

The manifest records, per dataset: train/test frame and session counts, the keypoints it labels
(direct / trainable / eval), its camera views, the raw source folder and a SHA-256 of that
folder's label CSVs, plus the converter config hash and the mouse-pose commit. It is the
machine-readable half of DATA_VERSIONS.md; `--diff` prints the human half.

    python scripts/data_manifest.py                  # write <data_dir>/MANIFEST.json
    python scripts/data_manifest.py --diff v1        # diff current data_dir against data/head-fixed-v1
    python scripts/data_manifest.py --diff /path/to/other/MANIFEST.json
    python scripts/data_manifest.py --register ibl --note "relabeled tongue"   # new dataset version

Dataset versions live in <poseinterface>/DATASET_VERSIONS.json: per dataset, a list of
{version, date, raw_folder, raw_csv_sha256, note}. A dataset's version is its raw CSV hash; the
manifest names the version each dataset was built from, so a corpus version is the tuple.
"""

import argparse
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path

import yaml

from mouse_pose.paths import load_paths


def sha256_files(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def raw_folder_for(dataset: str, repo: Path) -> str:
    """convert_dataset.py reads raw_dir/<dataset>; an optional `raw_folder:` key in the dataset
    config (for a re-labeled dataset delivered as e.g. _raw/ibl-v2) overrides that."""
    cfg = repo / "configs" / "datasets" / f"{dataset}.yaml"
    c = yaml.safe_load(cfg.read_text()) if cfg.exists() else {}
    return (c or {}).get("raw_folder") or dataset


def registry_path(raw_dir: Path) -> Path:
    return raw_dir.parent / "DATASET_VERSIONS.json"


def load_registry(raw_dir: Path) -> dict:
    p = registry_path(raw_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def register(dataset: str, raw_dir: Path, repo: Path, note: str) -> dict:
    """Append a new version for `dataset` (keyed by the hash of its raw CSVs) to the registry."""
    reg = load_registry(raw_dir)
    raw_name = raw_folder_for(dataset, repo); raw = raw_dir / raw_name
    csvs = sorted(raw.glob("CollectedData*.csv"))
    if not csvs:
        raise SystemExit(f"no CollectedData*.csv under {raw}")
    h = sha256_files(csvs); versions = reg.setdefault(dataset, [])
    if versions and versions[-1]["raw_csv_sha256"] == h:
        print(f"{dataset}: raw CSVs unchanged since {versions[-1]['version']} — nothing to register"); return reg
    entry = {"version": f"{dataset}@v{len(versions) + 1}", "date": str(date.today()), "raw_folder": raw_name,
             "raw_csv_sha256": h, "note": note}
    versions.append(entry)
    registry_path(raw_dir).write_text(json.dumps(reg, indent=1) + "\n")
    print(f"registered {entry['version']}  [{h}]  {note}")
    return reg


def build_manifest(data_dir: Path, raw_dir: Path, repo: Path) -> dict:
    inv = json.loads((data_dir / "dataset_inventory.json").read_text())
    reg = load_registry(raw_dir)
    datasets = {}
    for name, d in inv["datasets"].items():
        raw_name = raw_folder_for(name, repo)
        raw = raw_dir / raw_name
        csvs = sorted(raw.glob("CollectedData*.csv")) if raw.exists() else []
        cfg = repo / "configs" / "datasets" / f"{name}.yaml"
        datasets[name] = {
            "train_frames": d["train_frames"], "test_frames": d["test_frames"],
            "train_sessions": d["train_sessions"], "test_sessions": d["test_sessions"],
            "views": sorted({v for v in d.get("views", {}).values() if v}),
            "direct": sorted(d["direct"]), "trainable": sorted(d["trainable"]), "eval": sorted(d["eval"]),
            "excluded_sessions": len(d.get("excluded", [])),
            "raw_folder": raw_name, "raw_csv_sha256": sha256_files(csvs) if csvs else None,
            "dataset_version": next((v["version"] for v in reg.get(name, []) if csvs and v["raw_csv_sha256"] == sha256_files(csvs)), None),
            "converter_config_sha256": sha256_files([cfg]) if cfg.exists() else None,
        }
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    return {
        "data_version": data_dir.name, "built": str(date.today()),
        "canonical_keypoints": inv.get("canonical_keypoints"),
        "n_canonical_keypoints": len(inv.get("canonical_keypoints") or []),
        "mouse_pose_commit": commit, "datasets": datasets,
        "merged_tags": sorted(p.name[len("CollectedData_"):-len("_train.csv")]
                              for p in data_dir.glob("CollectedData_*+*_train.csv")),
    }


def diff(new: dict, old: dict) -> str:
    out = [f"## {new['data_version']} vs {old['data_version']}", ""]
    nd, od = new["datasets"], old["datasets"]
    for name in sorted(set(nd) | set(od)):
        if name not in od:
            d = nd[name]; out.append(f"- **{name}: ADDED** ({d['train_frames']} train / {d['test_frames']} test frames, "
                                     f"{len(d['direct'])} keypoints, views {', '.join(d['views'])})"); continue
        if name not in nd:
            out.append(f"- **{name}: REMOVED**"); continue
        a, b = od[name], nd[name]; changes = []
        for k in ("train_frames", "test_frames", "train_sessions", "test_sessions", "excluded_sessions"):
            if a[k] != b[k]: changes.append(f"{k} {a[k]} → {b[k]}")
        for k in ("direct", "eval", "views"):
            added, removed = sorted(set(b[k]) - set(a[k])), sorted(set(a[k]) - set(b[k]))
            if added: changes.append(f"{k} +{added}")
            if removed: changes.append(f"{k} -{removed}")
        if a.get("dataset_version") != b.get("dataset_version"):
            changes.append(f"dataset version {a.get('dataset_version')} → {b.get('dataset_version')}")
        elif a["raw_folder"] != b["raw_folder"]: changes.append(f"raw folder {a['raw_folder']} → {b['raw_folder']}")
        elif a["raw_csv_sha256"] != b["raw_csv_sha256"]: changes.append("raw label CSVs changed (same folder, UNREGISTERED — run --register)")
        if a["converter_config_sha256"] != b["converter_config_sha256"]: changes.append("converter config changed")
        out.append(f"- **{name}:** " + ("; ".join(changes) if changes else "unchanged"))
    if new.get("canonical_keypoints") != old.get("canonical_keypoints"):
        out.append(f"- **vocabulary:** {old['n_canonical_keypoints']} → {new['n_canonical_keypoints']} keypoints")
    out.append(f"- merged tags: {', '.join(new['merged_tags']) or 'none'}")
    unchanged = [n for n in sorted(set(nd) & set(od)) if nd[n].get("raw_csv_sha256") == od[n].get("raw_csv_sha256")
                 and nd[n].get("converter_config_sha256") == od[n].get("converter_config_sha256")]
    changed = sorted((set(nd) | set(od)) - set(unchanged))
    out.append(f"- REUSABLE dedicated models (dataset unchanged): {', '.join(unchanged) or 'none'}")
    out.append(f"- MUST RETRAIN: dedicated models of {', '.join(changed) or 'nothing'}; every leave-one-out trunk and "
               f"the all-data trunk (they train on {', '.join(changed) or 'no changed dataset'}); all few-shot cells on changed datasets")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diff", metavar="VERSION_OR_MANIFEST",
                    help="print changes of the current data_dir vs this version (e.g. v1) or manifest path")
    ap.add_argument("--no-write", action="store_true", help="do not (re)write MANIFEST.json")
    ap.add_argument("--register", metavar="DATASET", help="register a new version of this dataset from its raw CSVs")
    ap.add_argument("--note", default="", help="what changed (with --register)")
    args = ap.parse_args()

    paths = load_paths()
    data_dir, raw_dir = Path(paths["data_dir"]).resolve(), Path(paths["raw_dir"])
    repo = Path(__file__).resolve().parents[1]
    if args.register:
        register(args.register, raw_dir, repo, args.note); return
    manifest = build_manifest(data_dir, raw_dir, repo)
    if not args.no_write:
        (data_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"wrote {data_dir / 'MANIFEST.json'}")
    for name, d in manifest["datasets"].items():
        print(f"  {name:14s} train {d['train_frames']:5d}  test {d['test_frames']:5d}  keypoints {len(d['direct']):2d}  "
              f"views {len(d['views'])}  {d.get('dataset_version') or 'UNREGISTERED'} [{d['raw_csv_sha256']}]")
    if args.diff:
        other = Path(args.diff)
        if not other.exists():
            other = data_dir.parent / f"head-fixed-{args.diff}" / "MANIFEST.json"
        print("\n" + diff(manifest, json.loads(other.read_text())))


if __name__ == "__main__":
    main()
