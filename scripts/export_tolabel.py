#!/usr/bin/env python
"""
Export an active-labeling dataset (its CollectedData_tolabel.csv) to corpus format, keeping the
original train / test split, as a derived dataset next to the corpus CSVs.

The rows of `_raw/<dataset>/CollectedData_tolabel.csv` come from both original splits (the
manifest's `source` column says which). They are converted exactly like `convert_dataset.py`
converts the dataset (configs/datasets/<dataset>.yaml: exclusions, laterality, canonical names,
visibility: labeled = 2, empty = 1 occluded), plus the keypoints being labeled, mapped with --map
(their dataset config entry is only added once labeling is done). Image paths stay under
labeled-data/<dataset>/ (the corpus frames). Sessions the dataset config excludes are dropped.
Writes CollectedData_<name>_{train,test}.csv into data_dir (paths.yaml) and nothing else.

    python scripts/export_tolabel.py --dataset ibl --name ibl-digits \
        --map d1_tip_r=d1_tip_left d2_tip_r=d2_tip_left d3_tip_r=d3_tip_left d4_tip_r=d4_tip_left
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from mighty_mouse.configs import default_configs_dir, load_canonical_keypoints, load_dataset_config
from mighty_mouse.convert import (
    post_process,
    process_split,
    raw_dataset_dir,
    validate_dataset_config,
)
from mighty_mouse.labels import read_labels_csv
from mighty_mouse.paths import load_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",  required=True, help="raw dataset (subdirectory of raw_dir)")
    ap.add_argument("--name",     required=True, help="name of the derived corpus dataset")
    ap.add_argument("--map",      nargs="+", default=[], help="raw=canonical, labeled keypoints")
    ap.add_argument("--data_dir", type=Path, default=None, help="output dir (default: paths.yaml)")
    ap.add_argument("--dry_run",  action="store_true")
    args = ap.parse_args()

    paths    = load_paths()
    data_dir = args.data_dir or Path(paths["data_dir"])
    raw_root = Path(paths["raw_dir"])
    configs  = default_configs_dir()
    canon    = load_canonical_keypoints(configs)
    config   = load_dataset_config(configs, args.dataset)
    config["keypoints"] = {**config["keypoints"], **dict(m.split("=", 1) for m in args.map)}
    raw_dir  = raw_dataset_dir(raw_root, args.dataset, config)
    manifest = raw_root.parent / "results" / args.dataset / "active" / "tolabel_manifest.csv"

    tolabel = read_labels_csv(raw_dir / "CollectedData_tolabel.csv")
    source  = pd.read_csv(manifest, index_col=0)["source"]
    if set(source.index) != set(tolabel.index):
        sys.exit(f"ERROR: {manifest} and the tolabel rows differ")
    splits = {s: tolabel.loc[[f for f in tolabel.index if source[f] == s]]
              for s in ("train", "test")}
    errors = validate_dataset_config(config, canon, splits.values())
    if errors:
        sys.exit("validation errors:\n  " + "\n  ".join(errors))

    for split, df in splits.items():
        out = post_process(process_split(df, config, args.dataset, canon), config, args.dataset)
        missing = [f for f in out.index if not (data_dir / f).exists()]
        n_lab = {kp: int((out.xs((kp, "visible"), axis=1, level=(1, 2)).iloc[:, 0] == 2).sum())
                 for kp in (m.split("=", 1)[1] for m in args.map)}
        print(f"{split}: {len(df)} tolabel rows -> {len(out)} frames; frames missing in data_dir: "
              f"{len(missing)}; labeled (visible 2) per mapped keypoint: {n_lab}")
        if missing:
            sys.exit(f"ERROR: e.g. {missing[0]} is not in {data_dir}")
        target = data_dir / f"CollectedData_{args.name}_{split}.csv"
        if args.dry_run:
            print(f"  (dry run) would write {target}")
        else:
            out.to_csv(target)
            print(f"  wrote {target}")


if __name__ == "__main__":
    main()
