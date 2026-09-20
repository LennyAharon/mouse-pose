#!/usr/bin/env python
"""
Fit and evaluate ONE model from command-line arguments. The unit of work for a cluster.

Runs exactly what one cell of scripts/train_sweep.py runs (the same `litpose train` command,
same output layout, same evaluation on every dataset's test set), but for a single cell so a
scheduler can dispatch cells independently. Idempotent: a cell whose output directory already
holds eval/ is skipped, a cell whose training died is retrained from scratch. Exit code 0 only
when training and evaluation both succeeded, so the scheduler sees failures.

    python scripts/cluster/fit_one.py --config configs/model_zoominout.yaml \\
        --csv_file CollectedData_face+ibl+cheese+caz+kondo+hmv_train.csv --seed 0 \\
        --output_root <results_dir>/trunks [--sampling_temperature 2] [--head_mode shared]
        [--backbone vits_dinov3] [--train_frames 1] [--keep_checkpoints] [--dry_run]

Paths (data_dir, results_dir) come from paths.yaml; --data_dir / --output_root override them so
the same script runs on a cluster whose storage is mounted elsewhere.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from mouse_pose import train as T


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",               required=True, type=Path)
    ap.add_argument("--csv_file",             required=True, help="training CSV, relative to data_dir")
    ap.add_argument("--seed",                 required=True, type=int)
    ap.add_argument("--output_root",          type=Path, help="results root for this cell (default results_dir)")
    ap.add_argument("--data_dir",             type=Path, help="override data_dir from paths.yaml")
    ap.add_argument("--backbone",             default="vits_dinov3")
    ap.add_argument("--train_frames",         default="1")
    ap.add_argument("--sampling_temperature", default="2")
    ap.add_argument("--head_mode",            default="shared")
    ap.add_argument("--losses_to_use",        default="")
    ap.add_argument("--keep_checkpoints",     action="store_true")
    ap.add_argument("--eval_only",            action="store_true", help="skip training, evaluate an existing run")
    ap.add_argument("--dry_run",              action="store_true")
    args = ap.parse_args()

    if args.data_dir:
        T.DATA_DIR = args.data_dir.resolve()
    losses = [l for l in args.losses_to_use.split(",") if l]
    out = T.make_output_dir(args.csv_file, args.backbone, args.train_frames, args.seed, losses,
                            args.sampling_temperature, args.head_mode)
    if args.output_root:
        out = args.output_root / out.relative_to(T.RESULTS_DIR)
    label = f"{T.csv_stem(args.csv_file)} seed={args.seed} T={args.sampling_temperature} head={args.head_mode} {args.backbone}"
    print(f"── {label}\n   output: {out}")

    if (out / "eval").is_dir() and any((out / "eval").iterdir()):
        print("   already evaluated, nothing to do"); return 0

    if not args.eval_only:
        if out.exists():
            print("   incomplete run found, retraining from scratch")
        cmd = T.make_train_command(args.csv_file, args.backbone, args.train_frames, args.seed, losses, out,
                                   False, args.sampling_temperature, args.head_mode, config_file=args.config)
        print("   " + " ".join(cmd))
        if args.dry_run:
            return 0
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"   ERROR training exit {r.returncode}", file=sys.stderr); return r.returncode
        print(f"   trained in {(time.time() - t0) / 60:.0f} min")
    if args.dry_run:
        return 0
    try:
        T.evaluate_model(out, args.csv_file, keep_checkpoints=args.keep_checkpoints)
    except Exception as e:  # noqa: BLE001 — surface any evaluation failure to the scheduler
        print(f"   ERROR evaluation: {e}", file=sys.stderr); return 3
    print("   done"); return 0


if __name__ == "__main__":
    sys.exit(main())
