#!/usr/bin/env python3
"""
Cartesian training sweep over head-fixed datasets using the litpose CLI —
local, sequential.

Loops over all combinations of csv_files × backbones × train_frames × seeds
and calls `litpose train` for each, then evaluates in-process.

For remote, parallel execution on Lightning AI, see train_sweep_lightning.py —
it shares all combo/naming/command-building logic with this script via
mouse_pose.train, so the two only differ in how a command actually gets run.

Output lands at:
  <results_dir>/<tag>/<losses_tag>/tf<N>/<backbone>/seed<N>/
  └── eval/<dataset_name>/pixel_error.csv   ← per-model evaluation results

Merged CSVs (produced by build_dataset.py --tag <tag>) are evaluated against
all per-dataset test sets. Pixel error is NaN for keypoints absent from a dataset.

Usage examples:
  # dry run to preview all commands
  python scripts/train_sweep.py --dry_run \\
      --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl+cheese_train.csv" \\
      --train_frames "200;400;600" \\
      --seeds "0;1;2"

  # full sweep
  python scripts/train_sweep.py \\
      --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl+cheese_train.csv" \\
      --train_frames "200;400;600" \\
      --seeds "0;1;2" \\
      --backbones "vits_dino"

  # safe to re-run after interruption
  python scripts/train_sweep.py --skip_existing ...
"""

import argparse
import subprocess
import sys
from pathlib import Path


from mouse_pose.train import (
    RESULTS_DIR,
    build_combos,
    csv_stem,
    evaluate_model,
    make_output_dir,
    make_train_command,
    parse_semicolon_list,
)


def main():
    parser = argparse.ArgumentParser(
        description="Head-fixed LP training sweep (local, sequential)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv_files", default="CollectedData_face+ibl+cheese_train.csv",
        help='semicolon-separated CSV filenames relative to data_dir',
    )
    parser.add_argument(
        "--train_frames", default="1",
        help='semicolon-separated frame counts; 1 = all frames',
    )
    parser.add_argument("--seeds",        default="0",                   help='semicolon-separated rng seeds, e.g. "0;1;2"')
    parser.add_argument("--backbones",    default="resnet50_animal_ap10k", help='semicolon-separated backbone names')
    parser.add_argument("--losses_to_use", default="",                   help='comma-separated loss names; empty = supervised only')
    parser.add_argument(
        "--sampling_temperatures", default="",
        help='semicolon-separated multi-dataset sampling temperatures, e.g. "1;2;inf". '
             "1 = stock frame-proportional loader (no path component, same outputs as "
             "before this flag existed); empty = one stock run.",
    )
    parser.add_argument(
        "--head_modes", default="",
        help='semicolon-separated head modes, e.g. "shared;per_dataset". shared = stock '
             "single head (no path component); empty = one shared run.",
    )
    parser.add_argument("--debug",         action="store_true",          help="Smoke-test run (3 epochs)")
    parser.add_argument("--dry_run",       action="store_true",          help="Print commands without running")
    parser.add_argument("--skip_existing", action="store_true",          help="Skip combos whose output dir already exists")
    parser.add_argument("--eval_only",     action="store_true",          help="Skip training; only run evaluation on existing model dirs")
    parser.add_argument("--keep_checkpoints", action="store_true",        help="Retain *.ckpt after eval (needed for blind/oracle re-scoring and zero-shot)")
    parser.add_argument("--config_file", type=Path, help="Explicit training recipe YAML")
    parser.add_argument("--output_root", type=Path, help="Separate root for new experiment results")
    parser.add_argument("--stop_on_failure", action="store_true", help="Stop a sequential sweep on its first failure")
    args = parser.parse_args()
    if args.config_file is not None and not args.config_file.is_file():
        parser.error("--config_file must exist")

    csv_files    = parse_semicolon_list(args.csv_files)
    train_frames = parse_semicolon_list(args.train_frames)
    seeds        = parse_semicolon_list(args.seeds)
    backbones    = parse_semicolon_list(args.backbones)
    losses       = [l for l in args.losses_to_use.split(",") if l]
    temperatures = [t for t in args.sampling_temperatures.split(";") if t]
    head_modes   = [h for h in args.head_modes.split(";") if h]

    combos = build_combos(csv_files, backbones, train_frames, seeds, temperatures, head_modes)
    # Collected rather than raised so one bad combo doesn't abort the sweep, but the
    # process can still exit non-zero instead of reporting success after every job died.
    failures: list[str] = []
    print(f"Total jobs: {len(combos)}")

    def output_for(combo):
        output = make_output_dir(combo[0], combo[1], combo[2], combo[3], losses, combo[4], combo[5])
        return args.output_root / output.relative_to(RESULTS_DIR) if args.output_root else output

    if args.skip_existing and not args.eval_only:
        combos = [
            c for c in combos
            if not output_for(c).exists()
        ]
        print(f"After skipping existing: {len(combos)} remaining")

    for csv_file, backbone, train_frames_n, seed, temperature, head_mode in combos:
        output_dir = output_for((csv_file, backbone, train_frames_n, seed, temperature, head_mode))
        label = f"{csv_stem(csv_file)} | tf={train_frames_n} | {backbone} | seed={seed}"
        if temperature:
            label += f" | T={temperature}"
        if head_mode:
            label += f" | head={head_mode}"
        print(f"\n── {label}")

        if not args.eval_only:
            cmd = make_train_command(
                csv_file, backbone, train_frames_n, seed, losses, output_dir, args.debug,
                temperature, head_mode, config_file=args.config_file,
            )
            print("   " + " ".join(cmd))

            if not args.dry_run:
                output_dir.mkdir(parents=True, exist_ok=True)
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR: training failed (exit {e.returncode}), skipping eval...")
                    failures.append(f"{label} (training exit {e.returncode})")
                    if args.stop_on_failure:
                        break
                    continue

        if not args.dry_run:
            if not output_dir.exists():
                print(f"  WARNING: output dir not found, skipping eval: {output_dir}")
                failures.append(f"{label} (no output dir)")
                continue
            try:
                evaluate_model(output_dir, csv_file, keep_checkpoints=args.keep_checkpoints)
            except Exception as e:
                print(f"  ERROR: evaluation failed: {e}")
                failures.append(f"{label} (evaluation: {e})")
                if args.stop_on_failure:
                    break

    if args.dry_run:
        print(f"\n(dry run — {len(combos)} commands printed, nothing executed)")
        return

    print(f"\nComplete: {len(combos) - len(failures)}/{len(combos)} succeeded.")
    if failures:
        print("Failed:")
        for f in failures:
            print(f"  {f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
