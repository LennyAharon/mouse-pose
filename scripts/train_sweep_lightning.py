#!/usr/bin/env python3
"""
Cartesian training sweep over head-fixed datasets using the litpose CLI —
Lightning AI, parallel.

Shares all combo/naming/command-building logic with train_sweep.py via
mouse_pose.train — the CLI args are identical. The only real difference:
each combo is launched as an independent Lightning Job instead of run in a
sequential loop, and since a Job is a fresh remote process (no "come back to
this process after training" step), data extraction, training, and evaluation
are chained into one shell command per job:

    test -d <data_dir> || tar -xf <data_dir>.tar -C <data_dir's parent>
    && mkdir -p <data_dir>/videos
    && litpose train ...
    && python -m mouse_pose.train --output_dir ... --csv_file ...

Each Job is an isolated snapshot of the launching Studio's own filesystem, not
a shared mount multiple jobs write into — so every job independently checking
"does this exist yet, if not extract it" is safe (see make_extract_command's
docstring in mouse_pose/train.py). This is a different situation from a
HuggingFace-style dataset cache on genuinely shared storage, which would need
locking or pre-extraction to avoid concurrent-write races.

Output lands at the same place as the local script:
  <results_dir>/<tag>/<losses_tag>/tf<N>/<backbone>/seed<N>/
  └── eval/<dataset_name>/pixel_error.csv   ← per-model evaluation results

Prerequisites (not handled by this script):
  - This machine's paths.yaml must have data_dir pointing at wherever the
    Studio keeps a `<data_dir>.tar` archive of data/head-fixed_vN (e.g. built
    with `tar -cf head-fixed_v2.tar -C data head-fixed_v2` and uploaded to the
    Studio once) — creating and uploading that archive isn't handled here.
  - results_dir should point at persistent storage that outlives an individual
    job's isolated filesystem (e.g. a teamspace-mounted drive), since results
    need to survive after the job's compute is torn down.
  - The Studio (or whatever environment `lightning_sdk` launches jobs into)
    needs `litpose` on PATH and `mouse_pose` installed (`pip install -e .`).

Run from within a Lightning AI studio:
    python scripts/train_sweep_lightning.py \\
        --csv_files "CollectedData_facemap-600_train.csv;CollectedData_face+ibl+cheese_train.csv" \\
        --train_frames "200;400;600" \\
        --seeds "0;1;2" \\
        --backbones "vits_dino" \\
        --machine L4

Run from outside Lightning AI (set LIGHTNING_API_KEY env var first):
    LIGHTNING_API_KEY=<key> python scripts/train_sweep_lightning.py ...

  # dry run to preview all jobs without launching anything
  python scripts/train_sweep_lightning.py --dry_run ...
"""

import argparse
import time

from mouse_pose.train import (
    build_combos,
    make_eval_command,
    make_extract_command,
    make_job_name,
    make_output_dir,
    make_preflight_command,
    make_publish_command,
    make_train_command,
    parse_semicolon_list,
)


def main():
    parser = argparse.ArgumentParser(
        description="Head-fixed LP training sweep (Lightning AI, parallel)",
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
    parser.add_argument("--debug",         action="store_true",          help="Smoke-test run (3 epochs)")
    parser.add_argument("--dry_run",       action="store_true",          help="Print jobs without launching")
    parser.add_argument("--skip_existing", action="store_true",          help="Skip combos whose output dir already exists")
    parser.add_argument("--allow_stock_lp", action="store_true",        help="Allow released lightning-pose, not the local clone")
    parser.add_argument("--machine",       default="T4_SMALL",           help="Lightning Machine type, e.g. T4_SMALL, A10G, L4")
    parser.add_argument(
        "--publish_dir", default=None,
        help="Teamspace folder to copy each finished run into, e.g. "
             "/teamspace/gcs_folders/head-fixed-nips26. Required in practice: a Job's "
             "filesystem is torn down when it ends, so without this the results are lost. "
             "Copied only after evaluation succeeds, so failed jobs publish nothing.",
    )
    parser.add_argument(
        "--single_job", action="store_true",
        help="Run every combo sequentially inside ONE job instead of one job per combo. "
             "Snapshot, preflight and extract are then paid once rather than per combo. "
             "Remember to raise --max_runtime to cover the whole sequence.",
    )
    parser.add_argument(
        "--max_runtime", type=int, default=None,
        help="Seconds of machine time to allocate per job. Lightning defaults to 3h, which "
             "is not enough for a multi-combo --single_job run — the machine is reclaimed "
             "mid-training and the remaining combos never run.",
    )
    parser.add_argument("--poll_interval", type=int, default=30,         help="Seconds between job-status polls")
    args = parser.parse_args()

    csv_files    = parse_semicolon_list(args.csv_files)
    train_frames = parse_semicolon_list(args.train_frames)
    seeds        = parse_semicolon_list(args.seeds)
    backbones    = parse_semicolon_list(args.backbones)
    losses       = [l for l in args.losses_to_use.split(",") if l]

    combos = build_combos(csv_files, backbones, train_frames, seeds)
    print(f"Total jobs: {len(combos)}")

    if args.skip_existing:
        combos = [c for c in combos if not make_output_dir(*c, losses).exists()]
        print(f"After skipping existing: {len(combos)} remaining")

    # Identical for every job (depend only on data_dir/env, not the combo) — computed once,
    # but still have to run inside each job since they act on that job's own filesystem.
    # Preflight goes first so a wrong lightning_pose kills the job in a second rather than
    # after a full training run (see make_preflight_command).
    preflight_cmd = make_preflight_command(args.allow_stock_lp)
    extract_cmd   = make_extract_command()

    def combo_block(combo) -> tuple[str, "Path", str]:
        """(job name, output dir, `train && eval && publish` chain) for one combo.

        Internally `&&`: evaluation only runs on a trained model, and publishing only
        runs on an evaluated one, so a combo that dies partway leaves nothing behind on
        shared storage.
        """
        csv_file, backbone, train_frames_n, seed = combo
        output_dir = make_output_dir(csv_file, backbone, train_frames_n, seed, losses)
        name       = make_job_name(csv_file, backbone, train_frames_n, seed, losses)
        stages     = [
            " ".join(make_train_command(
                csv_file, backbone, train_frames_n, seed, losses, output_dir, args.debug,
            )),
            " ".join(make_eval_command(output_dir, csv_file)),
        ]
        if args.publish_dir:
            stages.append(make_publish_command(output_dir, args.publish_dir))
        return name, output_dir, " && ".join(stages)

    jobs_spec = []
    if args.single_job:
        # Combos are separated by `;`, not `&&`, so one bad seed doesn't cancel the rest of
        # the sequence. Which combos actually succeeded is then read off the publish
        # destination rather than the job's exit status: a combo only publishes if its own
        # train+eval chain completed, so a missing directory *is* the failure report.
        # Parenthesized as a group so preflight/extract still gate the whole sequence —
        # bare, `A && B && C ; D` would run D even when the preflight killed the job.
        blocks   = [combo_block(c) for c in combos]
        body     = " ; ".join(f"( {cmd} )" for _n, _o, cmd in blocks)
        name     = f"{blocks[0][0]}__x{len(blocks)}" if blocks else "empty"
        full_cmd = " && ".join([preflight_cmd, extract_cmd, f"( {body} )"])
        jobs_spec.append((name, [o for _n, o, _c in blocks], full_cmd))
    else:
        for combo in combos:
            name, output_dir, chain = combo_block(combo)
            full_cmd = " && ".join([preflight_cmd, extract_cmd, chain])
            jobs_spec.append((name, [output_dir], full_cmd))

    if args.dry_run:
        print("\n--- Job list ---")
        for name, _output_dirs, cmd in jobs_spec:
            print(f"\n{name}:\n  {cmd}")
        print(f"\n(dry run — {len(jobs_spec)} jobs printed, nothing launched)")
        return

    from lightning_sdk import Job, Machine, Studio

    machine = getattr(Machine, args.machine)
    studio  = Studio()
    run_kw  = {"max_runtime": args.max_runtime} if args.max_runtime else {}

    jobs = {}
    for name, output_dirs, cmd in jobs_spec:
        # Only pre-create local output dirs when results land locally. With --publish_dir
        # the job writes to its own ephemeral filesystem and publishes elsewhere, so these
        # would just be empty directories in the local results tree — which a later local
        # --skip_existing would read as "already done" and skip.
        if not args.publish_dir:
            for output_dir in output_dirs:
                output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Launching: {name}")
        job = Job.run(command=cmd, name=name, machine=machine, studio=studio, **run_kw)
        jobs[name] = job
        time.sleep(2)

    print(f"\nMonitoring {len(jobs)} jobs...")
    while True:
        statuses: dict[str, int] = {}
        for j in jobs.values():
            s = str(j.status)
            statuses[s] = statuses.get(s, 0) + 1
        print(f"  {statuses}")
        if not any(s in ("Running", "Pending") for s in statuses):
            break
        time.sleep(args.poll_interval)

    failed = [n for n, j in jobs.items() if str(j.status) == "Failed"]
    print(f"\nComplete: {len(jobs) - len(failed)}/{len(jobs)} succeeded.")
    if failed:
        print("Failed jobs:")
        for n in failed:
            print(f"  {n}")


if __name__ == "__main__":
    main()
