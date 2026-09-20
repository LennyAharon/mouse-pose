#!/usr/bin/env python
"""
Enumerate every model of an experiment round and dispatch them as independent cluster jobs.

Builds the list of cells (one line per model: config, training CSV, seed, output root, extras),
drops cells that already have results, writes the list to <results_dir>/_cluster/jobs_<stamp>.tsv,
and submits ONE SLURM job array over that file (scripts/cluster/fit_one.slurm reads its line by
$SLURM_ARRAY_TASK_ID and calls fit_one.py). Nothing runs on the submitting node.

    python scripts/cluster/orchestrate.py plan   [stages...] [--seeds "0;1;2"]      # print the cells
    python scripts/cluster/orchestrate.py submit [stages...] [--seeds "0;1;2"] [--max_concurrent 16]
    python scripts/cluster/orchestrate.py status                                   # squeue + finished cells
    python scripts/cluster/orchestrate.py collect                                  # eval tables of everything finished

Stages (default: all dedicated loo):
    all             one all-dataset trunk (Mighty Mouse) per seed
    dedicated       one single-dataset model per dataset per seed
    loo             one leave-one-out trunk per dataset per seed
    ablation:<name> configs/ablations/model_zoominout_<name>.yaml on the all-data CSV (+ loo CSVs with --ablation_loo)

Recipe: configs/model_zoominout.yaml, T=2, shared head, ViT-S DINOv3, all frames, checkpoints kept.
Without sbatch on PATH, `submit` prints the array command instead of running it.
"""

import argparse
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from mouse_pose.paths import load_paths
from mouse_pose.registry import load_registry

REPO    = Path(__file__).resolve().parents[2]
RECIPE  = "configs/model_zoominout.yaml"
TAGS    = {"facemap": "face", "ibl": "ibl", "cheese-2d": "cheese", "cazettes-side": "caz", "kondo": "kondo", "hantman-mv": "hmv"}
LEAF    = "supervised/sampling-T2/tf1/vits_dinov3"


def cells(stages: list[str], seeds: list[str], results: Path, ablation_loo: bool) -> list[dict]:
    ds = load_registry(); tag = lambda names: "+".join(TAGS[d] for d in names)
    out = []
    def add(stage, config, names, root, seed, area):
        csv = f"CollectedData_{tag(names) if len(names) > 1 else names[0]}_train.csv"
        run = results / area / f"{tag(names) if len(names) > 1 else names[0]}_train" / LEAF / f"seed{seed}"
        out.append(dict(stage=stage, config=config, csv_file=csv, seed=seed, output_root=str(results / area), run_dir=run))
    for s in seeds:
        for st in stages:
            if st == "all":
                add(st, RECIPE, list(ds), None, s, "trunks")
            elif st == "dedicated":
                for d in ds: add(st, RECIPE, [d], None, s, "dedicated")
            elif st == "loo":
                for d in ds: add(st, RECIPE, [x for x in ds if x != d], None, s, "trunks")
            elif st.startswith("ablation:"):
                name = st.split(":", 1)[1]; cfg = f"configs/ablations/model_zoominout_{name}.yaml"
                add(st, cfg, list(ds), None, s, f"ablations/{name}")
                if ablation_loo:
                    for d in ds: add(st, cfg, [x for x in ds if x != d], None, s, f"ablations/{name}")
            else:
                raise SystemExit(f"unknown stage {st}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "submit", "status", "collect"])
    ap.add_argument("stages", nargs="*", default=["all", "dedicated", "loo"])
    ap.add_argument("--seeds",          default="0")
    ap.add_argument("--max_concurrent", type=int, default=16, help="SLURM array throttle (%%N)")
    ap.add_argument("--ablation_loo",   action="store_true", help="ablation stages also train the leave-one-out trunks")
    ap.add_argument("--partition",      default=None)
    ap.add_argument("--account",        default=None)
    ap.add_argument("--time",           default="04:00:00", help="per-job wall time limit")
    args = ap.parse_args()

    P = load_paths(); results = Path(P["results_dir"])
    seeds = [s for s in args.seeds.split(";") if s]
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")

    if args.command == "status":
        if shutil.which("squeue"): subprocess.run(["squeue", "-u", subprocess.getoutput("whoami"), "-o", "%.10i %.9P %.30j %.8T %.10M %R"])
        done = sorted(p.parent.relative_to(results) for p in results.glob("**/eval") if (p / "ibl").is_dir())
        print(f"{len(done)} finished cells under {results}:"); [print("  ", d) for d in done]; return
    if args.command == "collect":
        subprocess.run([sys.executable, str(REPO / "scripts/eval_suite.py"), "--results_dir", str(results)], check=True); return

    todo = [c for c in cells(args.stages, seeds, results, args.ablation_loo) if not (c["run_dir"] / "eval").is_dir()]
    allc = cells(args.stages, seeds, results, args.ablation_loo)
    print(f"{len(allc)} cells, {len(allc) - len(todo)} already have results, {len(todo)} to run")
    for c in todo: print(f"  {c['stage']:12s} seed {c['seed']}  {c['csv_file']:55s} -> {c['output_root']}")
    if args.command == "plan" or not todo:
        return

    jobs_dir = results / "_cluster"; jobs_dir.mkdir(parents=True, exist_ok=True)
    jobs = jobs_dir / f"jobs_{stamp}.tsv"
    with open(jobs, "w") as f:
        for c in todo:
            f.write("\t".join([c["config"], c["csv_file"], str(c["seed"]), c["output_root"]]) + "\n")
    logs = jobs_dir / "logs"; logs.mkdir(exist_ok=True)
    cmd = ["sbatch", f"--array=0-{len(todo) - 1}%{args.max_concurrent}", f"--time={args.time}",
           f"--output={logs}/%A_%a.out", f"--error={logs}/%A_%a.err", "--job-name=mightymouse",
           f"--export=ALL,JOBS_FILE={jobs},REPO={REPO}"]
    if args.partition: cmd.append(f"--partition={args.partition}")
    if args.account: cmd.append(f"--account={args.account}")
    cmd.append(str(REPO / "scripts/cluster/fit_one.slurm"))
    print(f"\njobs file: {jobs}\n" + " ".join(cmd))
    if shutil.which("sbatch"):
        subprocess.run(cmd, check=True)
    else:
        print("(sbatch not on PATH: printed the submission command only)")


if __name__ == "__main__":
    main()
