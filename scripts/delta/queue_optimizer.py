#!/usr/bin/env python
"""
Queue optimizer: start unfinished runs on whatever Delta GPU partition can take them soonest.

Runs next to scripts/delta/interactive_feeder.sh, which keeps the interactive partitions busy.
Every --interval seconds it lists the unfinished runs of the experiment definitions (scripts/delta/
exp_specs.py, in --specs order; runs already training in a live job are not listed), asks the
scheduler (sbatch --test-only, nothing submitted) when a job of 1, 2 or 4 GPUs would start on each
regular / preempt partition, and when the best option starts within --lead minutes it submits one
pack job there (scripts/delta/pack_job.sh, one run per GPU, 16 CPUs and 60G per GPU) with the next
runs that no job of this optimizer already holds. Runs are resumable and claimed by the first job
that starts them (train_plan_job.sbatch), so a run also waiting in an interactive job never trains
twice: the later job's copy exits at once. It never cancels a job. At most --max_pending of its
own jobs wait at a time.

Every decision and the scheduler's estimates go to <results_dir>/_logs/queue_optimizer.log, the
last cycle to queue_optimizer_state.json (read them to tune --lead / shapes). For definitions with
`drop_checkpoints: true` it deletes a run's checkpoints once it is fully evaluated. Stops when no
run is left and none of its jobs is pending or running.

    cd mouse-pose && setsid nohup python scripts/delta/queue_optimizer.py \\
        --specs configs/experiments/v11-zoom-loo.yaml configs/experiments/v11-scaling.yaml \\
        > /dev/null 2>&1 < /dev/null & disown
"""

import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

import yaml

from mighty_mouse.paths import load_paths
from mighty_mouse.registry import load_registry

REPO       = Path(__file__).resolve().parents[2]
PARTITIONS = ["gpuA100x4", "gpuA100x8", "gpuA40x4", "gpuH200x8", "gpuA100x4-preempt",
              "gpuA40x4-preempt"]
GPUS       = [1, 2, 4]
TIMES      = ["01:30:00", "02:30:00"]   # runs are resumable; shorter limits backfill sooner
NAME       = "mm-opt"


def sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def est_start(part: str, gpus: int, limit: str) -> float | None:
    """Estimated start (epoch s) of a job of this shape, or None if it cannot be placed."""
    out = subprocess.run(["sbatch", "--test-only", "-A", "beez-delta-gpu", "-p", part,
                          f"--gpus-per-node={gpus}", f"--cpus-per-task={16 * gpus}",
                          f"--mem={60 * gpus}G", "-t", limit, "--wrap", "true"],
                         capture_output=True, text=True)
    for word in (out.stdout + out.stderr).split():
        if word.startswith("20") and "T" in word:
            try:
                return datetime.fromisoformat(word).timestamp()
            except ValueError:
                pass
    return None


def unfinished(specs: list[str]) -> list[str]:
    """Spec lines (exp_specs.py) of runs not started or resumable, in --specs order."""
    lines = []
    for d in specs:
        out = subprocess.run(["python", "scripts/delta/exp_specs.py", d], cwd=REPO,
                             capture_output=True, text=True).stdout
        lines += [ln for ln in out.splitlines() if ln.strip()]
    return lines


def my_jobs() -> list[dict]:
    """This optimizer's jobs: id, state, run names (job comment)."""
    out = sh("squeue", "-u", load_user(), "-h", "-n", ",".join(f"{NAME}{g}" for g in GPUS),
             "-o", "%i|%t|%k")
    return [dict(zip(("id", "state", "runs"), ln.split("|"), strict=True))
            for ln in out.splitlines() if ln]


def drop_checkpoints(specs: list[str], results: Path) -> list[Path]:
    """For definitions with `drop_checkpoints: true`: delete the checkpoints (and rolling resume
    checkpoint) of every finished run, i.e. training COMPLETED, test-set evaluation written and a
    zero-shot prediction file for each of its `zeroshot` datasets. Returns the runs cleaned."""
    short = dict(re.findall(r"\[([\w-]+)\]=(\w+)", (REPO / "scripts/train_plan.sh").read_text()))
    reg, cleaned = load_registry(), []
    for d in specs:
        spec = yaml.safe_load((REPO / d).read_text())
        if not spec.get("drop_checkpoints"):
            continue
        for exp_id, exp in spec["experiments"].items():
            for entry in spec["targets"].values():
                if not isinstance(entry, dict):
                    continue
                tag = "+".join(short[x] for x in reg if x in entry["keep"])
                for seed in exp.get("seeds", [0]):
                    run = (results / "experiments" / exp_id / f"{tag}_train" / "supervised"
                           / "sampling-T2" / "tf1" / spec["backbone"] / f"seed{seed}")
                    status = run / "train_status.json"
                    done = (status.exists() and "COMPLETED" in status.read_text()
                            and any(run.glob("eval/*/predictions.csv"))
                            and all((run / "zeroshot" / f"{z}_train_predictions.csv").exists()
                                    for z in entry.get("zeroshot", [])))
                    ckpts = list(run.glob("tb_logs/*/version_*/checkpoints/*.ckpt"))
                    if done and (ckpts or (run / "resume").exists()):
                        for c in ckpts:
                            c.unlink()
                        shutil.rmtree(run / "resume", ignore_errors=True)
                        cleaned.append(run)
    return cleaned


def load_user() -> str:
    return sh("whoami").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs",       nargs="+", required=True, help="experiment definitions")
    ap.add_argument("--interval",    type=int, default=300)
    ap.add_argument("--lead",        type=int, default=20, help="submit if it starts within N min")
    ap.add_argument("--max_pending", type=int, default=2)
    ap.add_argument("--run_min",     type=int, default=110, help="one pack's run time (score)")
    ap.add_argument("--once",        action="store_true")
    args = ap.parse_args()

    results = Path(load_paths()["results_dir"])
    logs    = results / "_logs"
    packs   = logs / "packs"
    log     = open(logs / "queue_optimizer.log", "a", buffering=1)
    say     = lambda m: log.write(f"[{datetime.now():%m-%d %H:%M:%S}] {m}\n")   # noqa: E731
    say(f"start: specs {args.specs}, lead {args.lead} min, max_pending {args.max_pending}")
    while True:
        for run in drop_checkpoints(args.specs, results):
            say(f"checkpoints deleted (drop_checkpoints): {run}")
        now   = time.time()
        jobs  = my_jobs()
        held  = {r for j in jobs for r in j["runs"].split(",") if r}
        lines = unfinished(args.specs)
        free  = [ln for ln in lines if ln.split("|")[0] not in held]
        n_pd  = sum(j["state"] == "PD" for j in jobs)
        options = []
        for part in PARTITIONS:
            for g in GPUS:
                for limit in TIMES:
                    t = est_start(part, g, limit)
                    if t is not None:
                        wait = max(0.0, (t - now) / 60)
                        options.append(dict(partition=part, gpus=g, time=limit,
                                            wait_min=round(wait), score=g / (wait + args.run_min)))
        soon = [o for o in options if o["wait_min"] <= args.lead and o["gpus"] <= len(free)]
        options.sort(key=lambda o: o["wait_min"])
        state = dict(time=datetime.now().isoformat(timespec="seconds"), unfinished=len(lines),
                     free=len(free), my_jobs=jobs, options=options[:12])
        if soon and free and n_pd < args.max_pending:
            best = max(soon, key=lambda o: (o["score"], o["time"]))
            pick = free[:best["gpus"]]
            spec = packs / f"{datetime.now():%Y%m%d-%H%M%S}_{best['partition']}_opt.txt"
            spec.write_text("\n".join(pick) + "\n")
            names = ",".join(ln.split("|")[0] for ln in pick)
            jid = sh("sbatch", "--parsable", "-A", "beez-delta-gpu", "-p", best["partition"],
                     "-t", best["time"], "-J", f"{NAME}{best['gpus']}", "--comment", names,
                     "--nodes=1", "--ntasks=1", f"--gpus-per-node={best['gpus']}",
                     f"--cpus-per-task={16 * best['gpus']}", f"--mem={60 * best['gpus']}G",
                     "-o", str(logs / "slurm" / "%x_%j.out"),
                     str(REPO / "scripts/delta/pack_job.sh"), str(spec)).strip()
            say(f"SUBMIT {jid} {best['partition']} {best['gpus']} GPU {best['time']} "
                f"(est. wait {best['wait_min']} min): {names}")
            state["submitted"] = dict(job=jid, **best, runs=names)
        else:
            top = state["options"][:3]
            say(f"no submit: {len(free)} free runs, {n_pd} pending of mine; best "
                + "; ".join(f"{o['partition']} {o['gpus']}G {o['time']} {o['wait_min']}m"
                            for o in top))
        (logs / "queue_optimizer_state.json").write_text(json.dumps(state, indent=1) + "\n")
        if args.once or (not lines and not jobs):
            say("stop" if not args.once else "once: done")
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
