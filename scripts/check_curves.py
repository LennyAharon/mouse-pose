#!/usr/bin/env python3
"""
Compare a training run's curves with a baseline at matching steps, and optionally stop it.

Reads TensorBoard scalars from <run>/tb_logs (a seed directory of train_sweep.py, or any litpose
output dir). Pixel metrics are compared, not losses: ablations can change what a loss contains
(the absent bin drops absent cells from the heatmap MSE), but px error means the same everywhere.

  report   train_supervised_rmse (window mean) and val_supervised_rmse (pooled and per dataset)
           of every run at the baseline's validation steps, plus each run's last point.

  --watch  poll every --every s; at every validation point from step unfreeze + --grace until the
           run ends, if its pooled val px is above --ratio x the baseline's at the same step (or
           not below --stall_frac of its value at the unfreeze step), kill that run's
           `litpose train` process and write
           <run>/STOPPED_BY_WATCHDOG.txt with the reason. Stop-and-rerun, not resume: an ablation
           that does not learn is fixed and relaunched, never continued.
           A run that is slower BY DESIGN (lower backbone lr, shorter warm-up...) must not get the
           baseline-ratio rule: pass --ratio 1000 so only the halving rule applies (2026-09-28: the
           2x rule stopped a healthy backbone-lr x0.1 run at 16 px vs 7).

    python scripts/check_curves.py --base <seed dir> --run A=<seed dir> --run B=<seed dir>
    python scripts/check_curves.py --base <seed dir> --run A=<dir> --watch --unfreeze 1000
"""

import argparse
import glob
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/teamspace/studios/this_studio/.tbdeps")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402


def load(run: Path) -> dict:
    files = sorted(glob.glob(str(run / "tb_logs" / "**" / "events*"), recursive=True))
    if not files:
        return {}
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return {t: {e.step: e.value for e in ea.Scalars(t)} for t in ea.Tags()["scalars"]}


def near(series: dict, step: int, w: int = 50) -> float:
    vals = [v for s, v in series.items() if abs(s - step) <= w]
    return float(np.mean(vals)) if vals else np.nan


def report(base: dict, runs: dict) -> None:
    steps = sorted(base.get("val_supervised_rmse", {}))
    if not steps or not any(d.get("val_supervised_rmse") for d in runs.values()):
        print("no validation points yet")
        return
    last = max((max(d.get("val_supervised_rmse", {0: 0})) for d in runs.values()), default=0)
    steps = [s for s in steps if s <= last + 1] or steps[:1]
    names = ["base"] + list(runs)
    allr = {"base": base, **runs}
    print(f"{'step':>6} | " + " | ".join(f"{n:>17}" for n in names) + "   (train px / val px)")
    for s in steps:
        cells = []
        for n in names:
            d = allr[n]
            cells.append(f"{near(d.get('train_supervised_rmse', {}), s):7.2f} / "
                         f"{d.get('val_supervised_rmse', {}).get(s, np.nan):7.2f}")
        print(f"{s:6d} | " + " | ".join(cells))
    s = steps[-1]
    ds = sorted(t.split("/", 1)[1] for t in base if t.startswith("val_supervised_rmse/"))
    print(f"\nper-dataset val px at step {s}")
    for d in ds:
        print(f"  {d:14s} " + "  ".join(
            f"{n}={allr[n].get('val_supervised_rmse/' + d, {}).get(s, np.nan):7.2f}" for n in names))


def train_pids(run: Path) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout
    return [int(line.split(None, 1)[0]) for line in out.splitlines()
            if "litpose train" in line and f"--output_dir {run}" in line]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base",       required=True, type=Path)
    ap.add_argument("--run",        action="append", required=True, help="name=seed dir")
    ap.add_argument("--watch",      action="store_true")
    ap.add_argument("--unfreeze",   type=int,   default=1000)
    ap.add_argument("--grace",      type=int,   default=500, help="steps after unfreeze before judging")
    ap.add_argument("--ratio",      type=float, default=2.0, help="stop if val px > ratio x baseline")
    ap.add_argument("--stall_frac", type=float, default=0.5, help="stop if val px > frac x its unfreeze value")
    ap.add_argument("--every",      type=int,   default=300)
    args = ap.parse_args()

    runs = {n: Path(p) for n, p in (r.split("=", 1) for r in args.run)}
    base = load(args.base)
    if not args.watch:
        report(base, {n: load(p) for n, p in runs.items()})
        return

    seen, done = {}, set()
    while len(done) < len(runs):
        for name, run in runs.items():
            if name in done:
                continue
            alive = bool(train_pids(run))
            d = load(run)
            val = d.get("val_supervised_rmse", {})
            new = [st for st in sorted(val) if st >= args.unfreeze + args.grace and st > seen.get(name, -1)]
            for step in new:
                seen[name] = step
                bval = base.get("val_supervised_rmse", {})
                ref = bval.get(step, near(bval, step, 130))
                at_unfreeze = near(val, args.unfreeze, 130)
                reasons = []
                if np.isfinite(ref) and val[step] > args.ratio * ref:
                    reasons.append(f"val px {val[step]:.1f} > {args.ratio} x baseline {ref:.1f} at step {step}")
                if np.isfinite(at_unfreeze) and val[step] > args.stall_frac * at_unfreeze:
                    reasons.append(f"val px {val[step]:.1f} not below {args.stall_frac} x its unfreeze "
                                   f"value {at_unfreeze:.1f} at step {step}")
                if reasons:
                    pids = train_pids(run)   # train_sweep sees the non-zero exit and skips eval
                    for pid in pids:
                        os.kill(pid, signal.SIGTERM)
                    msg = (f"{time.strftime('%F %T', time.gmtime())} UTC watchdog stopped {name}: "
                           + "; ".join(reasons))
                    (run / "STOPPED_BY_WATCHDOG.txt").write_text(msg + f"\nkilled pids {pids}\n")
                    print(msg, flush=True)
                    done.add(name)
                    break
                print(f"{name}: OK at step {step} (val px {val[step]:.1f}, baseline {ref:.1f})", flush=True)
            if name not in done and not alive and val:
                print(f"{name}: training process gone at step {max(val)}; watch ends", flush=True)
                done.add(name)
        if len(done) < len(runs):
            time.sleep(args.every)


if __name__ == "__main__":
    main()
