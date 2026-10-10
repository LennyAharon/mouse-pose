#!/usr/bin/env python
"""
Write a live board of the models being trained: running now, queued next, grid progress, finished.

    python scripts/runs_board.py                 # write <results_dir>/RUNS.md once
    python scripts/runs_board.py --loop 180      # rewrite it every 3 min (run detached, see below)

    setsid nohup python scripts/runs_board.py --loop 180 > /dev/null 2>&1 < /dev/null & disown

<results_dir>/RUNS_NOTES.md, if present, is shown under the title (the plan / order of the queues).
Sources, all read-only: the GPU processes (nvidia-smi; each run's --output_dir and its
train_status.json for the step and the ETA), every adaptation grid with a queue
(configs/adaptation/*.yaml -> <results_dir>/<out_subdir>/<name>/_queue: queue.log, started_*
markers, the queue's last plan in pending.txt, STOP, max_gpu), and the batch queues in
--batch_logs (batch<X>.sh with a live batch<X>.pid: STOP file, batch<X>.started_<id> markers,
the job list from `DRY=1 bash batch<X>.sh`, and batch<X>.log), and the job queues of
scripts/queue_runs.sh (<results_dir>/_queues/<name>/: jobs.tsv, started_/done_ markers,
queue.log). Times are UTC.
"""

import argparse
import json
import os
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mighty_mouse.adaptation import ALL, cell_dir, expand, load_grid
from mighty_mouse.paths import load_paths, repo_root

LOG_LINE = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\] (.*)$")


def now() -> datetime:
    return datetime.now(timezone.utc)


def parse_time(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def hm(t: datetime | None) -> str:
    if t is None:
        return ""
    return t.strftime("%H:%M") if t.date() == now().date() else t.strftime("%a %H:%M")


def alive(pid_file: Path, name: str) -> bool:
    """True when the pid in ``pid_file`` is a live process whose command line contains ``name``."""
    try:
        pid = int(pid_file.read_text().strip())
        return name in (Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace"))
    except (OSError, ValueError):
        return False


def log_lines(path: Path) -> list[tuple[datetime, str]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(errors="replace").splitlines():
        m = LOG_LINE.match(line)
        if m:
            out.append((parse_time(m.group(1)), m.group(2)))
    return out


# ── running now ──────────────────────────────────────────────────────────────


def gpu_state() -> tuple[list[tuple[int, int, str]], str]:
    """(pid, MiB, GPU index) of every GPU process, and a per-GPU summary."""
    q = ["nvidia-smi", "--format=csv,noheader,nounits"]
    try:
        apps = subprocess.run(q + ["--query-compute-apps=pid,used_memory,gpu_uuid"],
                              capture_output=True, text=True, timeout=30).stdout
        gpus = subprocess.run(q + ["--query-gpu=index,uuid,utilization.gpu,memory.used,"
                                   "memory.total"], capture_output=True, text=True,
                              timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return [], "nvidia-smi unavailable"
    rows  = [[x.strip() for x in line.split(",")] for line in gpus.splitlines() if line.strip()]
    index = {r[1]: r[0] for r in rows}
    procs = []
    for line in apps.splitlines():
        if line.strip():
            pid, mib, uuid = (x.strip() for x in line.split(","))
            procs.append((int(pid), int(mib), index.get(uuid, "?")))
    per = [f"GPU {i}: {sum(p[2] == i for p in procs)} jobs, {u} %, "
           f"{int(m) // 1024}/{int(t) // 1024} GB" for i, _, u, m, t in rows]
    return procs, "; ".join(per)


def output_dir(pid: int) -> Path | None:
    try:
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return None
    if "--output_dir" in argv:
        return Path(argv[argv.index("--output_dir") + 1])
    try:
        return Path(os.readlink(f"/proc/{pid}/cwd"))
    except OSError:
        return None


def cell_label(grid: str, dataset: str, arm: str, rest: list[str]) -> str:
    """``grid · dataset · arm [· hide kps] · N=n · draw d`` from a cell's id or directory parts."""
    mask = next((r[5:] for r in rest if r.startswith("mask-")), "")
    last = rest[-1] if "-draw" in rest[-1] else f"{rest[-2]}-{rest[-1]}"
    n, draw = last.replace("__", "-").replace("tf", "", 1).split("-draw")
    return f"{grid} · {dataset} · {arm}" + (f" · hide {mask}" if mask else "") + \
        f" · N={n} · draw {draw}"


def describe(run: Path, results: Path) -> tuple[str, str]:
    """(model, kind) of a run directory."""
    try:
        rel = run.relative_to(results).parts
    except ValueError:
        return str(run), "other"
    if len(rel) >= 5 and "_queue" not in rel and (results / rel[0] / rel[1] / "_queue").exists():
        return cell_label(rel[1], rel[2], rel[3], list(rel[4:])), "grid cell"
    if rel[0] == "experiments":
        return rel[1], "experiment"
    if rel[0].startswith("dedicated"):
        head = rel[0].removeprefix("dedicated").strip("-") or "linear"
        return f"{rel[1].removesuffix('_train')} ({head})", "dedicated"
    if rel[0].startswith("trunks"):
        return f"{rel[1].removesuffix('_train')} ({rel[0]})", "trunk"
    return "/".join(rel[:4]), rel[0]


def running_rows(procs: list[tuple[int, int, str]], results: Path) -> list[list[str]]:
    rows = []
    for pid, mib, gpu in procs:
        run = output_dir(pid)
        if run is None:
            continue
        model, kind = describe(run, results)
        try:
            elapsed = int(subprocess.run(["ps", "-o", "etimes=", "-p", str(pid)],
                                         capture_output=True, text=True).stdout.strip())
        except ValueError:
            elapsed = 0
        started = now() - timedelta(seconds=elapsed)
        step, eta = "", ""
        st = run / "train_status.json"
        if st.exists():
            try:
                s = json.loads(st.read_text())
                done, total = s["progress"]["completed"], s["progress"]["total"]
                step = f"{done} / {total} ({100 * done // max(total, 1)} %)"
                if s["status"] == "COMPLETED":
                    step += ", evaluating"
                elif done:
                    eta = hm(started + timedelta(seconds=elapsed * total / done))
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
        rows.append([model, kind, step, hm(started), eta, gpu, f"{mib / 1024:.1f} GB"])
    return rows


# ── queues ───────────────────────────────────────────────────────────────────


def grid_sections(results: Path) -> tuple[list[str], list[tuple[datetime, str, str]]]:
    """Markdown lines for every adaptation grid with a queue, and its finished cells."""
    lines, finished = [], []
    for cfg in sorted((repo_root() / "configs" / "adaptation").glob("*.yaml")):
        grid = load_grid(cfg)
        q = results / grid.out_subdir / grid.name / "_queue"
        if not q.exists():
            continue
        cells   = expand(grid)
        started = {p.name.removeprefix("started_") for p in q.glob("started_*")}
        done    = {c.id for c in cells if (cell_dir(results, grid, c) / ".done").exists()}
        log     = log_lines(q / "queue.log")
        failed  = {}
        for t, msg in log:
            m = re.match(r"end (\S+) \(exit (\d+)\): ?(.*)", msg)
            if m:
                cid, rc, tail = m.groups()
                if rc != "0":
                    failed[cid] = t
                px = re.search(r"all labelled ([\d.]+) px(, hidden ([\d.]+) px)?", tail)
                res = (f"{px.group(1)} px" + (f", hidden {px.group(3)} px" if px.group(2) else "")
                       if px and rc == "0" else f"FAILED (exit {rc})")
                ds, arm, *rest = cid.split("__")
                finished.append((t, cell_label(grid.name, ds, arm, rest), res))
        failed = {c for c in failed if c not in done}
        running = started - done - failed
        state = "not running"
        if alive(q / "run_local.pid", "run_local.sh"):
            limit = (q / "max_gpu").read_text().strip() if (q / "max_gpu").exists() else "?"
            gpus  = (q / "gpus").read_text().strip() if (q / "gpus").exists() else "all"
            state = f"running, {limit} per GPU on GPUs [{gpus}]" + (
                ", HELD by STOP" if (q / "STOP").exists() else "")
        recent = [t for t, msg in log if msg.startswith("end ") and "(exit 0)" in msg
                  and t > now() - timedelta(hours=1)]
        rate = len(recent)   # cells/h over the last hour (follows machine and limit changes)
        left = len(cells) - len(done)
        eta = (f", about {left / rate:.0f} h left at {rate} cells/h (last hour; all-frames cells "
               "are ~3x longer)" if rate else "")
        lines += [f"### Grid `{grid.name}` ({state})", "",
                  f"{len(done)} / {len(cells)} cells done, {len(running)} running, "
                  f"{len(failed)} failed{eta}.", ""]
        arms = list(dict.fromkeys(c.arm + (" (masked)" if c.mask else "") for c in cells))
        rows = []
        for a in arms:
            sel = [c for c in cells if c.arm + (" (masked)" if c.mask else "") == a]
            by_n = []
            for n in dict.fromkeys(c.n for c in sel):
                k = [c for c in sel if c.n == n]
                by_n.append(f"{'all' if n == ALL else n}: {sum(c.id in done for c in k)}/{len(k)}")
            rows.append([a, f"{sum(c.id in done for c in sel)} / {len(sel)}", ", ".join(by_n)])
        lines += md(["arm", "done", "by N"], rows) + [""]
        pend = (q / "pending.txt").read_text().split() if (q / "pending.txt").exists() else []
        nxt = [c for c in pend if c not in started][:8]
        if nxt:
            lines += ["Next: " + ", ".join(f"`{c}`" for c in nxt) + " ...", ""]
        if failed:
            lines += ["Failed (see `<queue dir>/<id>.out`): " +
                      ", ".join(f"`{c}`" for c in sorted(failed)), ""]
    return lines, finished


def batch_sections(batch_logs: Path) -> tuple[list[str], list[tuple[datetime, str, str]]]:
    lines, finished = [], []
    for script in sorted(batch_logs.glob("batch*.sh")):
        name = script.stem
        for t, msg in log_lines(batch_logs / f"{name}.log"):
            m = re.match(r"end (\S+) \(exit (\d+)\)", msg)
            if m:
                finished.append((t, f"{m.group(1)} ({name})",
                                 "done" if m.group(2) == "0" else f"FAILED (exit {m.group(2)})"))
        if not alive(batch_logs / f"{name}.pid", script.name):
            continue
        dry = subprocess.run(["bash", str(script)], env={**os.environ, "DRY": "1"},
                             capture_output=True, text=True, timeout=60).stdout
        jobs = re.findall(r"^== (\S+)", dry, flags=re.M)
        pending = [j for j in jobs if not (batch_logs / f"{name}.started_{j}").exists()]
        held = " — HELD by STOP" if (batch_logs / f"{name}.STOP").exists() else ""
        lines += [f"- **{name}** (queue alive{held}): {len(jobs) - len(pending)} / {len(jobs)} "
                  f"started; pending: " + (", ".join(pending) if pending else "none")]
    return lines, finished


def job_queue_sections(results: Path) -> tuple[list[str], list[tuple[datetime, str, str]]]:
    """Queues of scripts/queue_runs.sh: <results_dir>/_queues/<name>/jobs.tsv."""
    lines, finished = [], []
    for jobs in sorted((results / "_queues").glob("*/jobs.tsv")):
        q = jobs.parent
        ids = [line.split("\t")[0] for line in jobs.read_text().splitlines() if line.strip()]
        for t, msg in log_lines(q / "queue.log"):
            m = re.match(r"end (\S+) \(exit (\d+)\)", msg)
            if m:
                finished.append((t, f"{m.group(1)} ({q.name})",
                                 "done" if m.group(2) == "0" else f"FAILED (exit {m.group(2)})"))
        done    = [i for i in ids if (q / f"done_{i}").exists()]
        started = [i for i in ids if (q / f"started_{i}").exists() and i not in done]
        pending = [i for i in ids if not (q / f"started_{i}").exists()]
        state = "alive" if alive(q / "queue.pid", "queue_runs.sh") else "not running"
        if (q / "STOP").exists():
            state += ", HELD by STOP"
        lines += [f"- **{q.name}** ({state}): {len(done)} / {len(ids)} done, "
                  f"{len(started)} running or failed; pending: "
                  + (", ".join(pending) if pending else "none")]
    return lines, finished


def md(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def board(results: Path, batch_logs: Path) -> str:
    procs, gpu = gpu_state()
    grid_lines, fin_grid = grid_sections(results)
    batch_lines, fin_batch = batch_sections(batch_logs)
    job_lines, fin_jobs = job_queue_sections(results)
    finished = sorted(fin_grid + fin_batch + fin_jobs, reverse=True)[:40]
    lines = [f"# Runs board: {results.name}", "",
             f"Updated {now():%Y-%m-%d %H:%M} UTC by `mouse-pose/scripts/runs_board.py`. "
             f"GPU: {gpu}.", ""]
    notes = results / "RUNS_NOTES.md"   # hand-written plan / order of the queues, shown as is
    if notes.exists():
        lines += [notes.read_text().strip(), ""]
    lines += ["## Running now", ""]
    rows = running_rows(procs, results)
    lines += (md(["model", "kind", "step", "started", "ETA (training)", "GPU", "mem"], rows)
              if rows else ["Nothing on the GPU."]) + [""]
    lines += ["## Queued", ""] + grid_lines
    lines += ["### Job queues", ""] + (job_lines or ["None."]) + [""]
    lines += ["### Batch queues", ""] + (batch_lines or ["No batch queue alive."]) + [""]
    lines += ["## Finished (latest 40)", ""]
    lines += md(["finished", "model", "result"], [[hm(t), m, r] for t, m, r in finished])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",        default=None, type=Path,
                    help="board file (default <results_dir>/RUNS.md)")
    ap.add_argument("--batch_logs", default=None, type=Path,
                    help="batch queue dir (default <results_dir>/experiments/_logs)")
    ap.add_argument("--loop",       default=0, type=int, help="rewrite every N seconds")
    args = ap.parse_args()

    while True:
        results = Path(load_paths()["results_dir"])   # re-read: follows a corpus version change
        out     = args.out or results / "RUNS.md"
        logs    = args.batch_logs or results / "experiments" / "_logs"
        try:
            text = board(results, logs)
        except Exception as e:  # noqa: BLE001 -- a loop must survive one bad read
            text = f"# Runs board\n\nUpdate failed at {now():%H:%M} UTC: {e!r}\n"
        tmp = out.with_suffix(".md.tmp")
        tmp.write_text(text)
        os.replace(tmp, out)
        if not args.loop:
            print(f"wrote {out}")
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
