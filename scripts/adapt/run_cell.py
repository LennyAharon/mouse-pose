#!/usr/bin/env python
"""
Run one adaptation cell end to end: train, check, evaluate on the target dataset, record.

    python scripts/adapt/run_cell.py --config configs/adaptation/cosyne-2026.yaml \
        --cell ibl__mm-anchored-lora__tf10__draw0
    python scripts/adapt/run_cell.py --config ... --jobs <file> --index 7   # 0-based line (SLURM)
    python scripts/adapt/run_cell.py --config ... --cell <id> --dry_run     # print the command

Steps: (masked cells) write the masked training csv into data_dir if missing; resolve the arm
(trunk checkpoint, the trunk's head rebuilt from its config.yaml, anchored keypoints = everything
the trunk trained; an experts trunk with model.head_groups needs PYTHONPATH pointing at a
Lightning Pose that builds them, else ABORT); `litpose train configs/model.yaml` with the cell's
overrides; check the log (weights loaded, anchor attached, LoRA rank, head); evaluate on the
target dataset's test set only (`python -m mighty_mouse.train --datasets <ds>`); for LoRA arms
check that the reloaded model carries the trained adapters; delete checkpoints unless the grid
keeps them; write run_info.json and `.done`. Idempotent: a cell with `.done` is skipped; a
partial directory is moved aside. Exit code != 0 on any failure.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from mighty_mouse.adaptation import (
    INIT_TRUNK,
    best_checkpoint,
    build_overrides,
    cell_dir,
    expand,
    hide_keypoints,
    load_grid,
    log_checks,
    masked_csv_name,
    pixel_errors,
    summarize,
    trained_keypoints,
    trunk_datasets,
    trunk_head_overrides,
    trunk_problem,
)
from mighty_mouse.paths import load_paths, repo_root

LORA_CHECK = """
import sys, torch
from lightning_pose.api.model import Model
from lightning_pose.models.backbones.lora import LoRALinear
from lightning_pose.utils.io import ckpt_path_from_base_path
d = sys.argv[1]; mdl = Model.from_dir(d); mdl._load(); m = mdl.model
loras = [(n, l) for n, l in m.backbone.named_modules() if isinstance(l, LoRALinear)]
ck = ckpt_path_from_base_path(d, model_name="test")
sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
ok = len(loras) > 0 and all(
    torch.equal(l.lora_B.detach().cpu(), sd["backbone." + n + ".lora_B"]) for n, l in loras)
print("LoRA reload check:", "OK" if ok else "FAILED", f"({len(loras)} layers)")
sys.exit(0 if ok else 1)
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_head(path: Path) -> str:
    r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True)
    return r.stdout.strip() or "unknown"


def ensure_masked_csv(data_dir: Path, dataset: str, mask: tuple[str, ...]) -> str:
    """Write the masked training csv (atomically: concurrent cells cannot clash); its name."""
    name = masked_csv_name(dataset, mask)
    dst  = data_dir / name
    if not dst.exists():
        src = data_dir / f"CollectedData_{dataset}_train.csv"
        out, n_hidden = hide_keypoints(pd.read_csv(src, header=[0, 1, 2], index_col=0), list(mask))
        fd, tmp = tempfile.mkstemp(dir=data_dir, suffix=".csv.tmp")
        os.close(fd)
        out.to_csv(tmp)
        os.replace(tmp, dst)
        print(f"wrote {name}: hid {n_hidden} labels of {list(mask)} over {len(out)} rows")
    return name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config",  required=True, type=Path)
    ap.add_argument("--cell",    default=None, help="cell id (see plan.py)")
    ap.add_argument("--jobs",    default=None, type=Path, help="job file; --index picks the cell")
    ap.add_argument("--index",   default=None, type=int,
                    help="0-based line of --jobs (e.g. $SLURM_ARRAY_TASK_ID)")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    if args.cell is None:
        if args.jobs is None or args.index is None:
            sys.exit("give --cell, or --jobs and --index")
        args.cell = args.jobs.read_text().splitlines()[args.index].strip()

    grid    = load_grid(args.config)
    paths   = load_paths()
    data    = Path(paths["data_dir"])
    results = Path(paths["results_dir"])
    inv     = json.loads((data / "dataset_inventory.json").read_text())["datasets"]
    cells   = {c.id: c for c in expand(grid)}
    if args.cell not in cells:
        sys.exit(f"unknown cell {args.cell}")
    cell = cells[args.cell]
    arm  = grid.arms[cell.arm]
    out  = cell_dir(results, grid, cell)
    if (out / ".done").exists():
        print(f"skip (done): {out}")
        return

    # ── inputs ─────────────────────────────────────────────────────────────────
    if not cell.mask:
        train_csv = f"CollectedData_{cell.dataset}_train.csv"
    elif args.dry_run:
        train_csv = masked_csv_name(cell.dataset, cell.mask)
    else:
        train_csv = ensure_masked_csv(data, cell.dataset, cell.mask)
    trunk, trunk_dir = None, results / grid.datasets[cell.dataset]["trunk"]
    if arm["init"] == INIT_TRUNK:
        why = trunk_problem(trunk_dir)
        if why:
            sys.exit(f"ABORT: trunk {trunk_dir} not ready: {why}")
        ckpt  = best_checkpoint(trunk_dir)
        tcfg  = yaml.safe_load((trunk_dir / "config.yaml").read_text())
        trunk = {"checkpoint": str(ckpt), "backbone": tcfg["model"]["backbone"],
                 "head_overrides": trunk_head_overrides(tcfg),
                 "keypoints": trained_keypoints(trunk_datasets(trunk_dir, tcfg), inv)}
        if any("head_groups" in o for o in trunk["head_overrides"]):   # an experts trunk
            probe = "from lightning_pose.models.factory import resolve_head_groups"
            if subprocess.run([sys.executable, "-c", probe], capture_output=True).returncode:
                sys.exit("ABORT: the trunk has model.head_groups but this lightning_pose cannot "
                         "build them (it would load the grouped weights into a plain head); run "
                         "with PYTHONPATH=<lightning-pose-wt-head-groups-fanin>")
    overrides = build_overrides(grid, cell, data, train_csv, trunk)
    cmd = ["litpose", "train", "configs/model.yaml", "--output_dir", str(out),
           "--overrides", *overrides]
    if args.dry_run:
        print(" ".join(cmd))
        return

    # ── train ──────────────────────────────────────────────────────────────────
    os.chdir(repo_root())
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        stamp = datetime.now(timezone.utc).strftime("%m%d%H%M%S")
        out.rename(out.with_name(f"{out.name}.partial-{stamp}"))
    log = out.with_name(out.name + ".log")
    started = now()
    print(f"=== {cell.id}: training -> {out}")
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
    problems = log_checks(arm, log.read_text(errors="replace"))
    if rc != 0:
        problems.insert(0, f"litpose exit {rc}")
    status = out / "train_status.json"
    if not status.exists() or "COMPLETED" not in status.read_text():
        problems.append("training not COMPLETED")
    if trunk and any("head_groups" in o for o in trunk["head_overrides"]):
        if "head_groups" not in (out / "config.yaml").read_text():
            problems.append("the run's config lost the trunk's head groups")
    if problems:
        sys.exit(f"ABORT {cell.id}: {'; '.join(problems)} (log {log})")

    # ── evaluate on the target dataset, check adapters, clean up ───────────────
    with open(out.with_name(out.name + "-eval.log"), "w") as fh:
        ev = subprocess.run([sys.executable, "-m", "mighty_mouse.train", "--output_dir", str(out),
                             "--csv_file", train_csv, "--datasets", cell.dataset,
                             "--keep_checkpoints"],
                            stdout=fh, stderr=subprocess.STDOUT).returncode
    pred = out / "eval" / cell.dataset / "predictions.csv"
    if ev != 0 or not pred.exists():
        sys.exit(f"ABORT {cell.id}: evaluation failed (exit {ev})")
    if arm.get("adapter") == "lora":
        chk = subprocess.run([sys.executable, "-c", LORA_CHECK, str(out)],
                             capture_output=True, text=True)
        print(chk.stdout.strip())
        if chk.returncode != 0:
            sys.exit(f"ABORT {cell.id}: reloaded model lacks the trained LoRA adapters")
    if not grid.keep_checkpoints:
        for ck in out.rglob("*.ckpt"):
            ck.unlink()

    # ── record ─────────────────────────────────────────────────────────────────
    import lightning_pose
    lp_dir   = Path(lightning_pose.__file__).parent
    errs     = pixel_errors(pred, data / f"CollectedData_{cell.dataset}_test.csv")
    labelled = sorted(inv[cell.dataset]["direct"])
    result   = {"all_labelled": summarize(errs, labelled)}
    if cell.mask:
        result["hidden"] = summarize(errs, list(cell.mask))
    info = {
        "cell": cell.id, "grid": grid.name, "config": str(args.config),
        "dataset": cell.dataset, "arm": cell.arm, "arm_settings": arm, "n_frames": cell.n,
        "draw": cell.draw, "mask": list(cell.mask), "kind": cell.kind, "train_csv": train_csv,
        "trunk": str(trunk_dir) if trunk else None,
        "trunk_checkpoint": trunk["checkpoint"] if trunk else None, "overrides": overrides,
        "code": {"mouse_pose": git_head(repo_root()), "lightning_pose": git_head(lp_dir),
                 "lightning_pose_path": str(lp_dir)},
        "started_utc": started, "finished_utc": now(), "result": result,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2) + "\n")
    (out / ".done").touch()
    msg = f"DONE {cell.id}: all labelled {result['all_labelled']['mean_px']:.2f} px"
    if cell.mask:
        msg += f", hidden {result['hidden']['mean_px']:.2f} px"
    print(msg)


if __name__ == "__main__":
    main()
