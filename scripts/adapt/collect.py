#!/usr/bin/env python
"""
Collect an adaptation grid into one tidy table (finished cells + zero-shot + dedicated rows).

    python scripts/adapt/collect.py --config configs/adaptation/cosyne-2026.yaml

Reads saved test predictions only (no inference). For each target dataset, keypoint sets are
scored on its test frames (visible == 2 labels): `all` = every keypoint the dataset labels,
`supported` = those its leave-one-out trunk trained (the set on which zero-shot is meaningful),
`new` = the rest, `hidden` = a masked cell's hidden keypoints. Rows besides the cells:
`zero-shot` (the trunk, N = 0) and `dedicated-<head>` (N = all), and for masked settings the
`labelled` reference (the anchored-LoRA curve cell at the same N and draw, scored on the hidden
keypoints).

Writes <results_dir>/<out_subdir>/<name>/summary/cells.csv (one row per model x keypoint set x
checkpoint: `best` = best validation loss, `last` = final step when the grid has eval_last; empty
for the references) and table.md (mean +- sd over draws of the pooled mean px).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mighty_mouse.adaptation import (
    ALL,
    Cell,
    cell_dir,
    expand,
    keypoint_sets,
    load_grid,
    pixel_errors,
    summarize,
    trained_keypoints,
    trunk_datasets,
)
from mighty_mouse.paths import load_paths

ANCHORED = "mm-anchored-lora"   # the curve arm whose cells are the masked `labelled` reference
CKPTS    = (("eval", "best"), ("eval_last", "last"))   # evaluation folder -> checkpoint label


def md_table(piv: pd.DataFrame) -> str:
    """Markdown table of a pivot (multi-index rows allowed), without the tabulate dependency."""
    names = list(piv.index.names) if piv.index.nlevels > 1 else [piv.index.name or ""]
    cols  = [str(c) for c in piv.columns]
    out   = ["| " + " | ".join(names + cols) + " |", "|" + "---|" * (len(names) + len(cols))]
    for idx, row in piv.iterrows():
        keys = list(idx) if isinstance(idx, tuple) else [idx]
        vals = ["" if pd.isna(v) else str(v) for v in row]
        out.append("| " + " | ".join([str(k) for k in keys] + vals) + " |")
    return "\n".join(out)


def mean_sd(r: pd.Series) -> str:
    return f"{r['mean']:.2f}" + ("" if np.isnan(r["std"]) else f" +- {r['std']:.2f}")


def score(pred: Path, label_csv: Path, sets: dict) -> list[dict]:
    errs = pixel_errors(pred, label_csv)
    return [{"keypoints": name, **summarize(errs, kps)} for name, kps in sets.items()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    grid    = load_grid(args.config)
    paths   = load_paths()
    data    = Path(paths["data_dir"])
    results = Path(paths["results_dir"])
    inv     = json.loads((data / "dataset_inventory.json").read_text())["datasets"]
    out     = results / grid.out_subdir / grid.name / "summary"
    out.mkdir(parents=True, exist_ok=True)

    def test_csv(ds: str) -> Path:
        return data / f"CollectedData_{ds}_test.csv"

    def eval_pred(run: Path, ds: str, sub: str = "eval") -> Path:
        return run / sub / ds / "predictions.csv"

    trunk_kps = {}
    for ds, spec in grid.datasets.items():
        tdir = results / spec["trunk"]
        if (tdir / "config.yaml").exists():
            tcfg = yaml.safe_load((tdir / "config.yaml").read_text())
            trunk_kps[ds] = trained_keypoints(trunk_datasets(tdir, tcfg), inv)

    rows = []
    # ── references: zero-shot trunk (N = 0) and dedicated models (N = all) ──────
    for ds, spec in grid.datasets.items():
        if ds not in trunk_kps:
            continue
        sets = keypoint_sets(ds, inv, trunk_kps[ds])
        pred = eval_pred(results / spec["trunk"], ds)
        if pred.exists():
            rows += [{"dataset": ds, "arm": "zero-shot", "n": 0, "draw": 0, "mask": "",
                      "kind": "reference", "ckpt": "", **r}
                     for r in score(pred, test_csv(ds), sets)]
        for head, pattern in grid.dedicated.items():
            pred = eval_pred(results / pattern.format(ds=ds), ds)
            if pred.exists():
                rows += [{"dataset": ds, "arm": f"dedicated-{head}", "n": ALL, "draw": 0,
                          "mask": "", "kind": "reference", "ckpt": "", **r}
                         for r in score(pred, test_csv(ds), sets)]

    # ── cells ──────────────────────────────────────────────────────────────────
    done = 0
    for c in expand(grid):
        d    = cell_dir(results, grid, c)
        pred = eval_pred(d, c.dataset)
        if not (d / ".done").exists() or not pred.exists() or c.dataset not in trunk_kps:
            continue
        done += 1
        sets = keypoint_sets(c.dataset, inv, trunk_kps[c.dataset], c.mask)
        for sub, ckpt in CKPTS:   # best-validation checkpoint, and the final step if evaluated
            pred = eval_pred(d, c.dataset, sub)
            if pred.exists():
                rows += [{"dataset": c.dataset, "arm": c.arm, "n": c.n, "draw": c.draw,
                          "mask": "+".join(c.mask), "kind": c.kind, "ckpt": ckpt, **r}
                         for r in score(pred, test_csv(c.dataset), sets)]

    # ── masked references: zero-shot, and the curve cell that saw the labels ──────
    m = grid.masked
    for ds, settings in (m.get("settings") or {}).items():
        for kps in settings:
            hidden = {"hidden": list(kps)}
            zs = eval_pred(results / grid.datasets[ds]["trunk"], ds)
            if zs.exists():
                rows.append({"dataset": ds, "arm": "zero-shot", "n": 0, "draw": 0,
                             "mask": "+".join(kps), "kind": "masked", "ckpt": "",
                             **score(zs, test_csv(ds), hidden)[0]})
            for n in m.get("n_frames", []):
                for draw in m.get("draws", grid.draws):
                    ref = cell_dir(results, grid, Cell(ds, ANCHORED, n, draw))
                    for sub, ckpt in CKPTS:
                        p = eval_pred(ref, ds, sub)
                        if (ref / ".done").exists() and p.exists():
                            rows.append({"dataset": ds, "arm": "labelled", "n": n,
                                         "draw": draw, "mask": "+".join(kps), "kind": "masked",
                                         "ckpt": ckpt, **score(p, test_csv(ds), hidden)[0]})

    df = pd.DataFrame(rows)
    df.to_csv(out / "cells.csv", index=False)

    # ── markdown table: mean +- sd over draws ─────────────────────────────────
    lines = [f"# {grid.name}: pooled mean px on the target's test frames (mean +- sd over draws)",
             "", f"{done} finished cells; config `{args.config}`.", ""]
    if len(df):
        best = df[df.ckpt.isin(["best", ""])]
        for kset in ("all", "supported"):
            sub = best[(best.kind != "masked") & (best.keypoints == kset)]
            if sub.empty:
                continue
            agg = sub.groupby(["dataset", "arm", "n"])["mean_px"].agg(["mean", "std"])
            agg = agg.reset_index()
            agg["cell"] = agg.apply(mean_sd, axis=1)
            piv = agg.pivot_table(index=["dataset", "arm"], columns="n", values="cell",
                                  aggfunc="first")
            piv = piv[[c for c in [0] + grid.n_frames if c in piv.columns]]
            lines += [f"## keypoints: {kset}", "", md_table(piv), ""]
        for ckpt in ("best", "last"):
            m = df[(df.kind == "masked") & (df.keypoints == "hidden") & df.ckpt.isin([ckpt, ""])]
            if (m.ckpt == ckpt).sum() == 0:
                continue
            agg = m.groupby(["dataset", "mask", "arm"])["mean_px"].agg(["mean", "std"])
            agg = agg.reset_index()
            agg["cell"] = agg.apply(mean_sd, axis=1)
            piv = agg.pivot_table(index=["dataset", "mask"], columns="arm", values="cell",
                                  aggfunc="first")
            lines += [f"## masked-label protocol: error on the hidden keypoints ({ckpt} "
                      "checkpoint)", "", md_table(piv), ""]
    (out / "table.md").write_text("\n".join(lines) + "\n")
    print(f"{done} finished cells, {len(df)} rows -> {out}/cells.csv, table.md")


if __name__ == "__main__":
    main()
