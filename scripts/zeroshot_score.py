#!/usr/bin/env python
"""
Leave-one-out zero-shot scores (eval-suite skill, step 5): every leave-X-out trunk on the dataset X it
never saw, on BOTH X's test frames (<run>/eval/X/predictions.csv) and train frames
(<run>/zeroshot/X_train_predictions.csv), with the all-data trunk and the dedicated X model (which saw
X's train labels) on X's test frames as references.

Per (dataset, model, split, keypoint), over X's labels with visible == 2 (mighty_mouse.comparison
.visible_errors): n, median / mean px error, detection = share of labels predicted with likelihood
>= --conf, confident-but-wrong = share predicted with likelihood >= --conf AND error > --wrong_px.
A keypoint no other dataset supervises (not in the leave-out trunk's `trainable` union, e.g. ibl's
tongue_end_*) is marked "untrained": a zero-shot error there says nothing about transfer. Pooled rows:
"ALL" = every labelled cell; "TRAINED" = only keypoints the leave-out trunk was trained on (the fair
zero-shot number, same keypoint set for every model of that dataset). Pixel errors are in each dataset's own
image pixels (frame widths differ: 256 px aind-vbn to 1280 px cavelli left camera).

    python scripts/zeroshot_score.py                             # results_dir from paths.yaml
    python scripts/zeroshot_score.py --area trunks-headmlp256 --dedicated dedicated-headmlp256

Writes <results_dir>/qualitative/zeroshot-<MM-DD>/{summary.csv, per_keypoint.csv, README.md}; the README
has the pooled table and, per left-out dataset, a per-keypoint table: "mean / median" px for zero-shot
(test, train), the all-data trunk and the dedicated model (test), plus detection and confident-but-wrong.
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import json

from mighty_mouse.comparison import trained_keypoints, visible_errors
from mighty_mouse.paths import load_paths
from mighty_mouse.registry import load_registry

LEAF = "supervised/sampling-T2/tf1/vits_dinov3/seed0"


def read_preds(path: Path) -> pd.DataFrame:
    p = pd.read_csv(path, header=[0, 1, 2], index_col=0)
    return p[[c for c in p.columns if c[2] in ("x", "y", "likelihood")]]


def score(labels: pd.DataFrame, preds: pd.DataFrame, conf: float, wrong_px: float,
          trained: set[str]) -> pd.DataFrame:
    """One row per labelled keypoint plus pooled rows "ALL" and "TRAINED" (keypoints in `trained`)."""
    err = visible_errors(labels, preds)
    sp = preds.columns[0][0]
    rows, pooled, pooled_tr = [], {"e": [], "c": []}, {"e": [], "c": []}
    for k in err.columns:
        m = err[k].notna()
        if not m.any():
            continue
        e = err.loc[m, k].to_numpy(float)
        c = preds.loc[err.index[m], (sp, k, "likelihood")].to_numpy(float)
        pooled["e"].append(e); pooled["c"].append(c)
        if k in trained:
            pooled_tr["e"].append(e); pooled_tr["c"].append(c)
        rows.append(dict(keypoint=k, n=int(m.sum()), median_px=np.median(e), mean_px=e.mean(),
                         detect=(c >= conf).mean(), conf_wrong=((c >= conf) & (e > wrong_px)).mean()))
    for name, pool in (("TRAINED", pooled_tr), ("ALL", pooled)):
        if pool["e"]:
            e, c = np.concatenate(pool["e"]), np.concatenate(pool["c"])
            rows.append(dict(keypoint=name, n=len(e), median_px=np.median(e), mean_px=e.mean(),
                             detect=(c >= conf).mean(), conf_wrong=((c >= conf) & (e > wrong_px)).mean()))
    return pd.DataFrame(rows)


def keypoint_table(K: pd.DataFrame, ds: str, conf: float, wrong_px: float, trained: set[str]) -> list[str]:
    """Markdown rows: one per labelled keypoint of `ds` (pooled ALL last), "mean / median" px per model."""
    z, a, d = "zero-shot (leave-out)", "all-data trunk", "dedicated"
    sub = K[K.dataset == ds]

    def get(model, split, kp, col):
        r = sub[(sub.model == model) & (sub.split == split) & (sub.keypoint == kp)]
        return r[col].iloc[0] if len(r) else np.nan

    def mm(model, split, kp):
        m, md = get(model, split, kp, "mean_px"), get(model, split, kp, "median_px")
        return "" if np.isnan(m) else f"{m:.1f} / {md:.1f}"

    def pct(model, split, kp, col):
        v = get(model, split, kp, col)
        return "" if np.isnan(v) else f"{v:.0%}"

    kps = [k for k in dict.fromkeys(sub[~sub.keypoint.isin(["ALL", "TRAINED"])].keypoint)] + ["TRAINED", "ALL"]
    out = [f"| keypoint | n test / train | zero-shot TEST | zero-shot TRAIN | all-data TEST | dedicated TEST "
           f"| detect (>= {conf}) test / train | conf-but-wrong (> {wrong_px:g} px) test / train |",
           "|---|---|---|---|---|---|---|---|"]
    for kp in kps:
        nt, nr = get(z, "test", kp, "n"), get(z, "train", kp, "n")
        name = {"ALL": "**ALL keypoints (pooled)**", "TRAINED": "**TRAINED keypoints (pooled)**"}.get(
            kp, kp if kp in trained else f"{kp} *(untrained: no other dataset labels it)*")
        out.append(f"| {name} | {'' if np.isnan(nt) else int(nt)} / {'' if np.isnan(nr) else int(nr)} | "
                   f"{mm(z, 'test', kp)} | {mm(z, 'train', kp)} | {mm(a, 'test', kp)} | {mm(d, 'test', kp)} | "
                   f"{pct(z, 'test', kp, 'detect')} / {pct(z, 'train', kp, 'detect')} | "
                   f"{pct(z, 'test', kp, 'conf_wrong')} / {pct(z, 'train', kp, 'conf_wrong')} |")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    paths = load_paths()
    ap.add_argument("--results_dir", type=Path, default=Path(paths["results_dir"]))
    ap.add_argument("--data_dir", type=Path, default=Path(paths["data_dir"]))
    ap.add_argument("--area", default="trunks-headmlp256", help="trunk folder under results_dir")
    ap.add_argument("--dedicated", default="dedicated-headmlp256", help="dedicated folder under results_dir")
    ap.add_argument("--conf", type=float, default=0.7)
    ap.add_argument("--wrong_px", type=float, default=25.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    datasets = load_registry()
    inventory = json.loads((args.data_dir / "dataset_inventory.json").read_text())["datasets"]
    loo, all_run = {}, None
    for tag_dir in sorted((args.results_dir / args.area).glob("*_train")):
        run = tag_dir / LEAF
        if not (run / "eval").is_dir():
            continue
        zs = list((run / "zeroshot").glob("*_train_predictions.csv"))
        if len(zs) == 1:   # leave-ONE-out; trunks leaving out several datasets are not this table
            loo[zs[0].name[: -len("_train_predictions.csv")]] = run
        elif not zs and tag_dir.name.count("+") == len(datasets) - 1:
            all_run = run

    rows, untrained = [], {}
    for ds in datasets:
        if ds not in loo:
            continue
        trained = trained_keypoints(set(datasets) - {ds}, inventory)
        untrained[ds] = trained
        lab = {s: pd.read_csv(args.data_dir / f"CollectedData_{ds}_{s}.csv", header=[0, 1, 2], index_col=0)
               for s in ("test", "train")}
        models = [("zero-shot (leave-out)", loo[ds], {"test": loo[ds] / "eval" / ds / "predictions.csv",
                                                      "train": loo[ds] / "zeroshot" / f"{ds}_train_predictions.csv"})]
        if all_run:
            models.append(("all-data trunk", all_run, {"test": all_run / "eval" / ds / "predictions.csv"}))
        ded = args.results_dir / args.dedicated / f"{ds}_train" / LEAF
        if (ded / "eval" / ds / "predictions.csv").exists():
            models.append(("dedicated", ded, {"test": ded / "eval" / ds / "predictions.csv"}))
        for model, run, splits in models:
            for split, f in splits.items():
                if not f.exists():
                    continue
                t = score(lab[split], read_preds(f), args.conf, args.wrong_px, trained)
                t.insert(0, "split", split); t.insert(0, "model", model); t.insert(0, "dataset", ds)
                t["run"] = str(run.relative_to(args.results_dir))
                rows.append(t)
    if not rows:
        raise SystemExit(f"no leave-one-out runs with zero-shot predictions under {args.results_dir / args.area}")
    K = pd.concat(rows, ignore_index=True)
    out = args.out or args.results_dir / "qualitative" / f"zeroshot-{date.today():%m-%d}"
    out.mkdir(parents=True, exist_ok=True)
    K.round(4).to_csv(out / "per_keypoint.csv", index=False)
    S = K[K.keypoint == "TRAINED"].drop(columns="keypoint")
    S.round(4).to_csv(out / "summary.csv", index=False)

    def cell(ds, model, split, col, fmt):
        r = S[(S.dataset == ds) & (S.model == model) & (S.split == split)]
        return fmt.format(r[col].iloc[0]) if len(r) else ""
    def mm(ds, model, split):
        r = S[(S.dataset == ds) & (S.model == model) & (S.split == split)]
        return f"{r.mean_px.iloc[0]:.1f} / {r.median_px.iloc[0]:.1f}" if len(r) else ""
    hdr = ("| left-out dataset | zero-shot TEST | zero-shot TRAIN | all-data TEST | dedicated TEST | detect (>= "
           f"{args.conf}) test / train | conf-but-wrong (> {args.wrong_px:g} px) test / train |")
    lines = [hdr, "|---|---|---|---|---|---|---|"]
    for ds in [d for d in datasets if d in loo]:
        z = "zero-shot (leave-out)"
        lines.append(
            f"| {ds} | {mm(ds, z, 'test')} | {mm(ds, z, 'train')} | {mm(ds, 'all-data trunk', 'test')} | "
            f"{mm(ds, 'dedicated', 'test')} | "
            f"{cell(ds, z, 'test', 'detect', '{:.0%}')} / {cell(ds, z, 'train', 'detect', '{:.0%}')} | "
            f"{cell(ds, z, 'test', 'conf_wrong', '{:.0%}')} / {cell(ds, z, 'train', 'conf_wrong', '{:.0%}')} |")
    readme = [
        "---", f"title: Leave-one-out zero-shot, {args.data_dir.name} ({args.area})", f"date: {date.today()}",
        f"data_version: {args.data_dir.name}",
        "question: How well does a trunk that never saw dataset X predict X's labelled keypoints?",
        f"models: {args.area} leave-one-out trunks, all-data trunk, {args.dedicated}",
        "outputs: summary.csv, per_keypoint.csv", "finding: (to fill after reading)", "---", "",
        "# Leave-one-out zero-shot", "",
        f"Each leave-X-out trunk scored on X's labelled keypoints (visible == 2) on X's test AND train frames"
        f" (it saw neither); the all-data trunk and the dedicated X model (both trained on X's train frames)"
        f" on X's test frames are the references. detect = labels predicted with likelihood >= {args.conf};"
        f" conf-but-wrong = likelihood >= {args.conf} and error > {args.wrong_px:g} px. Errors in each dataset's"
        " own image pixels. Pooled over the keypoints the leave-out trunk was trained on (keypoints no other"
        " dataset labels are listed as untrained below). Cells: **mean / median px**. Per-keypoint rows:"
        " `per_keypoint.csv`.", "", *lines, "",
        "## Per keypoint, per left-out dataset", "",
        "Cells are **mean / median px** over that keypoint's labels (visible == 2). zero-shot = the leave-X-out",
        "trunk (never saw X); all-data trunk and dedicated were trained on X's train frames, so only their TEST",
        "column exists. An empty dedicated cell = that dedicated model has not finished yet (rerun to fill).", "",
        *[line for d in [x for x in datasets if x in loo] for line in
          (f"### {d}", "", *keypoint_table(K, d, args.conf, args.wrong_px, untrained[d]), "")],
        "Runs:", "", *[f"- leave-{d}-out: `{loo[d].relative_to(args.results_dir)}`" for d in loo],
    ]
    if all_run:
        readme.append(f"- all-data: `{all_run.relative_to(args.results_dir)}`")
    readme += ["", "```", "python scripts/zeroshot_score.py " + f"--area {args.area} --dedicated {args.dedicated}", "```"]
    (out / "README.md").write_text("\n".join(readme) + "\n")
    print("\n".join(lines)); print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
