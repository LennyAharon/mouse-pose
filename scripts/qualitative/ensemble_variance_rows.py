"""
One video, one ROW per ensemble-variance definition, same frames and same centre in every row.

Centre in every row = median of ALL models (per keypoint, frame, coordinate). A keypoint is drawn when at least one model
has confidence >= --conf (same keypoints in every row); hollow ring when the models' mean confidence < --conf. Only the
+-1 SD cross differs between rows (all at true length, x --bar_scale):
  row 1  ensemble var          var_x = Var_m(x_m)
  row 2  inv-conf-weighted     var_x = Var_m(x_m) / mean_m(c_m)
  row 3  mixture               var_x = mean_m(1 / c_m) + Var_m(x_m)   (components with variance 1/c, read as px^2)
(c floored at 1e-5; same for y). Columns = camera views. A strip beside each panel lists the SD x / y per keypoint.
Writes videos/<out>.mp4 and csv/<out>_per_frame_keypoint.csv (+ _summary.csv: per keypoint and variance type, median
SD, and on labelled cells the median error of the ensemble median and Spearman corr(error, SD)). No PNGs.

    python scripts/qualitative/ensemble_variance_rows.py --models <run> <run> <run> --title "..." \
        --out <delivery dir>/<name>.mp4 [--exclude regexes] [--no_gt]
        [--modes mixture] [--include ear_] [--conf 0]
--modes keeps only the named rows (default all three); --include keeps only keypoints matching any regex; --conf 0
disables confidence masking (every included keypoint is drawn on every frame, solid).
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mighty_mouse.paths import load_paths
from mighty_mouse.plots.house_style import (
    BANNER_H,
    Writer,
    color,
    csv_dir,
    fit,
    footer,
    header,
    legend_row,
    put,
    square,
    video_path,
)

MODES = [("ensemble var", "Var(x)"), ("inv-conf-weighted", "Var(x) / mean(c)"), ("mixture", "mean(1/c) + Var(x)")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models",    required=True, nargs="+")
    ap.add_argument("--target",    default="ibl")
    ap.add_argument("--data_dir",  default=None, help="corpus data dir (default: paths.yaml data_dir)")
    ap.add_argument("--title",     default="")
    ap.add_argument("--conf",      type=float, default=0.7)
    ap.add_argument("--bar_scale", type=float, default=1.0)
    ap.add_argument("--frames",    type=int, default=60)
    ap.add_argument("--panel",     type=int, default=440)
    ap.add_argument("--strip",     type=int, default=250)
    ap.add_argument("--fps",       type=int, default=3)
    ap.add_argument("--exclude",   default="")
    ap.add_argument("--include",   default="", help="comma regexes: keep only matching keypoints")
    ap.add_argument("--modes",     default="", help="comma list of rows to draw: 'ensemble var', "
                                                     "'inv-conf-weighted', 'mixture' (default all)")
    ap.add_argument("--no_gt",     action="store_true")
    ap.add_argument("--out",       required=True, help="<delivery dir>/<name>.mp4 (written into videos/)")
    args = ap.parse_args()

    D = Path(args.data_dir or load_paths()["data_dir"])
    preds = [pd.read_csv(Path(m) / "eval" / args.target / "predictions.csv", header=[0, 1, 2], index_col=0) for m in args.models]
    sp = preds[0].columns[0][0]
    kps = [k for k in dict.fromkeys(preds[0].columns.get_level_values(1)) if (sp, k, "likelihood") in preds[0].columns]
    if args.exclude:
        kps = [k for k in kps if not any(re.search(r, k) for r in args.exclude.split(","))]
    if args.include:
        kps = [k for k in kps if any(re.search(r, k) for r in args.include.split(","))]
    idx = preds[0].index
    preds = [p.loc[idx] for p in preds]
    x = np.stack([np.stack([p[(sp, k, "x")].values for k in kps], 1) for p in preds])     # models x frames x kps
    y = np.stack([np.stack([p[(sp, k, "y")].values for k in kps], 1) for p in preds])
    c = np.stack([np.stack([p[(sp, k, "likelihood")].values for k in kps], 1) for p in preds])
    shown = (c >= args.conf).any(0)
    mconf = c.mean(0)
    xm, ym = np.median(x, 0), np.median(y, 0)
    vx, vy = np.var(x, 0), np.var(y, 0)
    cf = np.maximum(c, 1e-5)
    comp = (1.0 / cf).mean(0)
    sd = [(np.sqrt(vx), np.sqrt(vy)),
          (np.sqrt(vx / np.maximum(mconf, 1e-5)), np.sqrt(vy / np.maximum(mconf, 1e-5))),
          (np.sqrt(comp + vx), np.sqrt(comp + vy))]

    modes = MODES
    if args.modes:
        want = [m.strip() for m in args.modes.split(",")]
        keep = [i for i, (name, _) in enumerate(MODES) if name in want]
        assert keep, f"--modes {want} matches none of {[m for m, _ in MODES]}"
        modes, sd = [MODES[i] for i in keep], [sd[i] for i in keep]
    gt = pd.read_csv(D / f"CollectedData_{args.target}_test.csv", header=[0, 1, 2], index_col=0).loc[idx]
    sg = gt.columns[0][0]
    labelled = [k for k in kps if (sg, k, "visible") in gt.columns]
    OUT = video_path(args.out)
    CSV = csv_dir(OUT)
    rows, summ = [], []
    for j, k in enumerate(kps):
        vis = (pd.to_numeric(gt[(sg, k, "visible")], errors="coerce") == 2).values if k in labelled else np.zeros(len(idx), bool)
        err = (np.hypot(xm[:, j] - gt[(sg, k, "x")].astype(float).values, ym[:, j] - gt[(sg, k, "y")].astype(float).values)
               if k in labelled else np.full(len(idx), np.nan))
        err = np.where(vis, err, np.nan)
        for i, f in enumerate(idx):
            if shown[i, j]:
                r = dict(frame=f, keypoint=k, mean_conf=mconf[i, j], x=xm[i, j], y=ym[i, j], err=err[i])
                for (name, _), (a, b) in zip(modes, sd):
                    r[f"sd_{name.replace(' ', '_')}"] = float(np.hypot(a[i, j], b[i, j]))
                rows.append(r)
    per = pd.DataFrame(rows)
    per.to_csv(CSV / (OUT.stem + "_per_frame_keypoint.csv"), index=False)
    for k, g in per.groupby("keypoint", sort=False):
        for name, _ in modes:
            col = f"sd_{name.replace(' ', '_')}"; lab = g[g.err.notna()]
            summ.append(dict(keypoint=k, variance=name, median_sd_px=g[col].median(), median_err_px=lab.err.median(),
                             spearman_err_sd=lab.err.corr(lab[col], method="spearman") if len(lab) > 2 else np.nan))
    pd.DataFrame(summ).to_csv(CSV / (OUT.stem + "_summary.csv"), index=False)

    views = {}
    for i, f in enumerate(idx):
        views.setdefault("left" if f.split("/")[2].endswith("_left") else "right (flipped)", []).append(i)
    views = {v: [ii[int(t)] for t in np.linspace(0, len(ii) - 1, min(args.frames, len(ii)))] for v, ii in sorted(views.items())}
    n = max(len(v) for v in views.values())
    P, Sw, bh = args.panel, args.strip, BANNER_H
    W, H = (P + Sw) * len(views), P * len(modes)
    vw = Writer(OUT, args.fps, (W, H + bh))
    for step in range(n):
        canvas = np.full((H + bh, W, 3), 14, np.uint8)
        legend_row(canvas, kps, fs=0.42)
        shown_rule = ("no confidence masking: every keypoint drawn on every frame" if args.conf <= 0 else
                      f"shown if any model conf >= {args.conf}; ring = mean conf < {args.conf}")
        cross = ("rows differ ONLY in the +-1 SD cross" if len(modes) > 1 else "cross = +-1 SD")
        scale = "" if args.bar_scale == 1 else f" x{args.bar_scale:g}"
        line = (f"{args.title} | {args.target} held-out test frames | dot = median of all {len(preds)} models; "
                f"{shown_rule} | {cross} (true length{scale}) | "
                + ("no ground truth drawn" if args.no_gt else "x = ground truth"))
        put(canvas, (8, 48), line, fit(line, W - 16), (205, 205, 205))
        for col_i, (v, frames) in enumerate(views.items()):
            i = frames[step % len(frames)]; f = idx[i]
            base, ox, oy, s = square(cv2.imread(str(D / f)), P)
            if not args.no_gt:
                for k in labelled:
                    if pd.to_numeric(pd.Series([gt.loc[f, (sg, k, "visible")]]), errors="coerce").iloc[0] == 2:
                        q = (int((float(gt.loc[f, (sg, k, "x")]) + ox) * s), int((float(gt.loc[f, (sg, k, "y")]) + oy) * s))
                        cv2.drawMarker(base, q, color(k), cv2.MARKER_TILTED_CROSS, 11, 2, cv2.LINE_AA)
            for r_i, ((mname, formula), (sx, sy)) in enumerate(zip(modes, sd)):
                p = base.copy()
                strip = np.full((P, Sw, 3), 24, np.uint8)
                put(strip, (6, 16), f"{mname} SD (px) x / y", 0.4, (220, 220, 220))
                ln, nd, sds = 34, 0, []
                for j, k in enumerate(kps):
                    if not shown[i, j]:
                        continue
                    q = (int((xm[i, j] + ox) * s), int((ym[i, j] + oy) * s)); cc = color(k); nd += 1
                    ex, ey = sx[i, j] * s * args.bar_scale, sy[i, j] * s * args.bar_scale
                    th = 2 if mconf[i, j] >= args.conf else 1
                    cv2.line(p, (int(q[0] - ex), q[1]), (int(q[0] + ex), q[1]), cc, th, cv2.LINE_AA)
                    cv2.line(p, (q[0], int(q[1] - ey)), (q[0], int(q[1] + ey)), cc, th, cv2.LINE_AA)
                    cv2.circle(p, q, 4 if th == 2 else 5, cc, -1 if th == 2 else 1, cv2.LINE_AA)
                    sds.append(np.hypot(sx[i, j], sy[i, j]))
                    if ln < P - 8:
                        cv2.circle(strip, (10, ln - 4), 4, cc, -1 if th == 2 else 1, cv2.LINE_AA)
                        put(strip, (20, ln), f"{k[:16]:16s} {sx[i, j]:.1f}/{sy[i, j]:.1f}  c{mconf[i, j]:.2f}", 0.34,
                            (215, 215, 215)); ln += 13
                header(p, f"{mname}: {formula} | {v} | {nd} kps | mean SD {np.mean(sds) if sds else 0:.2f} px")
                footer(p, f"{step % len(frames) + 1}/{len(frames)} {f.split('/')[2][:24]}/{Path(f).name}")
                x0 = col_i * (P + Sw); y0 = bh + r_i * P
                canvas[y0:y0 + P, x0:x0 + P] = p
                canvas[y0:y0 + P, x0 + P:x0 + P + Sw] = strip
        vw.write(canvas)
    vw.close()
    print(f"wrote {OUT} ({n} steps)")


if __name__ == "__main__":
    main()
