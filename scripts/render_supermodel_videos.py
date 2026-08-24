#!/usr/bin/env python3
"""
Render qualitative videos of a trained supermodel on held-out test frames.

Reads a model run directory's existing eval/<dataset>/predictions.csv files (no GPU
needed) and overlays predictions on the test frames listed there, writing one mp4 per
dataset plus, for --style perkp, a standalone legend figure.

Styles:
  classes  RED = ground-truth label (with a thin red error line to its prediction),
           GREEN = prediction for a keypoint this dataset labels,
           BLUE = "new" keypoint the unified model adds; names drawn next to markers.
  perkp    RED = ground-truth label (with error line); every predicted keypoint gets
           its own fixed color (identical across datasets/videos); no in-frame names —
           legend.png in the output folder maps color to keypoint name.

Low-confidence predictions (< --conf) are drawn hollow in both styles.
`pupil_center_right` is skipped (hflip-only channel, excluded by convention).

    python scripts/render_supermodel_videos.py \\
        --run_dir <results>/finetune-exp/trunk-sharedT2/seed0 \\
        --out_dir <results>/qualitative/supermouse-sharedT2-perkp \\
        --style perkp
"""

import argparse
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths
from mouse_pose.registry import load_registry

_paths   = load_paths()
DATA_DIR = Path(_paths["data_dir"])

RED = (0, 0, 255)  # BGR


def short(name: str) -> str:
    """Compact keypoint name that stays legible at video scale."""
    for a, b in [("_left", "_L"), ("_right", "_R"), ("bottom", "bot"), ("center", "ctr"),
                 ("front", "fr"), ("back", "bk"), ("upperlip", "uplip"), ("tongue", "tng")]:
        name = name.replace(a, b)
    return name


def perkp_colors(kps: list[str]) -> dict[str, tuple]:
    """One fixed, saturated BGR color per keypoint, decorrelated via golden-ratio hues
    so adjacent names (eye_back_left / eye_back_right) don't get adjacent colors."""
    colors = {}
    for i, k in enumerate(kps):
        h = (0.10 + i * 0.618033988749895) % 1.0
        # keep the wheel out of the red band, which is reserved for ground truth
        if h < 0.06 or h > 0.94:
            h = (h + 0.10) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.95, 1.0)
        colors[k] = (int(b * 255), int(g * 255), int(r * 255))
    return colors


def save_legend(colors: dict[str, tuple], path: Path) -> None:
    """Standalone legend figure mapping color to keypoint name."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(colors)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, ax = plt.subplots(figsize=(9, nrow * 0.32 + 1))
    ax.set_axis_off()
    for i, k in enumerate(names):
        col, row = divmod(i, nrow)
        b, g, r = colors[k]
        ax.scatter(col * 3.0, -row, s=80, color=(r / 255, g / 255, b / 255))
        ax.text(col * 3.0 + 0.25, -row, k, va="center", fontsize=9)
    ax.scatter(0, -nrow - 1, s=80, color=(1, 0, 0))
    ax.text(0.25, -nrow - 1, "ground-truth label (any keypoint)", va="center",
            fontsize=9, fontweight="bold")
    ax.set_xlim(-0.5, ncol * 3.0)
    ax.set_ylim(-nrow - 2, 1)
    fig.suptitle("Supermodel keypoint legend", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Render supermodel overlay videos from a run's eval predictions.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run_dir",  required=True, type=Path, help="model run dir containing eval/<dataset>/predictions.csv")
    parser.add_argument("--out_dir",  required=True, type=Path, help="output folder for mp4s (and legend.png for perkp)")
    parser.add_argument("--style",    default="classes", choices=["classes", "perkp"])
    parser.add_argument("--conf",       type=float, default=0.30, help="below this confidence, draw hollow")
    parser.add_argument("--max_frames", type=int,   default=120,  help="frames per dataset video (evenly sampled)")
    parser.add_argument("--fps",        type=int,   default=2)
    args = parser.parse_args()

    GREEN, BLUE = (0, 220, 0), (255, 140, 40)
    inv      = json.load(open(DATA_DIR / "dataset_inventory.json"))
    datasets = load_registry()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    colors_per_kp = None

    for ds in datasets:
        preds_csv = args.run_dir / "eval" / ds / "predictions.csv"
        if not preds_csv.exists():
            print(f"  {ds}: no predictions at {preds_csv} — skipped")
            continue
        preds  = pd.read_csv(preds_csv, header=[0, 1, 2], index_col=0)
        labels = pd.read_csv(DATA_DIR / f"CollectedData_{ds}_test.csv", header=[0, 1, 2], index_col=0)
        if labels.index[0] == labels.index.name:
            labels = labels.iloc[1:]
        kps = [k for k in preds.columns.get_level_values(1).unique()
               if k not in ("set", "pupil_center_right") and str(k) != "nan"
               and not str(k).startswith("Unnamed")]
        if colors_per_kp is None and args.style == "perkp":
            colors_per_kp = perkp_colors(kps)
            save_legend(colors_per_kp, args.out_dir / "legend.png")
        native = set(inv["datasets"][ds]["trainable"])
        frames = sorted(i for i in preds.index if isinstance(i, str) and i.startswith("labeled-data"))
        if len(frames) > args.max_frames:
            frames = [frames[int(i)] for i in np.linspace(0, len(frames) - 1, args.max_frames)]

        first = cv2.imread(str(DATA_DIR / frames[0]))
        h, w  = first.shape[:2]
        scale = max(1.0, 720.0 / min(h, w))
        vw, vh = int(w * scale), int(h * scale)
        writer = cv2.VideoWriter(str(args.out_dir / f"{ds}.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (vw, vh))

        for fr in frames:
            img = cv2.imread(str(DATA_DIR / fr))
            if img is None:
                continue
            img = cv2.resize(img, (vw, vh), interpolation=cv2.INTER_CUBIC)
            for k in kps:
                try:
                    x = float(preds.loc[fr, (slice(None), k, "x")].iloc[0])
                    y = float(preds.loc[fr, (slice(None), k, "y")].iloc[0])
                    c = float(preds.loc[fr, (slice(None), k, "likelihood")].iloc[0])
                except Exception:
                    continue
                if not (np.isfinite(x) and np.isfinite(y)):
                    continue
                px, py = int(x * scale), int(y * scale)

                gt = None
                if fr in labels.index:
                    try:
                        gx = float(labels.loc[fr, (slice(None), k, "x")].iloc[0])
                        gy = float(labels.loc[fr, (slice(None), k, "y")].iloc[0])
                        gv = float(labels.loc[fr, (slice(None), k, "visible")].iloc[0])
                        if np.isfinite(gx) and np.isfinite(gy) and gv == 2:
                            gt = (int(gx * scale), int(gy * scale))
                    except Exception:
                        pass

                if args.style == "perkp":
                    color = colors_per_kp[k]
                else:
                    color = GREEN if k in native else BLUE

                if gt is not None:
                    cv2.line(img, gt, (px, py), RED, 1, cv2.LINE_AA)
                    cv2.circle(img, gt, 4, RED, -1, cv2.LINE_AA)
                thickness = -1 if c >= args.conf else 1
                cv2.circle(img, (px, py), 4, color, thickness, cv2.LINE_AA)

                if args.style == "classes":
                    label = short(k)
                    cv2.putText(img, label, (px + 6, py + 3), cv2.FONT_HERSHEY_SIMPLEX,
                                0.36, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(img, label, (px + 6, py + 3), cv2.FONT_HERSHEY_SIMPLEX,
                                0.36, color, 1, cv2.LINE_AA)

            if args.style == "classes":
                n_new = sum(1 for k in kps if k not in native)
                banner = (f"{ds}  |  RED=ground truth  GREEN=prediction (labeled kp)  "
                          f"BLUE=NEW kp ({n_new})  hollow=conf<{args.conf}")
            else:
                banner = (f"{ds}  |  RED=ground truth  color per keypoint (see legend.png)  "
                          f"hollow=conf<{args.conf}")
            cv2.putText(img, banner, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(img, Path(fr).parent.name + "/" + Path(fr).name,
                        (10, vh - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (200, 200, 200), 1, cv2.LINE_AA)
            writer.write(img)
        writer.release()
        print(f"  {ds:15s} {len(frames):3d} frames -> {args.out_dir / (ds + '.mp4')}")


if __name__ == "__main__":
    main()
