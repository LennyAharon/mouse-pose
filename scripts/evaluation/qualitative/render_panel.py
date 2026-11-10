"""
Transfer-keypoint panel video: every dataset view of a corpus version, side by side.

One panel per camera view, held-out test frames, showing ONLY the keypoints that view's dataset
never supervises -- the transfer class -- at confidence >= 0.7. The grid grows with the corpus:
one panel per (dataset, view) in VIEW_ORDER. Colours are per keypoint group, one saturated hue
each, none white or grey; digit tips are split by side (right green, left red).

Transfer keypoint = not in the dataset's `trainable` set (its own labels plus the lateral partners
horizontal flips supervise). tongue_tip and pupil_center_right are drawn in every view on request;
where the dataset already supervises one the panel header says so. pupil_center_right is drawn
as a ring, never a filled dot.

    python scripts/evaluation/qualitative/render_panel.py --model <run dir with eval/> \
        --out <delivery dir>/<name>.mp4 \
        [--mode transfer|all|own] [--conf 0.7] [--frames 60]

Writes <out dir>/videos/<name>.mp4.
"""

import argparse
from pathlib import Path

from mighty_mouse.evaluation.qualitative.panel import render_panel
from mighty_mouse.paths import load_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--model",  required=True, help="run directory containing eval/<dataset>/")
    ap.add_argument("--out",    required=True,
                    help="<delivery dir>/<name>.mp4 (written into videos/)")
    ap.add_argument("--conf",   type=float, default=0.7)
    ap.add_argument("--mode",   default="transfer", choices=["transfer", "all", "own"],
                    help="transfer: only keypoints the dataset never supervises "
                         "(+ tongue tip, R pupil); all: every channel; "
                         "own: only the dataset's own labelled keypoints")
    ap.add_argument("--datasets", default=None, help="comma list to include (default: all)")
    # cheese-2d is left out by default; pass --exclude "" to include every dataset
    ap.add_argument("--exclude",  default="cheese-2d",
                    help="comma list of datasets to leave out (default: cheese-2d; '' for none)")
    ap.add_argument("--data_dir", default=None,
                    help="corpus data dir; pin it when rendering an older version than paths.yaml")
    ap.add_argument("--frames", type=int, default=60, help="max frames sampled per view")
    ap.add_argument("--panel",  type=int, default=300)
    ap.add_argument("--cols",   type=int, default=5)
    ap.add_argument("--fps",    type=int, default=3)
    ap.add_argument("--all_digits", action="store_true",
                    help="draw every digit tip "
                         "(default: none in facemap, near side only in side views)")
    ap.add_argument("--png",    default="",
                    help="comma list of steps (1-based) also saved as PNG stills in "
                         "<out dir>/png/")
    ap.add_argument("--hide",   default="",
                    help="one-off curation of single panels, e.g. for a figure still: "
                         "'cheese-3d/BC=d*_tip_*;tyssowski/side=eye_*' "
                         "(dataset/view=keypoint globs)")
    ap.add_argument("--png_only", action="store_true",
                    help="write only the --png stills, no video")
    ap.add_argument("--split",  default="test", choices=["test", "train", "both"],
                    help="test = eval/<ds>/predictions.csv (held-out test frames); "
                         "train = zeroshot/<ds>_train_predictions.csv in the run dir, when a "
                         "leave-one-out model was run on its left-out dataset's TRAIN frames; "
                         "both = one panel per view and split")
    args = ap.parse_args()

    render_panel(
        args.out, Path(args.model), Path(args.data_dir or load_paths()["data_dir"]),
        conf=args.conf, mode=args.mode,
        datasets=args.datasets.split(",") if args.datasets else None,
        exclude=[x for x in args.exclude.split(",") if x], frames=args.frames,
        panel=args.panel, cols=args.cols, fps=args.fps, split=args.split,
        all_digits=args.all_digits, hide=args.hide,
        png_steps=[int(x) for x in args.png.split(",") if x], png_only=args.png_only,
    )


if __name__ == "__main__":
    main()
