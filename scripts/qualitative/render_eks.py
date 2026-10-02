"""
House-style overlay for EKS outputs on continuous video: one panel per camera, side by side.

Matches the standing ensemble videos (ensemble_variance_rows.py): one saturated hue per
keypoint GROUP (legend in the banner, nothing white or grey), dot = EKS-smoothed position, +-1 posterior SD
cross (x and y separately, sqrt of x_posterior_var / y_posterior_var), a strip beside each panel listing
posterior SD x / y and likelihood for every drawn keypoint, and a per-panel footer with step / source frame /
clip. A keypoint is drawn when the EKS likelihood (ensemble mean) >= --conf. No ground truth (unlabelled
video). Posterior SDs are mostly sub-pixel (median ~0.4 px), so the cross is drawn at --bar_scale x true
length and the banner says so; the strip always gives true pixels. Writes <out dir>/videos/<name>.mp4 only (no PNGs).

--eks and --clip are glob patterns with {cam} replaced by each camera of --cams (EKS output csv with x, y,
likelihood, x_posterior_var, y_posterior_var per keypoint; the clip the EKS was run on).

    python scripts/qualitative/render_eks.py --eks '<dir>/csv/*_{cam}_singleview-eks.csv' \
        --clip '<dir>/videos/zeroshot-*_{cam}_frames*.mp4' --title '...' --out <dir>/<name>_nogt.mp4 \
        [--cams left,right] [--bar_scale 10] [--conf 0.7] [--scale 2] [--fps 60]
"""

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from mighty_mouse.plots.house_style import (
    BANNER_H,
    Writer,
    color,
    fit,
    footer,
    header,
    legend_row,
    put,
    video_path,
)


def one(pattern: str, cam: str) -> str:
    hits = sorted(glob.glob(pattern.replace("{cam}", cam)))
    assert len(hits) == 1, f"{pattern} for {cam}: {len(hits)} matches"
    return hits[0]


def load(cam, eks, clip_pattern):
    f = one(eks, cam)
    df = pd.read_csv(f, header=[0, 1, 2], index_col=0).droplevel(0, axis=1)
    kps = list(df.columns.get_level_values(0).unique())
    a = {q: np.stack([df[k][q].to_numpy(float) for k in kps], 1)
         for q in ("x", "y", "likelihood", "x_posterior_var", "y_posterior_var")}
    clip = one(clip_pattern, cam)
    return kps, a, clip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf",      type=float, default=0.7)
    ap.add_argument("--bar_scale", type=float, default=10.0)
    ap.add_argument("--scale",     type=float, default=2.0, help="display upscale of the 320x256 frames")
    ap.add_argument("--strip",     type=int,   default=260)
    ap.add_argument("--fps",       type=int,   default=60)
    ap.add_argument("--eks",       required=True, help="glob of the EKS csv, {cam} = camera")
    ap.add_argument("--clip",      required=True, help="glob of the video clip, {cam} = camera")
    ap.add_argument("--cams",      default="left,right")
    ap.add_argument("--title",     required=True, help="banner: which EKS, which models")
    ap.add_argument("--out",       required=True, help="<delivery dir>/<name>.mp4 (written into videos/)")
    args = ap.parse_args()

    cams = {c: load(c, args.eks, args.clip) for c in args.cams.split(",")}
    kps = next(iter(cams.values()))[0]
    assert all(v[0] == kps for v in cams.values()), "camera keypoint order differs"
    caps = {c: cv2.VideoCapture(v[2]) for c, v in cams.items()}
    n = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps.values())
    cap0 = next(iter(caps.values()))
    w0, h0 = int(cap0.get(3)), int(cap0.get(4))
    PW, PH, Sw, bh = int(w0 * args.scale), int(h0 * args.scale), args.strip, BANNER_H
    W = (PW + Sw) * len(cams)
    OUT = video_path(args.out)
    vw = Writer(OUT, args.fps, (W, PH + bh))

    for t in range(n):
        canvas = np.full((PH + bh, W, 3), 14, np.uint8)
        legend_row(canvas, fs=0.42)
        line = (f"{args.title} | dot = EKS position, cross = +-1 posterior SD (x{args.bar_scale:g} true length; "
                f"strip = true px) | shown if EKS likelihood >= {args.conf} | no ground truth")
        put(canvas, (8, 48), line, fit(line, W - 16), (205, 205, 205))
        for j, (cam, (_, a, clip)) in enumerate(cams.items()):
            ok, img = caps[cam].read()
            if not ok:
                img = np.zeros((h0, w0, 3), np.uint8)
            p = cv2.resize(img, (PW, PH), interpolation=cv2.INTER_CUBIC)
            strip = np.full((PH, Sw, 3), 24, np.uint8)
            put(strip, (6, 16), "posterior SD (px) x / y   lik", 0.4, (220, 220, 220))
            ln, nd, sds = 34, 0, []
            for i, k in enumerate(kps):
                lik = a["likelihood"][t, i]
                if not np.isfinite(lik) or lik < args.conf:
                    continue
                sx, sy = np.sqrt(a["x_posterior_var"][t, i]), np.sqrt(a["y_posterior_var"][t, i])
                q = (int(a["x"][t, i] * args.scale), int(a["y"][t, i] * args.scale))
                ex, ey = sx * args.scale * args.bar_scale, sy * args.scale * args.bar_scale
                cc = color(k)
                cv2.line(p, (int(q[0] - ex), q[1]), (int(q[0] + ex), q[1]), cc, 1, cv2.LINE_AA)
                cv2.line(p, (q[0], int(q[1] - ey)), (q[0], int(q[1] + ey)), cc, 1, cv2.LINE_AA)
                cv2.circle(p, q, 3, cc, -1, cv2.LINE_AA)
                nd += 1; sds.append(np.hypot(sx, sy))
                if ln < PH - 8:
                    cv2.circle(strip, (10, ln - 4), 4, cc, -1, cv2.LINE_AA)
                    put(strip, (20, ln), f"{k[:16]:16s} {sx:.2f}/{sy:.2f}  {lik:.2f}", 0.33, (215, 215, 215))
                    ln += 12
            header(p, f"{cam} camera | {nd} kps | median posterior SD {np.median(sds) if sds else 0:.2f} px")
            footer(p, f"{t + 1}/{n} | frame {t} | {Path(clip).name}")
            x0 = j * (PW + Sw)
            canvas[bh:, x0:x0 + PW] = p
            canvas[bh:, x0 + PW:x0 + PW + Sw] = strip
        vw.write(canvas)
    vw.close()
    print(f"wrote {OUT} ({n} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
