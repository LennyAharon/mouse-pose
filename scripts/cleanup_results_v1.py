#!/usr/bin/env python
"""
Execute docs/results_cleanup_plan.md on results/head-fixed (data version 1).

Deletes checkpoints where the model never needs to run again, moves superseded roots to
_archive/, deletes regenerable qualitative intermediates. Never touches an eval CSV.

    python scripts/cleanup_results_v1.py            # dry run: prints what would happen
    python scripts/cleanup_results_v1.py --execute  # do it, writing _cleanup_manifest.json
"""

import argparse
import fnmatch
import json
import shutil
from datetime import date
from pathlib import Path

from mouse_pose.paths import load_paths

R = Path(load_paths()["results_dir"])

# ── Tier A: keep root, drop checkpoints of draws 1-2 only ─────────────────────────────
DRAW0_ONLY = [
    "fewshot-exp-anchor-lora-conf1-zio-mask-*",
    "fewshot-exp-lora-r16-lr5e-5-head5e-4-zio",
    "fewshot-exp-lora-r16-lr5e-5-head5e-4-zio-mask-*",
]
# ── Tier B: keep root, drop every checkpoint ───────────────────────────────────────────
DROP_ALL_CKPT = [
    "zoom-aug-exp/*-T2-zoomaug",
    "zoom-aug-exp/*-headperds-zoominout",
    "zoom-aug-exp/*-T1-zoominout",
    "zoom-aug-exp/*-Tinf-zoominout",
    "face+ibl+cheese+caz+kondo_train", "face+cheese+caz+kondo_train", "face+ibl+cheese+kondo_train",
    "face+ibl+cheese+caz_train", "face+ibl+caz+kondo_train", "ibl+cheese+caz+kondo_train",
    "fewshot-exp-replay-lora-zio", "fewshot-exp-replay-lora-zio-mask-*",
]
# ── Tier C: drop every checkpoint, then move to _archive/ ──────────────────────────────
ARCHIVE = [
    "fewshot-exp", "fewshot-exp-lr5", "fewshot-exp-lora-r16-lr5e-5-head5e-4",
    "fewshot-exp-anchor-lora-conf1",
    "fewshot-exp-anchor-lora-conf1-mask-pupil_center_left",
    "fewshot-exp-lora-r16-lr5e-5-head5e-4-mask-pupil_center_left",
    "fewshot-exp-lr5-mask-pupil_center_left",
    "fewshot-exp-anchor-lora-conf1-zoomaug", "fewshot-exp-lora-r16-lr5e-5-head5e-4-zoomaug",
    "fewshot-exp-anchor", "fewshot-exp-anchor-w10", "fewshot-exp-anchor-lora", "fewshot-exp-anchor-lora-w10",
    "fewshot-exp-anchor-lora-w0.2-conf1", "fewshot-exp-anchor-lora-w0.2-conf1-zio-mask-pupil_center_left",
    "fewshot-exp-anchor-lora-video-conf1*", "fewshot-exp-anchor-conf1-zio-mask-pupil_center_left",
    "fewshot-exp-anchor-lora-conf1-ema0.999*", "fewshot-exp-anchor-lora-conf1-wf0.05*",
    "fewshot-exp-anchor-lora-conf1-zio-s4000", "fewshot-exp-anchor-lora-conf1-zio-mask-*-s4000",
    "fewshot-exp-headfreeze", "fewshot-exp-backfreeze", "fewshot-exp-replay", "fewshot-exp-replay-lora",
    "fewshot-exp-xfer-cheese2d",
    "zoom-aug-exp/*-20k", "zoom-aug-exp/*-perds-zoomaug", "zoom-aug-exp/*-T1-zoomaug",
    "token-exp", "scale-exp",
]
# ── Tier D: regenerable intermediates under qualitative/ ───────────────────────────────
INTERMEDIATES = [
    "qualitative/ibl-facecrop-transfer/_work",
    "qualitative/transfer-traces-200f/_clips/*.mp4",
    "qualitative/whisker-heatmaps/ear_why/_occl", "qualitative/whisker-heatmaps/ear_why/_ibl__occl",
    "qualitative/whisker-heatmaps/ear_finetune/data", "qualitative/whisker-heatmaps/ear_finetune/_video",
    "qualitative/whisker-heatmaps/_video", "qualitative/whisker-heatmaps/_video_ibl-ears",
    "qualitative/whisker-heatmaps/heatmaps.npz", "qualitative/whisker-heatmaps/ears_ibl/heatmaps.npz",
    "qualitative/whisker-heatmaps/ear_finetune/ears_ibl_finetuned/heatmaps.npz",
    "qualitative/whisker-heatmaps/ear_why/_cazettes-side_head-fixed", "qualitative/whisker-heatmaps/ear_why/_kondo_head-fixed",
]


def expand(patterns):
    out = []
    for pat in patterns:
        parent = R / Path(pat).parent if "/" in pat else R
        for p in sorted(parent.glob(Path(pat).name)):
            if p.exists() and "_archive" not in p.parts:
                out.append(p)
    return out


def size(paths):
    return sum(f.stat().st_size for p in paths for f in ([p] if p.is_file() else p.rglob("*")) if f.is_file())


def ckpts(root, draw0_only=False):
    found = list(root.rglob("*.ckpt"))
    if draw0_only:
        found = [c for c in found if "-draw0/" not in str(c) + "/"]
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], epilog=__doc__)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()
    manifest = {"date": str(date.today()), "deleted_ckpts": [], "archived": [], "deleted_intermediates": []}

    plan = []
    archive_roots = expand(ARCHIVE)                     # a root belongs to one tier only
    for root in expand(DRAW0_ONLY):
        if root not in archive_roots:
            plan.append(("ckpt", root, ckpts(root, draw0_only=True), False))
    for root in expand(DROP_ALL_CKPT):
        if root not in archive_roots:
            plan.append(("ckpt", root, ckpts(root), False))
    for root in archive_roots:
        plan.append(("archive", root, ckpts(root), True))
    inter = expand(INTERMEDIATES)

    freed = 0
    for kind, root, cks, move in plan:
        s = size(cks); freed += s
        print(f"{kind:8s} {str(root.relative_to(R)):78s} ckpts {len(cks):3d}  {s / 1e9:5.2f} GB" + ("  -> _archive/" if move else ""))
    si = size(inter); freed += si
    for p in inter:
        print(f"{'delete':8s} {str(p.relative_to(R)):78s}          {size([p]) / 1e9:5.2f} GB")
    print(f"\nroots touched: {len(plan)} | checkpoint files: {sum(len(c) for _, _, c, _ in plan)} | to free: {freed / 1e9:.1f} GB")
    if not args.execute:
        print("dry run only; add --execute to apply"); return

    arch = R / "_archive"; arch.mkdir(exist_ok=True)
    for kind, root, cks, move in plan:
        for c in cks:
            manifest["deleted_ckpts"].append(str(c.relative_to(R))); c.unlink(missing_ok=True)
        if move:
            dst = arch / root.relative_to(R)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(root), str(dst)); manifest["archived"].append(str(root.relative_to(R)))
    for p in inter:
        manifest["deleted_intermediates"].append(str(p.relative_to(R)))
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    mf = R / "_cleanup_manifest.json"            # merge with an earlier (partial) run
    if mf.exists():
        old = json.loads(mf.read_text())
        for k in ("deleted_ckpts", "archived", "deleted_intermediates"):
            manifest[k] = sorted(set(old.get(k, [])) | set(manifest[k]))
    mf.write_text(json.dumps(manifest, indent=1))
    print("done; manifest at _cleanup_manifest.json")


if __name__ == "__main__":
    main()
