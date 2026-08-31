"""Few-shot frames→error curves: DINO+N vs super-mouse(n−1)+N, per dataset, both keypoint sets.
    python scripts/fewshot_plot.py                 # zoom-out trunks: DINO / full FT / LoRA
    python scripts/fewshot_plot.py --trunk zio     # zoom-in/out trunks: DINO / full FT / LoRA / anchored LoRA
Lines = draw-0 cells (the complete grid); small hollow markers = other frame draws where
available (spread). Anchors: zero-shot (n−1 trunk, dashed) and dedicated model (dotted).
Metric: pooled mean pixel error over test (frame, keypoint) cells; pupil_center_right excluded.
"""
import argparse, json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
sys.path.insert(0, "/teamspace/studios/this_studio/mouse-pose")
from mouse_pose.paths import load_paths
P = load_paths(); R = Path(P["results_dir"]); D = Path(P["data_dir"])
inv = json.load(open(D / "dataset_inventory.json"))["datasets"]
DS = ["ibl", "kondo", "cazettes-side", "facemap", "cheese-2d"]
LOO = {"ibl": "face+cheese+caz+kondo", "kondo": "face+ibl+cheese+caz", "cheese-2d": "face+ibl+caz+kondo",
       "cazettes-side": "face+ibl+cheese+kondo", "facemap": "ibl+cheese+caz+kondo"}
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("--trunk", choices=["zoomaug", "zio"], default="zoomaug", help="which trunk generation the arms start from")
args = _ap.parse_args()
if args.trunk == "zoomaug":
    TRUNK_SUFFIX = "T2-zoomaug"; OUT_NAME = "fewshot_curves.png"
    ARMS = {"DINOv3 + N labels": ("fewshot-exp-dino", "#2a78d6"), "super-mouse (n−1) + N labels, full fine-tune": ("fewshot-exp-lr5", "#eb6834"),
            "super-mouse (n−1) + N labels, LoRA adapters": ("fewshot-exp-lora-r16-lr5e-5-head5e-4", "#1baf7a")}
else:
    TRUNK_SUFFIX = "T2-zoominout"; OUT_NAME = "fewshot_curves_zio.png"
    ARMS = {"DINOv3 + N labels": ("fewshot-exp-dino", "#2a78d6"), "super-mouse (n−1) + N labels, full fine-tune": ("fewshot-exp-lr5-zio", "#eb6834"),
            "super-mouse (n−1) + N labels, LoRA adapters": ("fewshot-exp-lora-r16-lr5e-5-head5e-4-zio", "#1baf7a"),
            "super-mouse (n−1) + N labels, anchored LoRA": ("fewshot-exp-anchor-lora-conf1-zio", "#6B4FBB")}
NS = [10, 25, 50]
def kps(ds, key="eval"):
    return [k for k in inv[ds][key] if k != "pupil_center_right"]
def sets(ds):
    full = kps(ds); others = set().union(*[set(kps(o, "trainable")) for o in DS if o != ds])
    return {"supported keypoints": [k for k in full if k in others], "all labeled keypoints": full}
def score(csv, keys):
    if not Path(csv).exists(): return np.nan
    df = pd.read_csv(csv, index_col=0)
    return float(np.nanmean(df[[k for k in keys if k in df.columns]].apply(pd.to_numeric, errors="coerce").values))

fig, axes = plt.subplots(2, len(DS), figsize=(3.4 * len(DS), 6.4), sharex=True)
for j, ds in enumerate(DS):
    S = sets(ds)
    for i, (sname, keys) in enumerate(S.items()):
        ax = axes[i, j]
        zs = score(R / "zoom-aug-exp" / f"{LOO[ds]}-{TRUNK_SUFFIX}/seed0/eval/{ds}/pixel_error.csv", keys)
        ded = score(R / f"{ds}_train/supervised/tf1/vits_dinov3/seed0/eval/{ds}/pixel_error.csv", keys)
        ax.axhline(zs, color="#52514e", ls="--", lw=1.2, label="zero-shot (n−1 trunk)")
        ax.axhline(ded, color="#52514e", ls=":", lw=1.2, label="dedicated (all labels)")
        for arm, (root, col) in ARMS.items():
            d0 = [score(R / root / ds / f"tf{N}-draw0/eval/{ds}/pixel_error.csv", keys) for N in NS]
            ax.plot(NS, d0, color=col, lw=2, marker="o", ms=6, label=arm)
            for draw in (1, 2):
                dd = [score(R / root / ds / f"tf{N}-draw{draw}/eval/{ds}/pixel_error.csv", keys) for N in NS]
                ax.plot(NS, dd, color=col, lw=0, marker="o", ms=5, mfc="white", mew=1.2, alpha=0.8)
        ax.set_xscale("log"); ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS]); ax.minorticks_off()
        ax.grid(axis="y", color="#e5e4df", lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if i == 0: ax.set_title(f"{ds}  ({len(keys)} kps)", fontsize=11)
        if j == 0: ax.set_ylabel(f"{sname}\nmean pixel error (test)", fontsize=9)
        if i == 1: ax.set_xlabel("labeled frames N")
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
fig.suptitle("Few-shot adaptation to a held-out dataset — identical protocol, only the initialization / adapter differs\n"
             f"trunks: {TRUNK_SUFFIX} · lines: frame draw 0 · hollow markers: draws 1–2 · 2000 steps · full FT lr 5e-5 · LoRA r16 (adapters 5e-5, head 5e-4)"
             + (" · anchored: + frozen-trunk distillation on unlabeled channels" if args.trunk == "zio" else ""), fontsize=11)
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
out = R / "qualitative" / "fewshot-curves"; out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / OUT_NAME, dpi=160); print("->", out / OUT_NAME)
