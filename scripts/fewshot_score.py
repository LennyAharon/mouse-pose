"""Score the few-shot grid: arm 2 (dino) vs arm 3 (trunk), with zero-shot and dedicated anchors.
Metric (project convention, matches the notebooks): POOLED mean pixel error over all labeled
(test frame, keypoint) cells of eval/<ds>/pixel_error.csv, pupil_center_right always excluded.
  full      = every keypoint the dataset labels
  supported = full ∩ union(trainable of the other 4 datasets)   (zero-shot rule, operations.md)
  new       = full \ supported (keypoints the n-1 trunk never trained)
  retain    = transfer-class retention proxy: pooled error on the OTHER four datasets' test sets,
              restricted to keypoints the target does NOT label (the trunk knowledge a lab gets for
              free and cannot measure); zero-shot row = what the trunk had before fine-tuning.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/teamspace/studios/this_studio/mouse-pose")
from mouse_pose.paths import load_paths
P = load_paths(); R = Path(P["results_dir"]); D = Path(P["data_dir"])
inv = json.load(open(D / "dataset_inventory.json"))["datasets"]
DS = ["ibl", "kondo", "cheese-2d", "cazettes-side", "facemap"]
LOO = {"ibl": "face+cheese+caz+kondo", "kondo": "face+ibl+cheese+caz", "cheese-2d": "face+ibl+caz+kondo",
       "cazettes-side": "face+ibl+cheese+kondo", "facemap": "ibl+cheese+caz+kondo"}
def kps(ds, key):
    v = inv[ds][key] if key in inv[ds] else inv[ds]["trainable"]
    return [k for k in v if k != "pupil_center_right"]
def sets(ds):
    full = kps(ds, "eval")
    others = set().union(*[set(kps(o, "trainable")) for o in DS if o != ds])
    return full, [k for k in full if k in others]
def retain(run_dir, ds):
    """Pooled error over the other datasets' test cells for keypoints `ds` does not label."""
    lab = set(kps(ds, "trainable")); vals = []
    for o in DS:
        if o == ds: continue
        csv = Path(run_dir) / f"eval/{o}/pixel_error.csv"
        if not csv.exists(): continue
        keys = [k for k in kps(o, "eval") if k not in lab]
        df = pd.read_csv(csv, index_col=0)
        df = df[[k for k in keys if k in df.columns]].apply(pd.to_numeric, errors="coerce")
        vals.append(df.values.ravel())
    v = np.concatenate(vals) if vals else np.array([np.nan])
    return float(np.nanmean(v))
def score(csv, keys):
    if not Path(csv).exists(): return np.nan
    df = pd.read_csv(csv, index_col=0)
    df = df[[k for k in keys if k in df.columns]].apply(pd.to_numeric, errors="coerce")
    return float(np.nanmean(df.values))
rows = []
for ds in DS:
    full, sup = sets(ds)
    anchors = {"zero-shot(n-1 trunk)": R / "zoom-aug-exp" / f"{LOO[ds]}-T2-zoomaug/seed0/eval/{ds}/pixel_error.csv",
               "dedicated(all labels)": R / f"{ds}_train/supervised/tf1/vits_dinov3/seed0/eval/{ds}/pixel_error.csv"}
    new = [k for k in full if k not in sup]
    for name, csv in anchors.items():
        rows.append(dict(ds=ds, N="-", draw="-", arm=name, full=score(csv, full), supported=score(csv, sup),
                         new=score(csv, new), retain=retain(csv.parent.parent.parent, ds)))
    for N in (10, 25, 50):
        for draw in (0, 1, 2):
            for arm, root in (("dino", "fewshot-exp-dino"), ("trunk5", "fewshot-exp-lr5"), ("trunk", "fewshot-exp"), ("trunk5-hf", "fewshot-exp-headfreeze"), ("trunk5-bf", "fewshot-exp-backfreeze"), ("lora", "fewshot-exp-lora-r16"), ("lora-lr5e-4", "fewshot-exp-lora-r16-lr5e-4"), ("lora-h5e-4", "fewshot-exp-lora-r16-lr5e-5-head5e-4"), ("replay", "fewshot-exp-replay"), ("replay-lora", "fewshot-exp-replay-lora"), ("dino-lora", "fewshot-exp-dino-lora-r16"), ("xfer-cheese2d", "fewshot-exp-xfer-cheese2d"), ("anchor", "fewshot-exp-anchor"), ("anchor-lora", "fewshot-exp-anchor-lora"), ("anchor-w10", "fewshot-exp-anchor-w10"), ("anchor-lora-conf1", "fewshot-exp-anchor-lora-conf1"), ("anchor-conf1", "fewshot-exp-anchor-conf1"), ("lora-zoomaug", "fewshot-exp-lora-r16-lr5e-5-head5e-4-zoomaug"), ("trunk5-zio", "fewshot-exp-lr5-zio"), ("lora-zio", "fewshot-exp-lora-r16-lr5e-5-head5e-4-zio"), ("anchor-lora-conf1-zio", "fewshot-exp-anchor-lora-conf1-zio"), ("anchor-video-conf1-zio", "fewshot-exp-anchor-lora-video-conf1-zio"), ("replay-lora-zio", "fewshot-exp-replay-lora-zio"), ("trunk5-zoomaug", "fewshot-exp-lr5-zoomaug"), ("anchor-lora-w0.2-conf1", "fewshot-exp-anchor-lora-w0.2-conf1"), ("anchor-lora-w10", "fewshot-exp-anchor-lora-w10"), ("anchor-lora-conf1-zoomaug", "fewshot-exp-anchor-lora-conf1-zoomaug"), ("anchor-lora-w0.5-conf1", "fewshot-exp-anchor-lora-w0.5-conf1"), ("anchor-video-conf1", "fewshot-exp-anchor-lora-video-conf1"), ("anchor-video-conf1-s4000", "fewshot-exp-anchor-lora-video-conf1-s4000")):
                csv = R / root / ds / f"tf{N}-draw{draw}/eval/{ds}/pixel_error.csv"
                if csv.exists():
                    rows.append(dict(ds=ds, N=N, draw=draw, arm=arm, full=score(csv, full), supported=score(csv, sup),
                                     new=score(csv, new), retain=retain(csv.parent.parent.parent, ds)))
    print(f"[{ds}] full={len(full)} kps, supported={len(sup)}: {sup}; new={new}")
df = pd.DataFrame(rows)
pd.set_option("display.width", 200)
for ds in DS:
    d = df[df.ds == ds]
    if d.N.astype(str).str.isdigit().any():
        print(f"\n== {ds} ==")
        print(d.to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
# paired summary
p = df[df.N != "-"].pivot_table(index=["ds", "N", "draw"], columns="arm", values=["full", "supported", "new", "retain"])
if not p.empty:
    print("\n== paired cells: dino vs trunk5 (identical protocol, lr 5e-5) vs trunk (lr 1e-5) ==")
    print(p.to_string(float_format=lambda x: f"{x:6.2f}"))
