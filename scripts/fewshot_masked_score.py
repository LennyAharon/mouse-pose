"""Score the masked-label transfer protocol: for every few-shot root ending in -mask-<kps>, report
the hidden keypoints' error on the TARGET's test set (the true transfer number), next to the
supported (labeled) keypoints, and the same arm without masking / zero-shot for reference.
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, "/teamspace/studios/this_studio/mouse-pose")
from mouse_pose.paths import load_paths
R = Path(load_paths()["results_dir"])
LOO = {"ibl": "face+cheese+caz+kondo", "kondo": "face+ibl+cheese+caz", "cheese-2d": "face+ibl+caz+kondo",
       "cazettes-side": "face+ibl+cheese+kondo", "facemap": "ibl+cheese+caz+kondo"}
def err(csv, keys):
    if not Path(csv).exists(): return np.nan
    df = pd.read_csv(csv, index_col=0); df = df[[k for k in keys if k in df.columns]].apply(pd.to_numeric, errors="coerce")
    return float(np.nanmean(df.values)) if df.size else np.nan
rows = []
for root in sorted(R.glob("fewshot-exp*-mask-*")):
    m = re.match(r"(fewshot-exp.*)-mask-(.+)$", root.name); base, slug = m.group(1), m.group(2); hidden = slug.split("+")
    trunk_sfx = "T2-zoominout" if base.endswith("-zio") else "T2-zoomaug"
    for cell in sorted(root.glob("*/tf*-draw*/eval")):
        ds = cell.parts[-3]; N, draw = re.match(r"tf(\d+)-draw(\d+)", cell.parts[-2]).groups()
        tgt = cell / ds / "pixel_error.csv"
        df = pd.read_csv(tgt, index_col=0); others = [c for c in df.columns if c not in hidden and c != "pupil_center_right" and not c.startswith("Unnamed") and c != "set"]
        rows.append(dict(ds=ds, N=int(N), draw=int(draw), arm=base.replace("fewshot-exp-", "") or "lr1e-5", hidden=slug,
                         hidden_err=err(tgt, hidden), labeled_err=err(tgt, others),
                         same_arm_unmasked=err(R / base / ds / f"tf{N}-draw{draw}/eval/{ds}/pixel_error.csv", hidden),
                         zero_shot=err(R / "zoom-aug-exp" / f"{LOO[ds]}-{trunk_sfx}/seed0/eval/{ds}/pixel_error.csv", hidden)))
if not rows: print("no masked cells yet"); sys.exit()
pd.set_option("display.width", 200)
print(pd.DataFrame(rows).sort_values(["ds", "N", "arm"]).to_string(index=False, float_format=lambda x: f"{x:6.2f}"))
