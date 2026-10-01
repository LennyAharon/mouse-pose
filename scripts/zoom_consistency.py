"""Item 4 step 1: zoom-consistency signal for each predicted keypoint (no training).

Predicts a dataset's TEST frames a second time after shrinking every image to --scale of its size and padding it back
(centred, black), maps the predictions back to original coordinates, and records per (frame, keypoint) how far the
keypoint moved between the original and the zoomed prediction (pixels, original frame). A model that has really found
a keypoint should put it in the same place; a guess tends to move. Writes <out>/csv/zoom__<name>__<dataset>.csv.

    python scripts/zoom_consistency.py --run <run_dir> --name <id> --dataset ibl --out <dir> [--scale 0.85]
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",      required=True, type=Path)
    ap.add_argument("--name",     required=True)
    ap.add_argument("--dataset",  required=True)
    ap.add_argument("--data_dir", default="/teamspace/studios/this_studio/poseinterface/data/head-fixed-v9", type=Path)
    ap.add_argument("--out",      required=True, type=Path)
    ap.add_argument("--scale",    type=float, default=0.85)
    ap.add_argument("--tmp",      default=None, help="scratch dir for the zoomed images (default: system temp)")
    a = ap.parse_args()

    from lightning_pose.api import Model
    from omegaconf import open_dict

    csv = a.data_dir / f"CollectedData_{a.dataset}_test.csv"
    gt = pd.read_csv(csv, header=[0, 1, 2], index_col=0)
    tmp = Path(tempfile.mkdtemp(dir=a.tmp))
    geom = {}
    for f in gt.index:
        img = cv2.imread(str(a.data_dir / f)); h, w = img.shape[:2]
        sw, sh = int(round(w * a.scale)), int(round(h * a.scale))
        small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros_like(img); ox, oy = (w - sw) // 2, (h - sh) // 2
        canvas[oy:oy + sh, ox:ox + sw] = small
        (tmp / f).parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(tmp / f), canvas)
        geom[f] = (ox, oy, sw / w, sh / h)
    shutil.copy(csv, tmp / csv.name)

    model = Model.from_dir(a.run)
    with open_dict(model.config.cfg):
        model.config.cfg.training.sampling_temperature = None
    orig = pd.read_csv(a.run / "eval" / a.dataset / "predictions.csv", header=[0, 1, 2], index_col=0)
    zoom = model.predict_on_label_csv(csv_file=str(tmp / csv.name), data_dir=str(tmp), compute_metrics=False).predictions
    shutil.rmtree(tmp)

    sp, sz = orig.columns[0][0], zoom.columns[0][0]
    kps = [k for k in dict.fromkeys(orig.columns.get_level_values(1)) if (sp, k, "likelihood") in orig.columns]
    G = np.array([geom[f] for f in zoom.index])
    rows = []
    for k in kps:
        xz = (zoom[(sz, k, "x")].values - G[:, 0]) / G[:, 2]
        yz = (zoom[(sz, k, "y")].values - G[:, 1]) / G[:, 3]
        xo, yo = orig.loc[zoom.index, (sp, k, "x")].values, orig.loc[zoom.index, (sp, k, "y")].values
        d = np.hypot(xz - xo, yz - yo)
        rows.append(pd.DataFrame({"frame": zoom.index, "keypoint": k, "zoom_disp": d,
                                  "conf_zoomed": zoom[(sz, k, "likelihood")].values}))
    out = pd.concat(rows)
    (a.out / "csv").mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out / "csv" / f"zoom__{a.name}__{a.dataset}.csv", index=False)
    print(f"wrote zoom consistency for {a.name} on {a.dataset}: median displacement {np.nanmedian(out.zoom_disp):.2f} px")


if __name__ == "__main__":
    main()
