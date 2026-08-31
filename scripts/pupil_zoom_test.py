"""Zero-shot pupil_center_left on facemap vs zoom-out factor, for any run directory (CPU).

Facemap has no pupil label; the pupil is scored by its distance to the centre of the four labeled
left-eye contour points. Frames are padded (zoomed out) by the given factors before inference,
so a scale-limited trunk shows a U-shaped curve with its best factor where facemap's eye size
matches its training scale.

    CUDA_VISIBLE_DEVICES= python scripts/pupil_zoom_test.py <run_dir> [--factors 1,2,3,4]
"""
import argparse, glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image

from mouse_pose.paths import load_paths


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--factors",  default="1,1.5,2,3,4,6")
    p.add_argument("--n_frames", type=int, default=25)
    a = p.parse_args()
    from lightning_pose.models.factory import get_model
    P = load_paths(); D = Path(P["data_dir"]); run = Path(a.run_dir)
    cfg = OmegaConf.load(run / "config.yaml")
    m = get_model(cfg, data_module=None, loss_factories={"supervised": None, "unsupervised": None})
    sd = torch.load(glob.glob(str(run / "tb_logs/test/version_0/checkpoints/*.ckpt"))[0], map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd, strict=False); m = m.cpu().eval()
    ip = list(cfg.data.keypoint_names).index("pupil_center_left")
    lab = pd.read_csv(D / "CollectedData_facemap_test.csv", header=[0, 1, 2], index_col=0)
    def kp(k):
        sub = lab.xs(k, axis=1, level=1)
        return (pd.to_numeric(sub.xs("x", axis=1, level=1).iloc[:, 0], errors="coerce"),
                pd.to_numeric(sub.xs("y", axis=1, level=1).iloc[:, 0], errors="coerce"))
    eyes = ["eye_back_left", "eye_front_left", "eye_top_left", "eye_bottom_left"]
    ecx = pd.concat([kp(k)[0] for k in eyes], axis=1).mean(axis=1); ecy = pd.concat([kp(k)[1] for k in eyes], axis=1).mean(axis=1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frames = list(lab.index[::4])[: a.n_frames]
    print(f"{run.name}\n{'zoom-out':>8s} {'median d(pupil,eye ctr)':>24s} {'mean':>6s} {'within 12px':>12s} {'median conf':>12s}")
    for s in [float(x) for x in a.factors.split(",")]:
        ds, cs = [], []
        for f in frames:
            im = Image.open(D / f).convert("RGB"); W, H = im.size; Wp, Hp = int(W * s), int(H * s)
            canvas = Image.new("RGB", (Wp, Hp), tuple(int(v) for v in np.array(im).reshape(-1, 3).mean(0)))
            ox, oy = (Wp - W) // 2, (Hp - H) // 2; canvas.paste(im, (ox, oy))
            x = torch.from_numpy(np.array(canvas.resize((256, 256), Image.BILINEAR))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            with torch.no_grad():
                hm = m.forward((x - mean) / std); kps, conf = m.head.run_subpixelmaxima(hm)
            kps = kps.reshape(-1, 2); px = kps[ip, 0].item() * Wp / 256 - ox; py = kps[ip, 1].item() * Hp / 256 - oy
            ds.append(np.hypot(px - ecx[f], py - ecy[f])); cs.append(conf.reshape(-1)[ip].item())
        ds = np.array(ds)
        print(f"{s:8.1f}x {np.median(ds):24.1f} {ds.mean():6.1f} {(ds < 12).mean() * 100:11.0f}% {np.median(cs):12.2f}")


if __name__ == "__main__":
    main()
