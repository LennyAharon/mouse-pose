#!/usr/bin/env python3
"""
Score a per-dataset-head model in blind mode against every per-dataset test CSV.

Blind mode runs all heads and combines predictions per keypoint (confidence-weighted,
supporting heads only) — no dataset identity at test time. Results land in
eval_blind/<dataset>/pixel_error.csv alongside the run's oracle eval/, so the two
modes can be compared per dataset. Requires the run to have kept its checkpoint
(train with --keep_checkpoints).

    python scripts/blind_eval.py --run_dir <results>/.../head-per_dataset/tf1/vits_dinov3/seed0
"""

import argparse
from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths
from mouse_pose.registry import load_registry

_paths   = load_paths()
DATA_DIR = Path(_paths["data_dir"])


def main():
    parser = argparse.ArgumentParser(
        description="Blind-mode evaluation for per-dataset-head models.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    args = parser.parse_args()

    from lightning_pose.api import Model
    from lightning_pose.data.datasets import HeatmapDataset
    from lightning_pose.metrics import pixel_error

    model = Model.from_dir(args.run_dir)
    model._load()  # model attribute is lazy
    model.model.predict_mode = "blind"

    # Model.from_dir loads with skip_data_module=True, so the supporting-set mask is
    # all-True; rebuild it from the run's training CSV (the pool the heads trained on)
    train_ds = HeatmapDataset(
        root_directory=DATA_DIR,
        csv_path=model.cfg.data.csv_file,
        image_resize_height=model.cfg.data.image_resize_dims.height,
        image_resize_width=model.cfg.data.image_resize_dims.width,
        imgaug_transform=iaa.Sequential([]),
        dataset_names=list(model.cfg.data.dataset_names),
    )
    model.model.set_head_keypoint_mask(
        visibility=train_ds.visibility,
        dataset_ids=train_ds.dataset_ids,
        keypoint_names=train_ds.keypoint_names,
        hflip=bool(model.cfg.training.imgaug_hflip),
    )
    print(f"blind eval: {args.run_dir}")
    print(f"  gamma={model.model.blind_gamma} floor={model.model.blind_conf_floor}")
    print(f"  mask per head: {model.model.head_keypoint_mask.sum(dim=1).tolist()}")

    for name in load_registry():
        test_csv = DATA_DIR / f"CollectedData_{name}_test.csv"
        result = model.predict_on_label_csv(csv_file=test_csv, data_dir=DATA_DIR,
                                            compute_metrics=False)
        preds_df  = result.predictions
        labels_df = pd.read_csv(test_csv, header=[0, 1, 2], index_col=0)
        if labels_df.index[0] == labels_df.index.name:
            labels_df = labels_df.iloc[1:]
        shared_idx = preds_df.index.intersection(labels_df.index)
        preds_df, labels_df = preds_df.loc[shared_idx], labels_df.loc[shared_idx]
        n   = len(shared_idx)
        kps = labels_df.columns.get_level_values(1).unique().tolist()
        xy  = ["x", "y"]
        p = preds_df.loc[:, preds_df.columns.get_level_values(2).isin(xy)].to_numpy().reshape(n, len(kps), 2)
        l = labels_df.loc[:, labels_df.columns.get_level_values(2).isin(xy)].to_numpy().reshape(n, len(kps), 2)
        err  = pixel_error(l, p)
        save = args.run_dir / "eval_blind" / name
        save.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(err, index=shared_idx, columns=kps).to_csv(save / "pixel_error.csv")
        print(f"  {name:15s} blind mean {np.nanmean(err):7.2f} px")


if __name__ == "__main__":
    main()
