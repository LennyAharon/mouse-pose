"""Build a replay fine-tuning CSV: the n-1 corpus + the N target frames of a few-shot cell.

The N frames are the ones Lightning Pose itself selects for ``train_frames=N`` with
``rng_seed_data_pt=<draw>`` on the target's train CSV, so a replay cell is trained on
exactly the frames of the corresponding plain few-shot cell. The N rows are repeated
``--repeat`` times so the target's sampling queue is not exhausted after N draws
(TemperatureSampler ends an epoch when any queue empties).

Usage: python scripts/build_replay_csv.py --dataset cheese-2d --n_frames 10 --draw 0
Writes <data_dir>/CollectedData_replay-<dataset>-tf<N>-draw<D>_train.csv and prints the
selected frames.
"""
import argparse
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from mouse_pose.paths import load_paths

LOO = {
    "ibl":           "face+cheese+caz+kondo",
    "kondo":         "face+ibl+cheese+caz",
    "cheese-2d":     "face+ibl+caz+kondo",
    "cazettes-side": "face+ibl+cheese+kondo",
    "facemap":       "ibl+cheese+caz+kondo",
}


def selected_frames(data_dir: Path, dataset: str, n_frames: int, draw: int) -> list[str]:
    """Reproduce the few-shot cell's train split: same config keys, same seed."""
    from lightning_pose.data import get_dataset, get_imgaug_transform
    from lightning_pose.data.factory import get_data_module

    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "configs" / "model.yaml")
    cfg.data.data_dir = str(data_dir)
    cfg.data.video_dir = str(data_dir / "videos")
    cfg.data.csv_file = f"CollectedData_{dataset}_train.csv"
    cfg.training.train_frames = n_frames
    cfg.training.rng_seed_data_pt = draw
    ds = get_dataset(cfg=cfg, data_dir=str(data_dir), imgaug_transform=get_imgaug_transform(cfg))
    dm = get_data_module(cfg=cfg, dataset=ds, video_dir=str(data_dir / "videos"))
    dm.setup(stage="fit")
    return [ds.image_names[i] for i in dm.train_dataset.indices]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset",  required=True, choices=sorted(LOO))
    parser.add_argument("--n_frames", type=int, required=True)
    parser.add_argument("--draw",     type=int, default=0)
    parser.add_argument("--repeat",   type=int, default=20, help="times each target row is repeated")
    args = parser.parse_args()

    data_dir = Path(load_paths()["data_dir"])
    frames   = selected_frames(data_dir, args.dataset, args.n_frames, args.draw)
    assert len(frames) == args.n_frames, (len(frames), args.n_frames)

    corpus = pd.read_csv(data_dir / f"CollectedData_{LOO[args.dataset]}_train.csv", header=[0, 1, 2], index_col=0)
    target = pd.read_csv(data_dir / f"CollectedData_{args.dataset}_train.csv",      header=[0, 1, 2], index_col=0)
    rows   = target.loc[frames]
    assert list(rows.columns) == list(corpus.columns), "column layout differs between corpus and target CSV"
    merged = pd.concat([corpus] + [rows] * args.repeat)

    out = data_dir / f"CollectedData_replay-{args.dataset}-tf{args.n_frames}-draw{args.draw}_train.csv"
    merged.to_csv(out)
    print(f"corpus {len(corpus)} rows + {len(rows)} target frames x{args.repeat} -> {len(merged)} rows -> {out}")
    for f in frames:
        print("  ", f)


if __name__ == "__main__":
    main()
