"""Zero-shot predictions of a leave-one-out trunk on the left-out dataset's TRAIN frames.

The run's own evaluation already predicts every dataset's test frames (eval/<ds>/predictions.csv);
this adds the left-out dataset's training frames, which the model never saw either, so zero-shot
transfer can be scored on all of that dataset's labelled frames. Writes
<run>/zeroshot/<dataset>_train_predictions.csv. Called by scripts/train_plan.sh after each loo run.

    python scripts/zeroshot_predict.py --run <run_dir> --dataset kaufman [--data_dir <data_dir>]
"""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",      required=True, type=Path, help="trained run directory")
    ap.add_argument("--dataset",  required=True,            help="left-out dataset name")
    ap.add_argument("--data_dir", default=None,  type=Path, help="corpus dir (default: paths.yaml)")
    args = ap.parse_args()

    from lightning_pose.api import Model
    from omegaconf import open_dict

    if args.data_dir is None:
        from mighty_mouse.paths import load_paths

        args.data_dir = Path(load_paths()["data_dir"])
    out = args.run / "zeroshot" / f"{args.dataset}_train_predictions.csv"
    out.parent.mkdir(exist_ok=True)
    model = Model.from_dir(args.run)
    # predict every frame of the csv, not the temperature sampler's training subset
    with open_dict(model.config.cfg):
        model.config.cfg.training.sampling_temperature = None
    res = model.predict_on_label_csv(
        csv_file=str(args.data_dir / f"CollectedData_{args.dataset}_train.csv"),
        data_dir=str(args.data_dir),
        compute_metrics=False,
    )
    res.predictions.to_csv(out)
    print(f"wrote {out} ({len(res.predictions)} frames)")


if __name__ == "__main__":
    main()
