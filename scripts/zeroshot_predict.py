"""Zero-shot predictions of a leave-one-out trunk on the left-out dataset.

The run's own evaluation already predicts every dataset's test frames (eval/<ds>/predictions.csv);
this adds the left-out dataset's training frames, which the model never saw either, so zero-shot
transfer can be scored on all of that dataset's labelled frames. Writes
<run>/zeroshot/<dataset>_train_predictions.csv. Called by scripts/train_plan.sh after each loo run.

With --every_ckpt, it also predicts the left-out dataset's TEST and TRAIN frames with every periodic
checkpoint the run kept (``*-periodic.ckpt``, from ``+training.ckpt_every_n_steps=N``) and with the
best checkpoint, writing <run>/zeroshot/steps/step<k>_<dataset>_{test,train}_predictions.csv — zero-shot
transfer along training, to see whether it is still improving when training stops.

    python scripts/zeroshot_predict.py --run <run_dir> --dataset kaufman [--data_dir <data_dir>]
    python scripts/zeroshot_predict.py --run <run_dir> --dataset ibl --every_ckpt
"""

import argparse
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run",        required=True, type=Path, help="trained run directory")
    ap.add_argument("--dataset",    required=True,            help="left-out dataset name")
    ap.add_argument("--data_dir",   default=None,  type=Path, help="corpus dir (default: paths.yaml)")
    ap.add_argument("--every_ckpt", action="store_true",      help="also predict test+train with every periodic checkpoint")
    args = ap.parse_args()

    from lightning_pose.api import Model
    from lightning_pose.api.model import load_model_from_checkpoint
    from omegaconf import open_dict

    if args.data_dir is None:
        from mighty_mouse.paths import load_paths

        args.data_dir = Path(load_paths()["data_dir"])
    model = Model.from_dir(args.run)
    # predict every frame of the csv, not the temperature sampler's training subset
    with open_dict(model.config.cfg):
        model.config.cfg.training.sampling_temperature = None

    def predict(split, out):
        res = model.predict_on_label_csv(
            csv_file=str(args.data_dir / f"CollectedData_{args.dataset}_{split}.csv"),
            data_dir=str(args.data_dir),
            compute_metrics=False,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        res.predictions.to_csv(out)
        print(f"wrote {out} ({len(res.predictions)} frames)")

    out = args.run / "zeroshot" / f"{args.dataset}_train_predictions.csv"
    if not out.exists():
        predict("train", out)
    if not args.every_ckpt:
        return
    ckpts = sorted((args.run / "tb_logs").glob("*/version_*/checkpoints/*-periodic.ckpt"),
                   key=lambda p: int(re.search(r"step=(\d+)", p.name).group(1)))
    if not ckpts:
        raise SystemExit(f"no *-periodic.ckpt in {args.run} (train with +training.ckpt_every_n_steps=N)")
    for ck in ckpts:
        step = int(re.search(r"step=(\d+)", ck.name).group(1))
        outs = {s: args.run / "zeroshot" / "steps" / f"step{step:05d}_{args.dataset}_{s}_predictions.csv"
                for s in ("test", "train")}
        if all(o.exists() for o in outs.values()):
            continue
        model.model = load_model_from_checkpoint(cfg=model.config.cfg, ckpt_file=str(ck), eval=True,
                                                 skip_data_module=True)
        print(f"checkpoint {ck.name}")
        for s, o in outs.items():
            if not o.exists():
                predict(s, o)


if __name__ == "__main__":
    main()
