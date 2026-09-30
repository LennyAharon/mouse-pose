"""Write <run>/run_info.json for a fine-tuning run: which trunk, method, settings, frames, code and result.

Called by scripts/anchor_ft.sh after evaluation, so every fine-tune folder can be identified later without
reading logs.

    python scripts/write_run_info.py --out <run_dir> --trunk <trunk_run_dir> --ckpt <ckpt> --dataset ibl ...
"""

import argparse
import datetime
import json
import subprocess
from pathlib import Path


def git_ref(repo: Path) -> str:
    """Return '<branch> <short hash>' of a git checkout, or '?' when unavailable."""
    try:
        h = subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
        b = subprocess.run(["git", "-C", str(repo), "branch", "--show-current"], capture_output=True, text=True)
        return f"{b.stdout.strip()} {h.stdout.strip()}".strip() or "?"
    except OSError:
        return "?"


def main() -> None:
    ap = argparse.ArgumentParser(epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",       required=True, type=Path)
    ap.add_argument("--trunk",     required=True)
    ap.add_argument("--ckpt",      required=True)
    ap.add_argument("--dataset",   required=True)
    ap.add_argument("--n_frames",  required=True)
    ap.add_argument("--steps",     required=True, type=int)
    ap.add_argument("--draw",      required=True, type=int)
    ap.add_argument("--method",    required=True)
    ap.add_argument("--full_ft",   required=True)
    ap.add_argument("--lora_rank", required=True, type=int)
    ap.add_argument("--lora_lr",   required=True, type=float)
    ap.add_argument("--head_lr",   required=True, type=float)
    ap.add_argument("--ft_lr",     required=True, type=float)
    ap.add_argument("--anchor_w",  required=True, type=float)
    ap.add_argument("--val_every", required=True, type=int)
    ap.add_argument("--backbone",  required=True)
    a = ap.parse_args()

    import lightning_pose

    full = a.full_ft == "1"
    info = {
        "trunk_run": a.trunk, "trunk_checkpoint": a.ckpt, "backbone": a.backbone,
        "target_dataset": a.dataset, "train_frames": a.n_frames, "draw": a.draw,
        "steps": a.steps, "val_every": a.val_every, "lr_halves_at": a.steps // 2,
        "method": a.method, "full_finetune": full,
        "lora_rank": None if full else a.lora_rank, "lora_lr": None if full else a.lora_lr,
        "head_lr": None if full else a.head_lr, "backbone_lr": a.ft_lr if full else None,
        "anchor_weight": a.anchor_w, "anchor_conf_power": 1, "anchor_mode": "unlabeled",
        "augmentation": "stock dlc (configs/model.yaml)",
        "code": {
            "mouse_pose": git_ref(Path(__file__).resolve().parents[1]),
            "lightning_pose": git_ref(Path(lightning_pose.__file__).resolve().parents[1]),
        },
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M"),
    }
    m = a.out / "eval" / a.dataset / "comparison_metrics.json"
    if m.exists():
        info[f"result_{a.dataset}_test_mean_px"] = json.load(open(m))["all"]["mean_px"]
    json.dump(info, open(a.out / "run_info.json", "w"), indent=2)
    print(f"wrote {a.out / 'run_info.json'}")


if __name__ == "__main__":
    main()
