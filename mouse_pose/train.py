"""
Shared utilities for Lightning Pose training sweeps over the head-fixed dataset.

Used by both `scripts/train_sweep.py` (local, sequential) and
`scripts/train_sweep_lightning.py` (Lightning AI, parallel), so combo
generation, naming, and command-building stay identical across backends —
only *how* a command gets executed differs between the two scripts.

Also invocable standalone to run evaluation only. This exists because a
Lightning AI Job is an independent remote process: there's no "come back to
this process after training finishes" step like the local sweep script has,
so evaluation has to be its own chainable command:

    python -m mouse_pose.train --output_dir <dir> --csv_file <CollectedData_..._train.csv>
"""

import argparse
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mouse_pose.paths import load_paths, repo_root

_paths      = load_paths()
DATA_DIR    = Path(_paths["data_dir"])
RESULTS_DIR = Path(_paths["results_dir"])
CONFIG_FILE = repo_root() / "configs" / "model.yaml"

# Every model is evaluated against each dataset's test CSV. Pixel error is NaN
# for keypoints absent from a given dataset; the plotting script handles filtering.
EVAL_DATASETS = ["facemap", "ibl", "cheese-2d", "cazettes-side", "kondo"]


# ── naming / output dirs ─────────────────────────────────────────────────────

def csv_stem(csv_file: str) -> str:
    """CollectedData_<tag>_train.csv -> <tag>_train"""
    return Path(csv_file).stem.split("_", 1)[1]


def infer_csv_file(output_dir: Path) -> str:
    """Inverse of csv_stem applied inside make_output_dir: the first path
    component under RESULTS_DIR is always csv_stem(csv_file), so the original
    train CSV filename can be recovered from output_dir alone."""
    rel = Path(output_dir).resolve().relative_to(RESULTS_DIR.resolve())
    return f"CollectedData_{rel.parts[0]}.csv"


def losses_tag(losses: list[str]) -> str:
    return "supervised" if not losses else "+".join(sorted(losses))


def sanitize(s: str) -> str:
    return str(s).replace("/", "_").replace(".", "_")


def make_output_dir(csv_file, backbone, train_frames, seed, losses) -> Path:
    return (
        RESULTS_DIR
        / csv_stem(csv_file)
        / losses_tag(losses)
        / f"tf{train_frames}"
        / sanitize(backbone)
        / f"seed{seed}"
    )


def make_job_name(csv_file, backbone, train_frames, seed, losses) -> str:
    """Lightning job name for one sweep combo (unused locally, but shares the
    same inputs as make_output_dir so job name <-> output dir stay traceable)."""
    return "__".join([
        sanitize(csv_stem(csv_file)),
        sanitize(backbone),
        losses_tag(losses),
        f"tf{train_frames}",
        f"s{seed}",
    ])


# ── sweep combo generation ───────────────────────────────────────────────────

def parse_semicolon_list(s: str) -> list[str]:
    return s.split(";")


def build_combos(csv_files, backbones, train_frames, seeds) -> list[tuple]:
    """Cartesian product of one sweep, in (csv_file, backbone, train_frames, seed) order."""
    return list(product(csv_files, backbones, train_frames, seeds))


# ── command building ─────────────────────────────────────────────────────────

def make_train_command(
    csv_file, backbone, train_frames, seed, losses, output_dir, debug=False,
) -> list[str]:
    """Build the `litpose train ...` argv for one sweep combo."""
    lr = 5e-5 if "vit" in backbone else 1e-3
    losses_hydra = f"[{','.join(losses)}]"

    overrides = [
        f"data.data_dir={DATA_DIR}",
        f"data.csv_file={csv_file}",
        f"model.backbone={backbone}",
        f"model.losses_to_use={losses_hydra}",
        f"training.train_frames={train_frames}",
        f"training.rng_seed_data_pt={seed}",
        f"training.optimizer_params.learning_rate={lr}",
    ]

    if "vitb_sam" in backbone:
        overrides.append("training.train_batch_size=16")

    if debug:
        # Step-based keys, not epoch-based: model.yaml schedules by step, and LP asserts
        # the two styles are never mixed. limit_train_batches pins an epoch to two steps
        # regardless of tag size, so the run still reaches the epoch boundary where
        # validation and checkpointing happen. Milestones must stay <= max_steps.
        overrides += [
            "+training.limit_train_batches=2",
            "training.min_steps=2",
            "training.max_steps=6",
            "training.unfreezing_step=1",
            "training.check_val_every_n_epoch=1",
            "training.lr_scheduler_params.multisteplr.milestone_steps=[2]",
            "eval.predict_vids_after_training=false",
        ]

    return (
        ["litpose", "train", str(CONFIG_FILE),
         "--output_dir", str(output_dir),
         "--overrides"] + overrides
    )


def make_eval_command(output_dir, csv_file) -> list[str]:
    """Build the standalone-evaluation argv for one sweep combo (chained onto
    make_train_command's output with `&&` for a single Lightning job command)."""
    return [
        "python", "-m", "mouse_pose.train",
        "--output_dir", str(output_dir),
        "--csv_file", str(csv_file),
    ]


def make_preflight_command(allow_stock_lp: bool = False) -> str:
    """Shell snippet that records which lightning_pose a job actually imported, and by
    default fails the job immediately if it resolved to a site-packages copy instead of
    an editable clone (chained ahead of everything else in a Lightning job command).

    A Lightning Job runs in an environment derived from the launching Studio, and an
    editable install depends on *two* things surviving that transition independently:
    the `.pth` marker, which lives in the env (outside the Studio's own directory), and
    the checkout it points at, which lives inside it. If the marker survives but the
    checkout doesn't, `import lightning_pose` fails loudly and the job dies — fine. The
    bad case is neither surviving while the job image happens to carry a stock PyPI
    lightning-pose: training then succeeds, pixel errors look entirely plausible, and
    every local modification to LP's data loaders is simply absent from the run.

    Failing in the first second of a job is much cheaper than discovering that after a
    48-job sweep, so this errs toward stopping. Pass allow_stock_lp=True to downgrade it
    to a log line for runs that legitimately want the released package.

    Returns a shell string (not an argv list, like make_extract_command) — it needs `;`
    and shell quoting, and is only meaningful for Lightning jobs.
    """
    check = (
        "import lightning_pose as lp, sys; "
        "p = lp.__file__; "
        'print(f"[preflight] lightning_pose {lp.__version__} <- {p}", flush=True); '
        "stock = \"site-packages\" in p; "
    )
    if allow_stock_lp:
        check += (
            'print("[preflight] WARNING: running the released package, not the local "'
            '      "clone — LP source changes are NOT in this job") if stock else None'
        )
    else:
        check += (
            'sys.exit("[preflight] FATAL: lightning_pose resolved to site-packages, so this '
            "job would train with the released package and silently ignore local LP changes. "
            'Re-launch with --allow_stock_lp if that is intended.") if stock else None'
        )

    return f"python -c '{check}'"


def make_extract_command() -> str:
    """Shell snippet that extracts DATA_DIR from a sibling `.tar` archive if
    DATA_DIR doesn't already exist, meant to run once at the start of each
    Lightning job (chained before make_train_command's output with `&&`).

    Each Lightning Job is an isolated snapshot of the launching Studio's own
    filesystem, not a shared mount multiple jobs write into concurrently — so
    unlike a HuggingFace-style dataset cache on shared storage, there's no race
    to guard against here. Every job independently checking "does this exist
    yet" and extracting its own local copy if not is safe to duplicate as-is.

    Also ensures DATA_DIR/videos exists: litpose's config always points
    `video_dir` at this path even though the head-fixed dataset trains from
    already-extracted frames, and the archive doesn't necessarily contain an
    entry for it if it was empty when tarred (`mkdir -p` after the
    test/extract step so it's created either way, whether or not extraction
    ran).

    Returns a shell string (not an argv list, unlike the other make_*_command
    functions) since it needs `||` — only meaningful for Lightning jobs; the
    local sequential script never calls this because DATA_DIR is expected to
    already exist on disk for local runs.

    The test/tar pair is parenthesized because `&&` and `||` have equal precedence
    and associate left to right. Bare, `A && test -d D || tar` would parse as
    `((A && test) || tar)`, so *any* failure in a command chained ahead of this one
    would fall through to the tar branch and let the rest of the chain proceed —
    silently defeating make_preflight_command. The parens keep the fallback bound to
    the test it belongs to.
    """
    archive = f"{DATA_DIR}.tar"
    return (
        f'( test -d "{DATA_DIR}" || tar -xf "{archive}" -C "{DATA_DIR.parent}" )'
        f' && mkdir -p "{DATA_DIR / "videos"}"'
    )


def make_publish_command(output_dir: Path, publish_root: Path) -> str:
    """Shell snippet that copies a finished run directory to persistent storage, meant to
    run last in a Lightning job's command chain.

    A Job's filesystem is an isolated snapshot that is torn down when the job ends, so
    anything written to `output_dir` is gone unless it is copied somewhere that outlives
    the compute. A teamspace folder is the only channel available: a Job cannot write back
    into the launching Studio (it holds a snapshot, not a live mount), and `path_mappings`
    /`artifacts_remote` on `Job.run` apply to docker jobs only, not `studio=` ones.

    Deliberately a copy at the end rather than training directly into the folder. Teamspace
    folders are S3/GCS-backed FUSE mounts, so the repeated checkpoint and tb_log writes a
    training run makes would pay network latency for data that `evaluate_model` deletes
    minutes later anyway. Copying once moves only what is kept — configs, eval CSVs, logs.

    The bigger reason is failure behavior. Chained with `&&` after evaluation, this runs
    only if training *and* evaluation both succeeded, so a job that dies partway through
    leaves nothing behind on shared storage. Training straight into the folder would instead
    publish a half-written directory that looks complete enough for `--skip_existing` to
    skip — exactly the stale-directory failure that lost the earlier `cazettes-side` seeds.

    Mirrors the run's path under `publish_root`, so a published tree has the same
    `<tag>/<losses>/tf<N>/<backbone>/seed<N>` layout as a local one and can be copied back
    into a local results dir wholesale.

    Returns a shell string (not an argv list, like make_extract_command) — it needs shell
    quoting, and is only meaningful for Lightning jobs.
    """
    rel  = Path(output_dir).resolve().relative_to(RESULTS_DIR.resolve())
    dest = Path(publish_root) / rel
    return f'mkdir -p "{dest}" && cp -a "{output_dir}/." "{dest}/"'


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(output_dir: Path, csv_file: str) -> None:
    """Evaluate a trained model against every per-dataset test CSV, then clean
    up the scratch prediction files litpose leaves behind in output_dir and
    delete model checkpoints (*.ckpt) — evaluation predictions/pixel-errors
    under output_dir/eval/ are what's kept long-term, not the weights."""
    from lightning_pose.api import Model
    from lightning_pose.metrics import pixel_error

    output_dir = Path(output_dir)

    print("  Loading model...")
    model = Model.from_dir(output_dir)

    for eval_name in EVAL_DATASETS:
        test_csv = DATA_DIR / f"CollectedData_{eval_name}_test.csv"

        if not test_csv.exists():
            print(f"  WARNING: {test_csv} not found — skipping {eval_name} eval")
            continue

        print(f"  Predicting on {eval_name} test set ({test_csv.name})...")
        result = model.predict_on_label_csv(
            csv_file=test_csv,
            data_dir=DATA_DIR,
            compute_metrics=False,
        )
        preds_df = result.predictions

        labels_df = pd.read_csv(test_csv, header=[0, 1, 2], index_col=0)
        if labels_df.index[0] == labels_df.index.name:
            labels_df = labels_df.iloc[1:]

        shared_idx = preds_df.index.intersection(labels_df.index)
        if len(shared_idx) == 0:
            print("  WARNING: no shared frames between predictions and labels — skipping")
            continue
        preds_df  = preds_df.loc[shared_idx]
        labels_df = labels_df.loc[shared_idx]
        n = len(shared_idx)

        kps = labels_df.columns.get_level_values(1).unique().tolist()
        xy  = ["x", "y"]
        preds_cols  = preds_df.columns.get_level_values(2).isin(xy)
        labels_cols = labels_df.columns.get_level_values(2).isin(xy)
        preds_arr  = preds_df.loc[:, preds_cols].to_numpy().reshape(n, len(kps), 2)
        labels_arr = labels_df.loc[:, labels_cols].to_numpy().reshape(n, len(kps), 2)

        error    = pixel_error(labels_arr, preds_arr)
        error_df = pd.DataFrame(error, index=shared_idx, columns=kps)

        save_dir = output_dir / "eval" / eval_name
        save_dir.mkdir(parents=True, exist_ok=True)
        preds_df.to_csv(save_dir / "predictions.csv")
        error_df.to_csv(save_dir / "pixel_error.csv")

        mean_err = float(np.nanmean(error))
        print(f"    Mean pixel error: {mean_err:.2f} px  →  {save_dir}")

    for fname in ("predictions.csv", "predictions_pixel_error.csv"):
        p = output_dir / fname
        if p.exists():
            p.unlink()
    image_preds = output_dir / "image_preds"
    if image_preds.exists():
        shutil.rmtree(image_preds)

    n_deleted = 0
    for ckpt in output_dir.rglob("*.ckpt"):
        ckpt.unlink()
        n_deleted += 1
    print(f"  Deleted {n_deleted} checkpoint file(s)")


def _main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained head-fixed model against every per-dataset test CSV. "
                     "Also useful standalone for finishing sweep jobs that trained successfully "
                     "but stalled before their chained eval step completed.",
    )
    parser.add_argument(
        "--output_dir", required=True, type=Path,
        help="Model output_dir passed to `litpose train`",
    )
    parser.add_argument(
        "--csv_file", default=None,
        help="Train CSV filename the model was trained on (for logging only). "
             "Inferred from output_dir's path structure if omitted.",
    )
    args = parser.parse_args()
    csv_file = args.csv_file or infer_csv_file(args.output_dir)
    print(f"output_dir: {args.output_dir}")
    print(f"csv_file:   {csv_file}")
    evaluate_model(args.output_dir, csv_file)


if __name__ == "__main__":
    _main()
