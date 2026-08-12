# Running the head-fixed training sweep

The sweep answers: **does adding more datasets to the training mix help?** See
[`build_dataset.md`](build_dataset.md) for how the tags are built and why the design is n=1 / n−1 /
n=all rather than every combination.

Two phases, in order:

1. **n=1 and n=all** — the headline comparison. Each dataset alone versus everything merged.
2. **n−1 leave-one-out** — the follow-up, attributing the result to individual datasets.

Every tag contains all available frames, so **every run uses `--train_frames 1`** (Lightning Pose's
"use every frame in the CSV" convention). There is no 200/400/600 learning-curve sweep — earlier
revisions of this doc described one against the balanced `-600` tags, which no longer exist.

## Prerequisites

`paths.yaml` must have `data_dir` pointing at the built dataset. For `results_dir`, follow the
versioning convention in the main README: point it at the unversioned `results/head-fixed` scratch
path, run the sweep, then rename to `results/head-fixed_vN` once everything finishes — keeping the
data and results version numbers in lockstep.

Lightning Pose asserts that `data.video_dir` (`<data_dir>/videos`) is a real directory before
training starts, even though the head-fixed datasets train from extracted frames and never predict
on video. `build_dataset.py` doesn't create it, so make it once per data dir — an empty directory is
enough, and without it every run dies at startup on a bare `AssertionError`:

```bash
mkdir -p <data_dir>/videos
```

## Smoke test first

`--dry_run` only prints commands; it cannot tell you whether Lightning Pose will accept them.
Run one `--debug` job before any real sweep — it trains three two-step epochs and then evaluates,
exercising compose → train → validate → checkpoint → evaluate → cleanup in a couple of minutes:

```bash
python scripts/train_sweep.py --debug \
    --csv_files "CollectedData_cheese-2d_train.csv" \
    --train_frames "1" --seeds "0" --backbones "vits_dino"
```

Expect four `Mean pixel error:` lines — one per dataset in `EVAL_DATASETS` — and
`Deleted 1 checkpoint file(s)`. The pixel errors are meaningless (six training steps); what is
being checked is that every stage runs and that `eval/<dataset>/pixel_error.csv` has 43 columns.

**Delete the output afterwards.** A debug run writes to exactly the same
`<tag>/<losses>/tf1/<backbone>/seed<N>` path a real run uses, so leaving it there causes
`--skip_existing` to skip that combo and keep the garbage numbers:

```bash
rm -rf <results_dir>/cheese-2d_train
```

## Phase 1 — n=1 and n=all

15 jobs: 5 tags (4 single-dataset baselines + 1 all-dataset merge) × 3 seeds. Always `--dry_run` first to
inspect the exact `litpose train` commands before committing to a multi-hour run:

```bash
python scripts/train_sweep.py --dry_run \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_ibl_train.csv;CollectedData_cheese-2d_train.csv;CollectedData_cazettes-side_train.csv;CollectedData_face+ibl+cheese+caz_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino"
```

Then for real (`--skip_existing` makes it safe to re-run after an interruption):

```bash
python scripts/train_sweep.py \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_ibl_train.csv;CollectedData_cheese-2d_train.csv;CollectedData_cazettes-side_train.csv;CollectedData_face+ibl+cheese+caz_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --skip_existing
```

Each model is evaluated against **every** per-dataset test CSV, so a single-dataset model's
performance on the other datasets' test sets comes for free. Pixel error is NaN for keypoints absent
from a given dataset.

## Phase 2 — n−1 leave-one-out

12 jobs: 4 leave-one-out tags × 3 seeds. 27 jobs across both phases.

```bash
python scripts/train_sweep.py \
    --csv_files "CollectedData_face+ibl+cheese_train.csv;CollectedData_face+ibl+caz_train.csv;CollectedData_face+cheese+caz_train.csv;CollectedData_ibl+cheese+caz_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --skip_existing
```

Read each against the n=all result: the gap between `face+ibl+cheese` and `face+ibl+cheese+caz` is
what `cazettes-side` contributes.

## Lightning AI, parallel

`train_sweep_lightning.py` takes identical CLI arguments — only the launch mechanism differs, since
combo generation, naming, and command building are shared via `mouse_pose/train.py`. Add
`--machine` (e.g. `L4` for `vits_dino`) and drop `conda`/sequential assumptions:

```bash
pip install -e ".[lightning]"

python scripts/train_sweep_lightning.py \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_ibl_train.csv;CollectedData_cheese-2d_train.csv;CollectedData_cazettes-side_train.csv;CollectedData_face+ibl+cheese+caz_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --machine L4 \
    --skip_existing
```

Both phases can be launched together — Lightning runs each Job independently, so there's no
sequencing benefit to separating them the way the local loop needs. Check the total against your
account's concurrent-job limit before launching everything at once.

**One-time setup on the Studio:** archive the data directory and upload the archive (not the
extracted 10k-file directory — file-by-file snapshotting is what was slow) to wherever this
machine's `paths.yaml` has `data_dir` pointing:

```bash
tar -cf head-fixed.tar -C data head-fixed
```

Each job extracts it into place itself if `data_dir` doesn't already exist (see
`make_extract_command` in `mouse_pose/train.py`) — safe to duplicate per job, since each Lightning
Job is an isolated snapshot of the Studio's filesystem rather than a shared mount. `results_dir`
must point at storage that outlives an individual job (e.g. a teamspace-mounted drive); unlike
`data_dir`, that one *should* be shared and persistent.

**Preflight:** every job command begins with a check that `lightning_pose` resolved to the local
editable clone rather than a released site-packages build, and aborts the job if not (see
`make_preflight_command`). Without it, a job whose environment silently fell back to the released
package would train successfully and produce plausible-looking pixel errors while ignoring every
local change to LP's data loaders. Pass `--allow_stock_lp` when running against the released package
is intentional.

## After training

Freeze the run alongside the data version it was trained against:

```bash
mv results/head-fixed results/head-fixed_vN
```

Model checkpoints are deleted once evaluation completes — `eval/<dataset>/` is what's kept
long-term, not the weights (see `evaluate_model` in `mouse_pose/train.py`). If a job trained
successfully but died before its chained eval step, finish it without retraining:

```bash
python -m mouse_pose.train --output_dir <results_dir>/<tag>/<losses>/tf1/<backbone>/seed<N>
```
