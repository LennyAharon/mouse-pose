# Operations guide — training, evaluation, and inference on the head-fixed corpus

Operational reference for running everything this project currently supports, written so
an agent (or person) with no prior context can operate the pipeline. Design rationale
lives elsewhere (`docs/build_dataset.md`, the experiment plan); this file is *how*, not
*why*. When this file and the code disagree, the code wins — update this file.

## Environment facts that override the README

- One conda env (`cloudspace`), always active. Run `python ...` / `litpose ...` directly —
  **never** `conda run -n pose` (that env does not exist here).
- `lightning_pose` is installed **editable from the local `lightning-pose/` clone**
  (branch `mouse_model`). The checked-out branch IS the installed version; edits take
  effect on the next Python process. Long-running sweeps import at launch — editing the
  clone mid-sweep silently splits the sweep across two code versions.
- `paths.yaml` (repo root, gitignored) declares `raw_dir` / `data_dir` / `results_dir`.
  It is read into module constants at import time; editing it mid-process does nothing.
- An L4 GPU is attached. Two concurrent 12k-step trainings fit (~11 GB total); a third
  process (eval, rendering) also fits.

## Data pipeline

```bash
python scripts/convert_dataset.py --dataset <name>     # raw -> canonical CSVs, per dataset
python scripts/build_dataset.py --tag <tag> ...        # merge per-dataset CSVs into a tag
python -m mouse_pose.inventory                         # validate + regenerate inventory
```

The dataset registry (`configs/dataset_registry.yaml`) is the single ordered source of
dataset ids — list position = id, append-only. `python -m mouse_pose.inventory` validates
the 36-keypoint schema, asserts train/test session disjointness, derives the
direct/trainable/eval keypoint masks, and regenerates `docs/dataset_inventory.md` plus a
machine-readable `dataset_inventory.json` next to the data. Run it before freezing any
data change.

Merged tags currently built: 5 singles, 5 leave-one-out, 1 all-dataset
(`face+ibl+cheese+caz+kondo`). `CollectedData_<tag>_{train,test}.csv` live in `data_dir`.

## Training

All training goes through the sweep scripts (never bare `litpose` for real runs — the
sweep wires the registry, output layout, and evaluation):

```bash
python scripts/train_sweep.py \
    --csv_files "CollectedData_face+ibl+cheese+caz+kondo_train.csv" \
    --train_frames "1" --seeds "0" --backbones "vits_dinov3" \
    --sampling_temperatures "2" --head_modes "shared" \
    --keep_checkpoints --skip_existing
```

Dimensions (semicolon-separated except `--losses_to_use`, which is comma-separated):

| flag | values | meaning |
|---|---|---|
| `--sampling_temperatures` | `1;2;inf` | supervision-space sampling temperature. 1/unset = stock frame-proportional loader (no path component); >1 adds `sampling-T<x>/` to the output path and pulls the registry into the config |
| `--head_modes` | `shared;per_dataset` | shared = one 36-kp head (stock); per_dataset = one complete head per registry dataset, adds `head-per_dataset/` path component |
| `--seeds` | `0;1;2` | varies the data split/batch order only (`rng_seed_model_pt` stays 0) |
| `--keep_checkpoints` | flag | retain `*.ckpt` after eval. **Required** for any model whose weights are needed later: per-dataset heads (blind eval), trunks (fine-tune), leave-one-out (zero-shot) |
| `--debug` | flag | 6-step smoke test; delete its output afterwards |

Output layout: `results_dir/<tag>_train/<losses>/[head-<mode>/][sampling-T<x>/]tf<N>/<backbone>/seed<N>/`.
Stock cells (shared, T=1) keep the historical path with no extra components, so old
results stay recognized by `--skip_existing`.

Schedules are step-based (12,000 steps; validation every `val_check_interval: 1000`
global steps with `check_val_every_n_epoch: null` — both keys required together, do not
restore an integer to the epoch key). Batch size 32, backbone `vits_dinov3` for all
factorial work.

Every completed run is automatically evaluated against **all five** per-dataset test
CSVs → `seed<N>/eval/<dataset>/{predictions,pixel_error}.csv`. Multi-dataset runs also
save `split_manifest.json` (exact split membership + per-dataset counts).

### Fine-tuning from a trunk

`get_model` loads initial weights from `model.checkpoint`. Two Hydra traps: checkpoint
filenames contain `=` (quote the value) and the key is absent from model.yaml (prepend `+`):

```bash
litpose train configs/model.yaml --output_dir <out> --overrides \
    data.data_dir=<data_dir> data.csv_file=CollectedData_<d>_train.csv \
    model.backbone=vits_dinov3 "model.losses_to_use=[]" \
    "+model.checkpoint='<trunk>/tb_logs/.../epoch=NN-step=12000-best.ckpt'" \
    training.train_frames=1 training.rng_seed_data_pt=0 \
    training.optimizer_params.learning_rate=1e-05 \
    training.min_steps=2000 training.max_steps=2000 training.unfreezing_step=1 \
    training.val_check_interval=250 \
    "training.lr_scheduler_params.multisteplr.milestone_steps=[1000]"
```

Then evaluate standalone (below). Verify the log contains `loading weights from` — the
factory loads with `strict=False`, so a broken path can otherwise pass silently.

## Evaluation

Automatic after each sweep run. Standalone / re-run:

```bash
python -m mouse_pose.train --output_dir <run_dir> [--csv_file <train csv>] --keep_checkpoints
```

(`--csv_file` is inferred from standard sweep paths but must be given for non-standard
dirs like `finetune-exp/…`.) Omitting `--keep_checkpoints` **deletes the checkpoints**
after eval — the historical default.

**Blind mode** (per-dataset-head models only — scores without test-time dataset identity):

```bash
python scripts/blind_eval.py --run_dir <run_dir>   # writes eval_blind/<dataset>/pixel_error.csv
```

**Zero-shot (leave-one-out) scoring:** score the held-out dataset only on keypoints
that at least one of the n−1 training datasets could teach — the intersection of the
target's `eval` keypoints with the union of the training datasets' `trainable` lists
(both from `dataset_inventory.json`). Keypoints exclusive to the held-out dataset are
untrained garbage by construction and are excluded from means. Always report the
per-keypoint breakdown with each keypoint's teaching datasets; view mismatch
(left/right/midline) strongly modulates transfer within the supported set.

Metric convention: mean pixel error over labeled keypoints, always excluding
`pupil_center_right` (hflip-only channel — never score it).

## Inference on unseen video

The shared-head model runs on video directly (no dataset id needed):

```python
from lightning_pose.api import Model
model = Model.from_dir(run_dir)             # run must have kept its checkpoint
model.predict_on_video_file(
    video_file=..., output_dir=...,
    compute_metrics=False, generate_labeled_video=True,
)
```

Per-dataset-head models need either `model._load(); model.model.predict_dataset = "<name>"`
(oracle routing for a known source) or `model.model.predict_mode = "blind"` (unknown
source / zero-shot). Note: IBL right-camera training data was **flipped**; unflipped
right-camera video is out of distribution.

## Qualitative overlay videos

Everything already rendered is cataloged in `docs/qualitative_catalog.md` (folder →
model, frames, conventions) — check it before re-rendering, append to it after
delivering something new.

```bash
python scripts/render_supermodel_videos.py \
    --run_dir <run_dir> --out_dir <folder> --style classes   # GT red + green/blue + names
python scripts/render_supermodel_videos.py ... --style perkp # color per keypoint + legend.png
```

CPU-only (reads the run's existing `eval/*/predictions.csv`); safe alongside training.
Re-encode to H.264 for in-IDE playback:
`ffmpeg -i x.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4` (binary via
`python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`).

## Remote jobs (Lightning AI)

`scripts/train_sweep_lightning.py` — same args as the local sweep, plus:

- `--publish_dir /teamspace/gcs_folders/head-fixed-nips26` — a Job's filesystem dies with
  the job; results are copied to this GCP teamspace folder **only after eval succeeds**
  (a failed job publishes nothing; a missing directory is the failure report).
- `--single_job` — chain all combos sequentially in one job; pair with
  `--max_runtime <seconds>` (default allocation is 3 h — too short for multi-combo jobs).
- Every job starts with a preflight that aborts if `lightning_pose` resolved to
  site-packages instead of the local clone.
- `reuse_snapshot=True` is the default: after editing Lightning Pose code, a reused stale
  snapshot silently trains the old code — verify with one `--debug` job.
- Do not point `results_dir` at the folder; jobs publish, the Studio reads.

## Known gotchas (each has cost real time)

- Sweep list args are semicolon-separated; `--losses_to_use` is comma-separated.
  `--train_frames 1` means "all frames".
- `paths.yaml` is read at import time; only `convert_dataset.py` has CLI overrides.
- Hydra: quote values containing `=`; `+key=` for keys absent from model.yaml.
- TensorBoard: run `./tensorboard.sh` from the studio root, never bare `tensorboard`.
  Do not `pkill -f tensorboard` (matches the invoking shell); kill by port/PID.
- Checkpoints are best-by-`val_supervised_loss`, written only at validation points.
- The `finetune-exp/` and `qualitative/` folders under `results_dir` are experiment
  areas — nothing there is consumed by the sweeps' `--skip_existing`.
