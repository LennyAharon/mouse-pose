---
name: adaptation-curves
description: Zero-shot / few-shot / all-frames adaptation curves per dataset (dedicated model vs DINOv3 from scratch vs Mighty Mouse + anchored LoRA) and the masked-label-protocol bars (MM + plain LoRA vs MM + anchored LoRA). Use to run the grid, rerun it after a dataset or corpus change, add a dataset or an arm, hand it off, run it on a SLURM cluster (ACCESS), or collect and plot its results.
---

# Adaptation curves

The question: how fast does each method reach good accuracy on a lab it has never seen? For every
target dataset the grid trains and scores:

| arm | init | trained | x positions |
|---|---|---|---|
| zero-shot | the leave-that-dataset-out MM trunk (shared nonlinear head) | nothing | N = 0 |
| `mm-anchored-lora` | the same trunk | LoRA r64 + head; the frozen trunk is distilled into every channel the frame does not label | 10, 25, 50, all |
| `mm-lora` | the same trunk | LoRA r64 + head, no anchor (paper baseline "LoRA") | 10, 25, 50, all |
| `mm-full-ft` | the same trunk | every weight at 5e-5 (paper baseline "full fine-tuning") | 10, 25, 50, all |
| `dino-linear`, `dino-nonlinear` | DINOv3 ViT-S + a new head | every weight, lr 5e-5 | 10, 25, 50 |
| dedicated (linear, nonlinear) | DINOv3, recipe of record on all frames, 12k steps | everything | reference line |

The masked-label bars: hide one keypoint group from the N = 10 frames the target DOES label,
adapt with `mm-full-ft`, `mm-lora` or `mm-anchored-lora`, and score that group on its held-back
test labels.
The references are zero-shot and `labelled` (the anchored curve cell at the same N and draw,
which saw those labels). Anchored LoRA keeps a keypoint the lab never labelled; full
fine-tuning degrades it and plain LoRA can erase it (the paper's finding).

Scoring is pooled mean px over visible == 2 test labels of the target only (never other datasets).
`all` = every keypoint the target labels; `supported` = those its trunk trained, where zero-shot
is comparable; `new` = the rest; `hidden` = the masked group.

## Pieces

| file | role |
|---|---|
| `configs/adaptation/<name>.yaml` | the whole grid: datasets + their trunks, dedicated paths, N, draws, steps, arms, masked settings, launch `order` of the blocks (e.g. `[10, 25, masked, 50, all]`; a running queue picks it up at its next launch). Header comments explain every field. |
| `mighty_mouse/adaptation.py` | grid expansion, cell ids and directories, Lightning Pose overrides, log checks, scoring (unit-tested: `tests/test_adaptation.py`) |
| `scripts/adapt/plan.py` | prerequisites, done / runnable / blocked, `--jobs` (pending ids), `--status`, `--bundle` (files to copy elsewhere) |
| `scripts/adapt/run_cell.py` | one cell end to end: masked csv, train, log checks, eval on the target, LoRA reload check, delete ckpts, `run_info.json` + `.done` |
| `scripts/adapt/run_local.sh` | queue on this machine: re-plans before each launch (blocked cells join once their trunk finishes), max GPU jobs, STOP file, restart-safe |
| `scripts/adapt/slurm_array.sbatch` | the same cells as a SLURM array (ACCESS) |
| `scripts/adapt/collect.py` | `summary/cells.csv` + `summary/table.md` (cells + zero-shot + dedicated + masked references) |
| `scripts/adapt/plot.py` | `summary/curves_<keypoints>.{pdf,svg}`, `summary/masked.{pdf,svg}` (`--png` only when asked) |

Cells land in `<results_dir>/adaptation/<name>/<dataset>/<arm>/[mask-<kps>/]tf<N>-draw<d>/`, each
with a training log next to it (`<dir>.log`, `<dir>-eval.log`). `run_info.json` records the
overrides, the trunk checkpoint, both repos' commits and the result.

## Prerequisites

1. The corpus version `paths.yaml` selects is built (`dataset-update`), including the
   leave-one-out csv tags the trunks use (`docs/build_dataset.md`).
2. One leave-that-dataset-out trunk per target with the shared nonlinear head, COMPLETED, with its
   `*-best.ckpt` kept and `eval/<target>/predictions.csv` written (training does that):

       python scripts/train_sweep.py --config_file configs/model_zoominout.yaml \
           --csv_files CollectedData_<LOO tag>_train.csv --train_frames 1 --seeds 0 \
           --backbones vits_dinov3 --sampling_temperatures 2 --head_modes shared \
           --keep_checkpoints --skip_existing --output_root <results_dir>/experiments/<id> \
           --extra_overrides "+training.ckpt_every_n_steps=2000;+model.head_hidden_channels=256"

   Datasets that share rig and mice (cheese-2d and cheese-3d) share one trunk that leaves out both.
3. Dedicated models (reference lines only; the grid runs without them): the same command with
   `--csv_files CollectedData_<ds>_train.csv --output_root <results_dir>/dedicated-headmlp256`,
   and without `--head_modes` / `--extra_overrides` into `<results_dir>/dedicated` for the linear head.

`python scripts/adapt/plan.py --config <cfg>` reports what is missing. A running trunk already has
a `-best.ckpt`, so readiness is `train_status.json` == COMPLETED, not a checkpoint. It also flags
any masked setting whose keypoints the target does not label or its trunk does not train.

## Run on this machine

Commit first: queues and `run_info.json` record the commit, and a dirty main checkout breaks the
gated queues of other batches. GPU budget (user, 2026-10-09): at most 3 GPU jobs at once, counting every queue (4 was too
many). `echo N > <queue dir>/max_gpu` changes the limit of a running queue.

1. Canary, one cell per arm type, before any full launch. Read the overrides first:

       python scripts/adapt/run_cell.py --config <cfg> --cell <id> --dry_run

   then queue the canary cells `ibl__dino-linear__tf10__draw0`, `ibl__dino-nonlinear__tf10__draw0`,
   `ibl__mm-anchored-lora__tf10__draw0` and `ibl__mm-lora__mask-pupil_center_left__tf10__draw0`
   (one id per line in a file) with `run_local.sh <cfg> 3 <file>`. Check their `DONE ... px`
   against the zero-shot and dedicated references in `summary/table.md`. Anchored LoRA at N = 10
   should beat zero-shot; DINOv3 at N = 10 should be far worse than both. Look at the train / val
   curves, and note the minutes per cell.
2. Full grid, durable (survives session restarts; never `run_in_background`):

       setsid nohup bash scripts/adapt/run_local.sh <cfg> 3 > /dev/null 2>&1 < /dev/null & disown

   The queue dir is `<results_dir>/adaptation/<name>/_queue`. Hold launches with `touch
   <queue dir>/STOP`; running cells finish. Track progress in `<queue dir>/queue.log` or with
   `plan.py --status`. Before every launch the queue re-plans and takes the first runnable cell in
   grid order, so cells blocked on a trunk join as soon as that trunk is COMPLETED. While only
   blocked cells remain, it polls every 10 min. A failed cell (`end ... (exit 1)` in the log) is
   never relaunched by itself: read `<queue dir>/<id>.out` and the cell's log, fix the cause, delete
   `<queue dir>/started_<id>`, and restart the queue if it has ended.
3. `python scripts/adapt/collect.py --config <cfg>`, then `python scripts/adapt/plot.py --config <cfg>`
   (also `--keypoints supported`). Report the table, not only the figure.

## Run on a SLURM cluster (ACCESS) or hand it off

Nothing in the code is machine-specific: paths come from `paths.yaml` and the config's paths are
relative to `results_dir`.

1. Environment (match this studio: Python 3.12, torch 2.8, Lightning Pose 2.3.1 from the branch):

       git clone -b post_sub_lp git@github.com:paninski-lab/lightning-pose.git && pip install -e lightning-pose
       git clone -b post_sub_mm git@github.com:LennyAharon/mouse-pose.git && pip install -e mouse-pose

   Check out the exact commits the local cells used (`run_info.json` -> `code`) when mixing cells
   from two machines in one figure. `litpose` must be on PATH inside the job.
2. `paths.yaml` in the clone: `data_dir` and `results_dir` on the cluster's project or scratch
   storage, with the SAME corpus version name (`head-fixed-v<N>`). `data_dir` must be writable
   (masked cells write their training csv there).
3. Copy the inputs (v9: data 3.9 GB, four trunks + references 1.1 GB):

       rsync -aL <data_dir>/ <cluster>:<data_dir>/        # -L: frames and videos are symlinks here
       python scripts/adapt/plan.py --config <cfg> --bundle bundle.txt
       rsync -a --files-from=bundle.txt <results_dir>/ <cluster>:<results_dir>/

4. On the cluster: edit the `#SBATCH` lines of `scripts/adapt/slurm_array.sbatch` (partition,
   account / allocation, environment activation), then

       python scripts/adapt/plan.py --config <cfg> --jobs jobs.txt
       sbatch --array=0-$(( $(wc -l < jobs.txt) - 1 ))%16 scripts/adapt/slurm_array.sbatch <cfg> jobs.txt

   Cells are idempotent: after failures or time-outs, regenerate `jobs.txt` (done cells drop out)
   and resubmit. Raise `--time` for the all-frames cells on slow GPUs.
5. Bring the cells back (checkpoints are already deleted, so they are small) and collect locally:
   `rsync -a <cluster>:<results_dir>/adaptation/<name>/ <results_dir>/adaptation/<name>/`.

## Rerun after the data change

- **A dataset's labels changed or a dataset was added** (e.g. IBL ears / fingertips): run
  `dataset-update` first. The new corpus version has its own `results_dir`, so the grid lands
  in a fresh tree. Apply the reuse rule there: a trunk is reusable only if every dataset it
  trained on is unchanged. For a change to dataset X, only the leave-X-out trunk survives
  (symlink it); every other trunk, and X's dedicated models, are retrained. Copy the config,
  point `datasets:` at the new trunks, add new keypoint groups to `masked.settings` (e.g. IBL
  ears once IBL labels them), and run `plan.py`: it rejects masked keypoints the target does not
  label or its trunk does not train. Collect re-scores saved predictions against the current
  labels, so references follow label fixes automatically.
- **New dataset**: add it under `datasets:` with its own leave-one-out trunk, and under both
  dedicated patterns. Add masked settings for groups it labels that other labs also label.
- **Same corpus, different trunks or arms** (e.g. the experts head): copy the config under a new
  `name` so cells never mix. An experts trunk (`model.head_groups`) needs
  `PYTHONPATH=<lightning-pose-wt-head-groups-fanin>` for every cell; `run_cell.py` aborts without it.
  Without that guard the grouped weights would load silently into a plain head.

## Checks that run by themselves

`run_cell.py` aborts a cell, with no `.done`, if any of these hold:
- litpose exits non-zero, or training is not COMPLETED;
- the log lacks "loading weights from" (trunk arms), "anchor: frozen teacher attached" (anchored),
  "LoRA: wrapped ... (rank R," (LoRA arms) or "nonlinear head:" (nonlinear DINOv3);
- a DINOv3 arm got LoRA;
- evaluation writes no predictions;
- the reloaded model lacks the trained LoRA adapters.

`plan.py` refuses to queue a cell whose trunk is not ready.

## Cost (L4, 2 jobs at once)

Measured on the first grid (v9, 2026-10-09, two cells sharing the L4): an anchored or plain LoRA
cell takes 12-21 min and a DINOv3-from-scratch cell 8-11 min (2000 steps each), or about 7
few-shot cells per hour. All-frames cells (6000 steps) are roughly 3x longer. The 314-cell grid
takes about 2 days, and the 15 dedicated models about another day. The GPU is far from
saturated (3.5 GB and well under 100 % utilisation per cell), so a cluster with more GPUs scales
almost linearly: one GPU per array task.

## Gotchas

- `paths.yaml` is read at import: restart after editing it.
- Never `pkill -f` / `pgrep -f` (they match the invoking shell); find queues by their pid file.
- The v1 grid (`scripts/fewshot_cell.sh`, `fewshot-exp*/`) is head-fixed-v1 history: different
  trunks, validation only at steps 1000 and 2000. `scripts/anchor_ft.sh` is the single-run
  version of `mm-anchored-lora` (same overrides; it evaluates every dataset).
- No PNGs unless asked; figures are PDF + SVG.
