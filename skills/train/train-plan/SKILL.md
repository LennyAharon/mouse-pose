---
name: train-plan
description: Run after a corpus version is built (or when the user asks what to train). Shows which recipe-of-record models exist or are missing for the current data version (all-data trunk, dedicated per-dataset models, leave-one-out trunks), asks the user which stages and seeds to run, and launches them in sequence on the GPU with scripts/train/train_plan.sh.
---

# Training plan

**Never launch training on your own.** `plan` and `dry` are always safe and are what this skill
runs by default. `run` starts GPU jobs and is only allowed when the user has named, in this
conversation, the stages (and seeds) to run; if in doubt, show the table and ask.

`scripts/train/train_plan.sh` knows the recipe of record (shared **nonlinear** head, T=2, per-dataset zoom-in/out).
Since 2026-11-15 that is **B** = ViT-B DINOv3, 384 px stable recipe (`configs/model_recipe_b384.yaml`: 14k steps,
unfreeze 3000, lr halvings 6k/8k/10k, linear backbone warm-up 1000 + gradient clip 1.0; results under `trunks_b384/`,
`dedicated_b384/`), the default arch. **L** = the same recipe with ViT-L DINOv3 (`trunks_l384/`, `dedicated_l384/`;
~45 GB GPU memory, so 80 GB A100 / H100 / H200 only). **S** = the old ViT-S 12k 256 px recipe
(`configs/model_zoominout.yaml`, `trunks/`, `dedicated/`), kept for comparison. `--arch B|L|S` or combinations
(e.g. `BL`). It never retrains something that exists (`--skip_existing`), so it can be re-run at any time.
The geometric unfreeze warm-up and layer-wise lr decay are NOT in the recipe (still being tested, 2026-11-15).

**Head (user, 2026-10-22): nonlinear is the default and the go-to.** Every stage adds
`+model.head_hidden_channels=256` (the adaptation-curves trunk
recipe) and writes to `trunks_b384-headmlp256/`, `dedicated_b384-headmlp256/` (`trunks-headmlp256/` ... for S).
`--head linear` gives the old stock linear head (`trunks/`, `dedicated/`) and only when the user asks.
Check the first run's log for lightning-pose's `nonlinear head: ConvT -> 256 -> ...` line.
On NCSA Delta, launch with `submit` (one SLURM job per run) and/or `scripts/delta/interactive_feeder.sh`
(see `skills/train/delta-gpu-queue`), not `run`.

1. `bash scripts/train/train_plan.sh plan` and show the table to the user. Rows are `all` (the
   Mighty Mouse all-data trunk), `dedicated` (one per dataset), `loo` (one leave-one-out per
   dataset; `loo:<ds>` for one); columns are arch x seed. `done` = evaluated, `PARTIAL` = crashed or running, `missing` =
   never trained, `NO CSV` = build the merged tag first (`dataset-update` skill step 4).
2. If a previous corpus version exists, tell the user which models the data change invalidated
   (`python scripts/data_manifest.py --no-write --diff v<N-1>`), and that a vocabulary change
   invalidates everything.
3. Ask which stages and how many seeds, and wait for the answer. Suggested order: `all` first (it is what every downstream
   experiment uses), then `dedicated` (upper-bound references, small and fast), then `loo` (only
   needed for zero-shot / few-shot transfer experiments). Seeds: 1 for everything first; 3 for
   `all` when ensembles or seed-variance are needed. Train only what the user asked for (on
   2026-09-29: the all-data trunk in both sizes, then leave-one-out for selected datasets). On Delta
   (one GPU, 16 CPU per run) a B run takes ~3.5 h on an A100 and an L run ~4 h on an H200 (both
   ~1 step/s; 1-h interactive slots resume across jobs).
4. `bash scripts/train/train_plan.sh show <stages>` prints the script `run` would launch (check it),
   then `bash scripts/train/train_plan.sh run <stages> --seeds "..." [--arch ...]`. It runs detached
   (`setsid nohup`, survives the session) and keeps **2 training runs** on the GPU whenever it has
   work left (user, 2026-09-29): before each job it waits until fewer than 2 are running, counted
   globally (every `scripts/train/train_sweep.py` process, including other sessions'), then launches it
   in the background. So a queue always fills a free second slot, a plan started next to one other
   run starts at once, and a third run always waits. Never use `wait -n` for slot control (it once
   let 3 runs start). After every `loo` run it writes zero-shot
   predictions on the left-out dataset's train frames (`scripts/evaluation/zeroshot_predict.py` ->
   `<run>/zeroshot/<ds>_train_predictions.csv`); test frames are in `<run>/eval/<ds>/`. Log at
   `<results_dir>/_logs/train_plan_<stamp>_<stages>.log`; `bash scripts/train/train_plan.sh status` to
   check. A `PARTIAL` dir from a crash must be deleted before re-running (skip_existing keys on the
   dir). Queued runs read `paths.yaml` when they launch: do not repoint it while a plan is live.
5. When done: rerun `plan` to confirm every row is `done`, note the outcome in
   `docs/results_catalog.md` (roots under `trunks/` and `dedicated/` for this version) and in
   memory, and tell the user which experiments are now unblocked (few-shot cells need the `loo`
   trunks; `scripts/fewshot_cell.sh` must be pointed at the new trunk paths first). Evaluation:
   the `eval-suite` skill (leave-one-out zero-shot scoring is its step 5).

Adding a dataset: append its short name to the `TAG` map in `train_plan.sh` (and the registry).
