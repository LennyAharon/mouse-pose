---
name: train-plan
description: Run after a corpus version is built (or when the user asks what to train). Shows which recipe-of-record models exist or are missing for the current data version (all-data trunk, dedicated per-dataset models, leave-one-out trunks), asks the user which stages and seeds to run, and launches them in sequence on the GPU with scripts/train_plan.sh.
---

# Training plan

**Never launch training on your own.** `plan` and `dry` are always safe and are what this skill
runs by default. `run` starts GPU jobs and is only allowed when the user has named, in this
conversation, the stages (and seeds) to run; if in doubt, show the table and ask.

`scripts/train_plan.sh` knows the recipe of record (shared head, T=2, per-dataset zoom-in/out) in
the two sizes the user trains (2026-09-29): **S** = ViT-S DINOv3 12k steps
(`configs/model_zoominout.yaml`, results under `trunks/`, `dedicated/`) and **B** = ViT-B DINOv3
24k steps (`configs/ablations/model_zoominout_24k.yaml`, under `trunks_24k/`, `dedicated_24k/`);
`--arch S|B|SB` (default both). It never retrains something that exists (`--skip_existing`), so it
can be re-run at any time.

1. `bash scripts/train_plan.sh plan` and show the table to the user. Rows are `all` (the
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
   2026-09-29: the all-data trunk in both sizes, then leave-one-out for selected datasets). Alone
   on the L4 a ViT-S 12k trunk takes ~1.5 h and a ViT-B 24k trunk ~5.5 h; two at a time each is
   slower (~2.5-3 h / ~8-10 h).
4. `bash scripts/train_plan.sh show <stages>` prints the script `run` would launch (check it),
   then `bash scripts/train_plan.sh run <stages> --seeds "..." [--arch ...]`. It runs detached
   (`setsid nohup`, survives the session). A plan runs its own jobs **one at a time**; before each
   job it waits until fewer than **2 training runs** are on the GPU, counted globally (every
   `scripts/train_sweep.py` process, including other sessions'). So when the user asks for another
   run while one is training, a new plan starts right away and runs next to it (user, 2026-09-29);
   a third waits. Never use `wait -n` for slot control (it once let 3 runs start). After every `loo` run it writes zero-shot
   predictions on the left-out dataset's train frames (`scripts/zeroshot_predict.py` ->
   `<run>/zeroshot/<ds>_train_predictions.csv`); test frames are in `<run>/eval/<ds>/`. Log at
   `<results_dir>/_logs/train_plan_<stamp>_<stages>.log`; `bash scripts/train_plan.sh status` to
   check. A `PARTIAL` dir from a crash must be deleted before re-running (skip_existing keys on the
   dir). Queued runs read `paths.yaml` when they launch: do not repoint it while a plan is live.
5. When done: rerun `plan` to confirm every row is `done`, note the outcome in
   `docs/results_catalog.md` (roots under `trunks/` and `dedicated/` for this version) and in
   memory, and tell the user which experiments are now unblocked (few-shot cells need the `loo`
   trunks; `scripts/fewshot_cell.sh` must be pointed at the new trunk paths first). Evaluation:
   the `eval-suite` skill (leave-one-out zero-shot scoring is its step 5).

Adding a dataset: append its short name to the `TAG` map in `train_plan.sh` (and the registry).
