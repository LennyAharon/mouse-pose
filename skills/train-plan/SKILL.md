---
name: train-plan
description: Run after a corpus version is built (or when the user asks what to train). Shows which recipe-of-record models exist or are missing for the current data version (all-data trunk, dedicated per-dataset models, leave-one-out trunks), asks the user which stages and seeds to run, and launches them in sequence on the GPU with scripts/train_plan.sh.
---

# Training plan

**Never launch training on your own.** `plan` and `dry` are always safe and are what this skill
runs by default. `run` starts GPU jobs and is only allowed when the user has named, in this
conversation, the stages (and seeds) to run; if in doubt, show the table and ask.

`scripts/train_plan.sh` knows the recipe of record (`configs/model_zoominout.yaml`: shared head,
T=2, per-dataset zoom-in/out, 12k steps, ViT-S DINOv3) and the three model families. It never
retrains something that exists (`--skip_existing`), so it can be re-run at any time.

1. `bash scripts/train_plan.sh plan` and show the table to the user. Rows are `all` (the
   Mighty Mouse all-data trunk), `dedicated` (one per dataset), `loo` (one leave-one-out per
   dataset); columns are seeds. `done` = evaluated, `PARTIAL` = crashed or running, `missing` =
   never trained, `NO CSV` = build the merged tag first (`dataset-update` skill step 4).
2. If a previous corpus version exists, tell the user which models the data change invalidated
   (`python scripts/data_manifest.py --no-write --diff v<N-1>`), and that a vocabulary change
   invalidates everything.
3. Ask which stages and how many seeds, and wait for the answer. Suggested order: `all` first (it is what every downstream
   experiment uses), then `dedicated` (upper-bound references, small and fast), then `loo` (only
   needed for zero-shot / few-shot transfer experiments). Seeds: 1 for everything first; 3 for
   `all` when ensembles or seed-variance are needed. One 12k-step trunk on the L4 takes roughly
   1-2 h; dedicated models less.
4. Check nothing else is using the GPU (`nvidia-smi`), then `bash scripts/train_plan.sh run
   <stages> --seeds "..."`. It runs detached (`setsid nohup`), one job at a time, log at
   `<results_dir>/_train_plan_<stamp>.log`; `bash scripts/train_plan.sh status` to check. A
   `PARTIAL` dir from a crash must be deleted before re-running (skip_existing keys on the dir).
5. When done: rerun `plan` to confirm every row is `done`, note the outcome in
   `docs/results_catalog.md` (roots under `trunks/` and `dedicated/` for this version) and in
   memory, and tell the user which experiments are now unblocked (few-shot cells need the `loo`
   trunks; `scripts/fewshot_cell.sh` must be pointed at the new trunk paths first).

Adding a dataset: append its short name to the `TAG` map in `train_plan.sh` (and the registry).
