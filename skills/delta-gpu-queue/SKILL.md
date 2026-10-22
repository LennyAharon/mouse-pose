---
name: delta-gpu-queue
description: Use EVERY time a GPU job is about to be submitted on NCSA Delta (sbatch / srun / train_plan.sh submit / any training, scoring or inference run), every time one starts or finishes, and whenever the user asks about job status or wait times. Measures where the request would start soonest (sbatch --test-only, nothing submitted), picks the fastest route, and reports the expected wait, run time and finish in the chat. Mandatory: the user asked for this report on every run (2026-10-22).
---

# Delta GPU queue: report the wait, minimise the wait

The user wants, **in the chat, for every run**: how long it will wait, why, how long it will run,
and when it will finish, and the route that gets a GPU soonest. Never submit a GPU job without
this report, and never leave a queued job unreported.

## 1. Before submitting: measure

```bash
scripts/delta/gpu_wait.sh --time <HH:MM:SS> [--gpus N --cpus N --mem 60G]
```
It submits nothing. Sections: (1) cluster health, i.e. GPU nodes per state and the newest GPU job
start of ANY user (if nothing started for >2 h the cluster is broken, not the queue: say so, point
to https://status.ncsa.illinois.edu and help@ncsa.illinois.edu); (2) per partition: priority
factor, max time, queue depth and the **estimated start for this exact request** (`sbatch
--test-only`); (3) GPU-hours left and fair-share; (4) my jobs with estimated starts; (5) how long
my recent jobs waited and ran.

Choose `--time` from a measured duration (table below, or section 5) plus ~30%, never a blind
maximum: interactive partitions cap at 1 h, and short accurate limits backfill sooner.

## 2. Pick the route (fastest first)

1. **Interactive partitions** (`gpuA100x4-interactive`, `gpuA40x4-interactive`,
   `gpuA100x8-interactive`): priority factor **10000** vs 100 regular / 30 preempt, so the job goes
   to the front of the queue. Limits: **1 h max**, **1 SUBMITTED job per user per partition**
   (`QOSMaxSubmitJobPerUser`: a pending job holds the slot too, so a second job cannot even queue
   behind it). Use them whenever the job fits in ~55 min. A job can hold several GPUs of one node
   (more runs at once) but then waits for that many free GPUs. A run killed at 1 h is lost
   (training does not resume): only if the measured time fits. **No H200** (user, 2026-10-22: not
   needed, and its few nodes are never the fast queue); A40 / A100 are what we use.
   To give a slot the feeder holds to another job (e.g. an analysis job), stop the feeder, submit
   when the slot frees, restart the feeder: it skips a partition that has any job of mine.
2. **Pack small runs onto one GPU**: e.g. a 3-seed ensemble as ONE job
   (`scripts/delta/train_lp.sbatch` with `SEEDS="0 1 2"`); one job waits instead of three.
3. **List every acceptable partition in one job** (`-p a,b,c`): SLURM takes the first that frees.
   Interactive ones can be in the list too; a job is held to the strictest max time of its list,
   so for >1 h jobs leave them out.
4. **Preemptible partitions** (`*-preempt`, shorter queue) ONLY with a requeue-safe script
   (`scripts/delta/train_plan_job.sbatch` deletes an unfinished run dir on restart; otherwise a
   requeued `--skip_existing` run "finishes" without training).
5. Request only what is needed (1 GPU, 16 CPU, 60G for one LP run); larger requests wait longer.
6. Skip partitions that section 1 shows drained, and anything whose estimate is days away.
7. Don't waste GPU-hours or queue time on avoidable failures: smoke-test a new config first
   (2 epochs / a few hundred steps, interactive), then submit the real runs.

## 3. Report in the chat (every submission)

A short block, before or right after submitting:

| | |
|---|---|
| request | GPUs / CPU / mem / time limit, and why that time (measured X min + margin) |
| fastest options | top 3 partitions with estimated start + wait, from section 2 |
| chosen | partition list, and why (incl. why NOT interactive / preempt if not used) |
| expected | start, run time, finish (start + measured duration) |
| context | cluster health one-liner, fair-share, GPU-hours left |

Then the job ids and `squeue -u $USER -o "%.10i %.30j %.3t %.20S %R"` (`%S` = estimated start).

## 4. Follow up (every job)

- Arm a background watcher (`sacct -j <id> -X -n -o State` in an until-loop; `squeue` sometimes
  times out under load: treat an empty answer as "unknown", never as "finished").
- On **start**: report the actual wait (start - submit) vs the estimate, node and GPU.
- A few minutes in: check the log for errors before the other jobs reach a GPU.
- On **finish**: report run time vs the estimate and update the duration table below.
- When the user asks for status: `scripts/delta/train_progress.sh` (per running run: steps / 12000,
  steps/s, projected end of training vs the job's hard limit, flagged if it may not fit, errors, and
  the feeder's state) plus `scripts/delta/gpu_wait.sh --jobs` for queued jobs. Give the numbers,
  not a description. The user can run `watch -n 60 scripts/delta/train_progress.sh` themselves.
- Many runs: `scripts/delta/interactive_feeder.sh` keeps every interactive partition busy with one
  multi-GPU job each (`scripts/delta/pack_job.sh`, one run per GPU); the same runs can stay queued
  on regular/preempt partitions, `train_plan_job.sbatch` claims make duplicates exit. Experiment
  definitions: `--specs <configs/experiments/*.yaml>` on the feeder, and as backups
  `python scripts/delta/exp_specs.py <definition> | scripts/delta/submit_specs.sh -p <parts> -t <time>`
  (a definition's `code:` makes both run the experiment worktrees' code).

## Gotchas

- `sbatch` copies the script at submit time: editing it does not change queued jobs. To apply a
  change, cancel and resubmit (cancelling needs the user's OK).
- Queued runs read `paths.yaml` when they START: don't repoint it while jobs are queued.
- `sbatch --test-only` answers per partition; with a multi-partition list it reports only one of
  them, so measure partitions one by one (the script does).
- The account is `beez-delta-gpu` (group-shared GPU-hours, see section 3).

## Measured durations (update after every new kind of run)

| run | GPU | run time | notes |
|---|---|---|---|
| LP ViT-S DINOv3, 300 epochs on 441 frames, 3 seeds on 1 GPU | A100 40GB | 27.5 min | `lp-ibl-r0`, interactive, started at once |
| `active_labeling.py score`, 3 models x 9k frames | A100 40GB | 1.5 min | interactive |
| recipe S trunk (ViT-S DINOv3, 12k steps, bs 32, zoom-in/out) | L4 (Studio) | ~1.5 h | from train-plan skill |
| same, linear head, 1500-step timing | A100 40GB | ~6 it/s | CPU-bound (same speed frozen / unfrozen backbone) |
| same, nonlinear head 256, 4 runs packed on one A40 node (16 CPU each) | A40 | 4.1-4.9 it/s, ~43-48 min train | 2026-10-22; + eval |
| same, nonlinear head 256, 4 packed on A100x8 node / 1 on A100 | A100 | 5.1-5.7 it/s, ~37-40 min train | 2026-10-22; + eval |
| dedicated model (small dataset), nonlinear head | H200 | 11.8-12.6 it/s, ~17 min train | 2026-10-22 (H200 no longer used) |
| zero-shot along training: 11 leave-out trunks x 6 checkpoints, test + train frames | A40 | 14 min | 2026-10-22, interactive, 16 CPU |
