# AGENTS.md

Instructions for any coding agent (Claude Code, Codex, etc.) working in this repo.

## What this is

`mighty-mouse` combines multiple pose-estimation datasets into one training corpus for
more generalizable [Lightning Pose](https://github.com/danbider/lightning-pose) models.
"head-fixed" is the first combined-dataset experiment. This repo holds all the code;
the data itself lives in sibling directories of the parent `poseinterface/` checkout
(`_raw/`, `data/`, `results/` — see `paths.yaml`).

## Setup

- All Python / Lightning Pose commands run in whatever environment has Lightning Pose
  installed on your machine — the env name isn't fixed (e.g. `conda run -n <env> python ...`,
  or just `python ...` in an already-activated env). Docs write bare `python`/`litpose`.
- `paths.yaml` (repo root) is machine-specific and **not committed** — each person
  creates their own, pointing `raw_dir` / `data_dir` / `results_dir` at their local
  checkout. See `README.md` for the required format.

## Where things live

Full directory layout: see [`README.md`](README.md#directory-layout). Workflows
you'll hit often have their own docs under `skills/<name>/SKILL.md` (plain markdown with a
short `name`/`description` frontmatter — readable by any agent, not just Claude):

- **Onboarding a new raw dataset** → [`skills/preprocess-new-dataset/SKILL.md`](skills/preprocess-new-dataset/SKILL.md)
- **Versioning a dataset's label CSVs** → [`skills/bump-dataset-version/SKILL.md`](skills/bump-dataset-version/SKILL.md)
- **Training a standalone LP model on one raw dataset** → [`skills/train-lightning-pose-model/SKILL.md`](skills/train-lightning-pose-model/SKILL.md)
- **Pseudo-labeling a raw dataset (from another dataset's model, precomputed predictions, or its own hand-corrected rows)** → [`skills/transfer-pseudo-labels/SKILL.md`](skills/transfer-pseudo-labels/SKILL.md)

Read the relevant one before starting the task — they have hard-won detail (what
to ask before converting a dataset, why a version bump can refuse as a no-op, etc.)
that isn't worth re-deriving from scratch.

## Rules for working in this repo

- **Ask before advancing past stage 1 of a new dataset.** Converting a raw dataset to
  standard LP format (`_raw/<name>/`) is a complete, valid stopping point on its own.
  Registering it into the shared corpus (new keypoints in `configs/keypoints.yaml` /
  `model.yaml`, `ALL_DATASETS`, running `convert_dataset.py`) is a separate decision
  that needs its own explicit go-ahead — never assume it just because stage 1
  finished, even in the same conversation. See
  [`skills/preprocess-new-dataset/SKILL.md`](skills/preprocess-new-dataset/SKILL.md)
  for the full list of decisions (new keypoints, laterality, multi-view merging,
  train/test split) that need a human call rather than an inferred default.
- **Don't hardcode values that drift.** The keypoint count, for example, changes as
  datasets are added — read `configs/model.yaml`'s `data.num_keypoints` fresh each
  time rather than writing a specific number into docs, comments, or code.
- **Renaming or deprecating a dataset** touches several places that don't cross-check
  each other (`_raw/<name>/`, `configs/datasets/<name>.yaml`, `ALL_DATASETS`,
  hardcoded path constants inside preprocessing scripts, docs). See
  [`README.md`](README.md#renaming-or-deprecating-a-dataset) for the full checklist
  before doing this.
- **Git: don't stage or commit on the user's behalf.** Make file edits freely, but
  leave `git add` / `git commit` / `git push` to the repo owner unless explicitly
  asked to do otherwise.
