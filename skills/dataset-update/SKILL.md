---
name: dataset-update
description: Run when the user says a dataset's labels (CollectedData CSVs) were updated or a new dataset was added. Registers the dataset version, builds the next corpus version, writes the manifest and changelog entry, sets up the results tree, and says which models can be reused and which must be retrained.
---

# Dataset update

The corpus changes often: label CSVs of one dataset are replaced (frames never change), or a new
dataset arrives. Every such event is a new **dataset version** and a new **corpus version**.
Details and rules: `docs/data_versioning.md`. Do these steps in order; do not skip the register
step even for a "tiny" fix, and never rebuild into the current corpus version.

## 1. Find out what changed

Ask, or read from the message: which dataset(s), where the new `CollectedData*.csv` files are,
and one sentence on what changed (relabeled keypoints, added/removed sessions, new keypoint,
new split). New dataset: also its raw folder name, camera views, and which canonical keypoints
it maps to (needs a new `configs/datasets/<name>.yaml`; see README "Adding a new dataset").

## 2. Put the raw files in place

- Updated labels for an existing dataset: the user overwrites `CollectedData*.csv` in
  `_raw/<dataset>/` (frames stay). `_raw` always holds the current labels.
- New dataset: `_raw/<name>/` with the standard layout, plus `configs/datasets/<name>.yaml` and the
  registrations in `mouse_pose/datasets.py` (README "Adding a new dataset").

## 3. Register the dataset version (before building anything)

`python scripts/data_manifest.py --register <dataset> --note "<what changed>"` hashes the raw
CSVs, snapshots them to `_raw_versions/<dataset>@v<k>/`, and appends the version to
`poseinterface/DATASET_VERSIONS.json`. Repeat per changed/added dataset. If the vocabulary
(`configs/keypoints.yaml`) changed too, say so in the note: it means nothing is reusable.

## 3b. Vocabulary (only if keypoints were added or removed)

`configs/keypoints.yaml` is the single source of truth. Add/remove the canonical names there
(append new ones at the END: channel order is baked into every checkpoint), then
`python scripts/sync_keypoint_configs.py` rewrites `data.keypoint_names` and `data.num_keypoints`
in every `configs/model*.yaml` and `configs/zero_shot/*.yaml`; `--check` (exit 1 on drift) is
also run by step 4 before converting. New dataset needing its own zoom range: add it to
`imgaug_per_dataset_zoom` in `configs/model_zoominout.yaml` (measure eye->nose as % of frame width
against the other rigs; same scale as an existing rig = same range).

## 4. Build the next corpus version

1. Next free N: `ls poseinterface/data | grep head-fixed-v`. Set `paths.yaml` `data_dir` and
   `results_dir` to `head-fixed-v<N>` (both, same N).
2. `python scripts/sync_keypoint_configs.py --check`, then
   `python scripts/convert_dataset.py --dataset <ds> --link_frames` for every dataset (changed and unchanged;
   unchanged ones reproduce the same CSVs — needed so the version is self-contained), then
   `python scripts/build_dataset.py` for the tags in use (`docs/build_dataset.md`).
   Frames: convert with `--link_frames` (symlink `labeled-data/<ds>` to the shared frame pool)
   once that flag exists; until then copies are acceptable.
3. `python scripts/data_manifest.py` (writes `MANIFEST.json`), then
   `python scripts/data_manifest.py --diff v<N-1>` and paste its output into
   `poseinterface/DATA_VERSIONS.md` under `## v<N> — built <YYYY-MM-DD>` with: the user's one-line
   "why", and per changed dataset the date the labels changed (the `date` of its entry in
   `DATASET_VERSIONS.json`, i.e. when it was registered) and what changed.

## 5. Set up the results tree and the reuse decision

Create `results/head-fixed-v<N>/README.md` (data version, manifest hash, which datasets changed).
The `--diff` output lists unchanged datasets. Reuse rule: **a model is reusable in v<N> only if
every dataset it was trained on is unchanged** (same raw hash and converter config).

- Dedicated single-dataset models of unchanged datasets: symlink
  `results/head-fixed-v<N>/dedicated/<ds>` → `../../head-fixed-v<N-1>/<ds>_train` (or the
  previous tree's path) and say so in the README. Do not copy.
- Anything trained on a changed dataset (its dedicated model, every leave-one-out trunk that
  includes it, the all-data model, all few-shot cells on it) must be retrained. Print the list.

## 6. Record it

- Commit the repo changes (configs, registry, `docs/dataset_inventory.md`, any converter/config
  edits) as ONE commit whose subject starts with `data v<N>:` and whose body lists the dataset
  versions (`facemap@v2, hantman-mv@v1, ...`) and vocabulary size; push. Then append the commit
  hash to the `## v<N>` entry in `poseinterface/DATA_VERSIONS.md` (`code: mouse-pose <hash>`),
  so the data version and the code that built it are cross-referenced both ways.
- `CLAUDE.md` "Phase" paragraph: current corpus version and date.
- Memory: one note per corpus version bump (what changed, which models were reused/retrained).
- Never touch `results/head-fixed-v<N-1>` afterwards except to read.
- Next: the `train-plan` skill (what to train now).
