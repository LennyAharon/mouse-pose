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

## 2. Put the raw files in place, append-only

- Updated labels for an existing dataset: put the new CSVs in a NEW raw folder
  (`_raw/<dataset>-v<k>/`, next free k) with `labeled-data` symlinked to the original frames
  (`ln -s ../<dataset>/labeled-data`), and point `raw_folder:` in `configs/datasets/<dataset>.yaml`
  at it. Never overwrite the CSVs in the original raw folder.
- New dataset: `_raw/<name>/` with the standard layout.

## 3. Register the dataset version

`python scripts/data_manifest.py --register <dataset> --note "<what changed>"` appends the
version (hash of the raw CSVs, date, note, raw folder) to `poseinterface/DATASET_VERSIONS.json`.
Repeat per changed/added dataset.

## 4. Build the next corpus version

1. Next free N: `ls poseinterface/data | grep head-fixed-v`. Set `paths.yaml` `data_dir` and
   `results_dir` to `head-fixed-v<N>` (both, same N).
2. `python scripts/convert_dataset.py --dataset <ds>` for every dataset (changed and unchanged;
   unchanged ones reproduce the same CSVs — needed so the version is self-contained), then
   `python scripts/build_dataset.py` for the tags in use (`docs/build_dataset.md`).
   Frames: convert with `--link_frames` (symlink `labeled-data/<ds>` to the shared frame pool)
   once that flag exists; until then copies are acceptable.
3. `python scripts/data_manifest.py` (writes `MANIFEST.json`), then
   `python scripts/data_manifest.py --diff v<N-1>` and paste its output into
   `poseinterface/DATA_VERSIONS.md` under `## v<N> — <date>` with the user's one-line "why".

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

- `CLAUDE.md` "Phase" paragraph: current corpus version and date.
- Memory: one note per corpus version bump (what changed, which models were reused/retrained).
- Never touch `results/head-fixed-v<N-1>` afterwards except to read.
