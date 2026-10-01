---
name: dataset-update
description: Run when the user says a dataset's labels (CollectedData CSVs) were updated or a new dataset was added. Registers the dataset version, builds the next corpus version, writes the manifest and changelog entry, sets up the results tree, and says which models can be reused and which must be retrained.
---

# Dataset update

The corpus changes often: label CSVs of one dataset are replaced (frames never change), or a new
dataset arrives. Every such event is a new **dataset version** and a new **corpus version**.
Details and rules: `docs/data_versioning.md`. Do these steps in order; do not skip the register
step even for a "tiny" fix, and never rebuild into the current corpus version — with one exception
the user set on 2026-09-29: when the raw labels did not change (only converter code did) and nothing
trained on the current version is worth keeping, the user may choose to rebuild it under the same
number (delete its results, rebuild the data, record the rebuild in its `DATA_VERSIONS.md` entry).
Ask; never assume.

A converter change is a data change too: per-dataset visibility rules live in `POST_PROCESS` in
`mighty_mouse/convert.py` (upstream refactor 2026-09-29; `scripts/convert_dataset.py` is only the
CLI). A change there alters converted CSVs without touching any raw CSV or `configs/datasets/*.yaml`.

## 1. Find out what changed

Ask, or read from the message: which dataset(s), where the new `CollectedData*.csv` files are,
and one sentence on what changed (relabeled keypoints, added/removed sessions, new keypoint,
new split). New dataset: also its raw folder name, camera views, and which canonical keypoints
it maps to (needs a new `configs/datasets/<name>.yaml`; see README "Adding a new dataset").

## 2. Put the raw files in place

- Updated labels for an existing dataset: the user overwrites `CollectedData*.csv` in
  `_raw/<dataset>/` (frames stay). `_raw` always holds the current labels.
- New dataset: `_raw/<name>/` with the standard layout, plus `configs/datasets/<name>.yaml` and the
  registrations in `mighty_mouse/datasets.py` (README "Adding a new dataset").

## 2b. Lab label versions (MW's scheme; upstream `skills/bump-dataset-version`)

Each raw dataset also carries the lab's own label version: `_raw/<ds>/VERSION.txt` names the version
the live CSVs represent, `_raw/<ds>/versions/CollectedData[_test]_versionN.csv` are its immutable
snapshots, and `scripts/preprocessing/<ds>/CHANGELOG.md` has a `(version N)` entry saying what changed.
MW's numbers are the dataset versions of record for the team; ours (`<ds>@v<k>`, hash-keyed) only
tie a dataset version to a corpus build. So:

- Labels arriving from MW: read the changelog to name his version N, and pass `--mw-version N` in
  step 3 if `VERSION.txt`/`versions/` did not come along (often only the live CSVs are copied). The
  entry is then UNVERIFIED; once the folder is in place, re-running `--register` (CSVs unchanged)
  verifies it byte-for-byte against `versions/` and upgrades the entry in place.
- Labels WE edit (pseudo-labels, corrections): bump MW's version first with
  `python scripts/bump_version.py <ds> --initials LA --message-file <msg.md>` (writes VERSION.txt,
  versions/, and the changelog entry), then register as below. Never edit a `versions/` file.

## 3. Register the dataset version (before building anything)

`python scripts/data_manifest.py --register <dataset> [--mw-version N] --note "<what changed>"` hashes the raw
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

0. Before building: `python -m pytest tests` must pass (pytest is not installed in `cloudspace`;
   `pip install --target <scratch>/pytestdeps pytest` and put that dir on `PYTHONPATH`). Check
   whether runs are training from this checkout (`ps aux | grep "[s]cripts/train_sweep.py"`): queued
   runs read `paths.yaml` when they launch, so do NOT repoint it while a queue is live. Build from a
   separate worktree instead (`git worktree add --detach ../mouse-pose-build <commit>`, its own
   `paths.yaml` pointing at v<N>, run every script with `PYTHONPATH=<worktree>` so `mighty_mouse`
   and `paths.yaml` resolve there), and switch the main `paths.yaml` once the queue is done.
1. Next free N: `ls poseinterface/data | grep head-fixed-v`. Set `paths.yaml` `data_dir` and
   `results_dir` to `head-fixed-v<N>` (both, same N).
2. Videos never change between versions: `ln -s ../head-fixed-v<N-1>/videos data/head-fixed-v<N>/videos`
   (and `videos_ibl_leftcam`); Lightning Pose refuses to train without a `videos` folder.
   `python scripts/sync_keypoint_configs.py --check`, then
   `python scripts/convert_dataset.py --dataset <ds> --link_frames` for every dataset (changed and unchanged;
   unchanged ones reproduce the same CSVs — needed so the version is self-contained), then
   `python scripts/build_dataset.py` for the tags in use (`docs/build_dataset.md`).
   Frames: convert with `--link_frames` (symlink `labeled-data/<ds>` to the shared frame pool)
   once that flag exists; until then copies are acceptable.
3. `python -m mighty_mouse.inventory` (writes `dataset_inventory.json`, which the manifest needs,
   and `docs/dataset_inventory.md`), then `python scripts/data_manifest.py` (writes `MANIFEST.json`), then
   `python scripts/data_manifest.py --diff v<N-1>` and paste its output into
   `poseinterface/DATA_VERSIONS.md` under `## v<N> — built <YYYY-MM-DD>` with: the user's one-line
   "why", and per changed dataset the date the labels changed (the `date` of its entry in
   `DATASET_VERSIONS.json`, i.e. when it was registered) and what changed.

## 5. Set up the results tree and the reuse decision

Create `results/head-fixed-v<N>/README.md` (data version, manifest hash, which datasets changed).
The `--diff` output lists unchanged datasets, but it compares only raw hashes and
`configs/datasets/*.yaml`: it does NOT see `POST_PROCESS` / converter-code changes (2026-09-29 it
called kaufman "unchanged" after a converter fix). Always confirm with a byte comparison of the
converted CSVs against the previous version (`cmp data/head-fixed-v<N-1>/CollectedData_<ds>_*.csv
data/head-fixed-v<N>/...`), and for a changed dataset, a cell-level diff of which columns / flags
changed. Reuse rule: **a model is reusable in v<N> only if its training CSV is byte-identical** to
the previous version's (compare the tag CSV itself; a leave-X-out tag without the changed dataset
usually is). Reusable data is not a reusable model — check the model directory actually exists.

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
- Studio `CLAUDE.md` (= `AGENTS.md`): update the "as of" corpus version and date in the "Phase"
  paragraph, then reread the whole file for any line this update made untrue (a dataset, keypoint,
  path or branch it names) and fix it. Keep counts and lists out of it: point to their source
  (`configs/keypoints.yaml`, `configs/datasets/`, `MANIFEST.json`) instead.
- Memory: one note per corpus version bump (what changed, which models were reused/retrained).
- Never touch `results/head-fixed-v<N-1>` afterwards except to read.
- Next: the `train-plan` skill (what to train now).
