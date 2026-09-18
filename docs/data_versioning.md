# Data versioning — how dataset iterations, built data and results stay together

The corpus will change many times: the label CSVs of one dataset get replaced in `_raw` (the
frames never change), or a new dataset arrives. Two levels of versioning:

- **Dataset version** `<dataset>@v<k>`: the hash of that dataset's raw `CollectedData*.csv` files,
  registered in `poseinterface/DATASET_VERSIONS.json` with a date and a note
  (`python scripts/data_manifest.py --register <dataset> --note "..."`).
- **Corpus version** `head-fixed-v<N>`: one built-data directory + one results directory with the
  same suffix, whose `MANIFEST.json` records which dataset version each dataset was built from.

The rule that keeps this sane: **a model is reusable in a new corpus version only if every dataset
it trained on has the same dataset version.** Dedicated single-dataset models of unchanged datasets
are reused (symlinked); anything that touched a changed dataset (its dedicated model, every
leave-one-out trunk that includes it, the all-data model, few-shot cells on it) is retrained.
`data_manifest.py --diff` prints both lists. Nothing else changes: `paths.yaml` selects the corpus
version, every script already routes through it. The `dataset-update` skill (`skills/`) walks
through an update end to end.

## Layout

```
poseinterface/
  DATA_VERSIONS.md                 the changelog: one section per version (below)
  DATASET_VERSIONS.json            registry: per dataset, its versions (raw CSV hash, date, note)
  _raw/                            source datasets, ALWAYS THE CURRENT labels
    facemap/  ibl/  cheese-2d/  cazettes-side/  kondo/  <new-dataset>/
                                   updated labels: overwrite CollectedData*.csv in place (frames never change),
                                   then register; a new dataset is just a new folder
  _raw_versions/<dataset>@v<k>/    label CSVs of every registered version (written by --register), so an
                                   old corpus version can be rebuilt from raw if ever needed
  data/
    head-fixed-v1/                 = today's data/head-fixed, frozen once results exist for it
    head-fixed-v2/                 next build (convert + build with paths.yaml pointing here)
      MANIFEST.json                auto-written by scripts/data_manifest.py: per-dataset counts,
                                   sessions, keypoints, views, raw source folder + hash of its CSVs
      CollectedData_<ds>_{train,test}.csv, CollectedData_<tag>_{train,test}.csv, videos/
      labeled-data/<ds> -> shared frames (convert_dataset.py --link_frames), so a version costs MBs, not GBs
      derived/                     experiment-derived CSVs (masked, pseudo-label, replay) — not in the root
  results/
    head-fixed-v1/                 = today's results/head-fixed (cleaned, frozen)
    head-fixed-v2/
      README.md                    data version, MANIFEST hash, recipe of record, code commits
      trunks/  dedicated/  fewshot/  baselines/  probes/  qualitative/  _archive/
```

`paths.yaml` (gitignored, machine-specific) is the only switch:

```yaml
raw_dir:     /teamspace/studios/this_studio/poseinterface/_raw
data_dir:    /teamspace/studios/this_studio/poseinterface/data/head-fixed-v2
results_dir: /teamspace/studios/this_studio/poseinterface/results/head-fixed-v2
```

Every training run already copies its config (with `data_dir`) into its output folder, so a result
can always be traced to the data version it was trained on even without the README.

## Rules

1. **Version numbers are sequential integers** (`v1`, `v2`, ...). The name carries no content; the
   changelog says what changed. Don't name versions after their contents (that stops working the
   moment two things change at once).
2. **A data version is immutable once a results tree exists for it.** Any change to the built data,
   however small (one session excluded, one keypoint renamed, a fixed label), is a new version.
   Building a version is minutes; mixing results from two data states is unrecoverable.
3. **`_raw` is the current truth; versions are snapshots.** Updated labels overwrite the CSVs in
   `_raw/<dataset>/`; `--register` hashes them, snapshots them to `_raw_versions/<dataset>@v<k>/`
   and records the version. Register BEFORE building, and never build a corpus version from raw
   CSVs that are not registered (the manifest would say UNREGISTERED). (`raw_folder:` in a dataset
   config remains available if a dataset must read from a differently named raw folder.)
4. **Results never cross versions.** `results_dir` and `data_dir` always share a suffix. A model
   from v1 evaluated on v2 data is a v2 experiment: it lives in `results/head-fixed-v2/probes/`
   with the v1 checkpoint path recorded in its README.
5. **Experiment-derived CSVs go under `data/<version>/derived/`**, not the data root. (v1 has
   `*_mask-*`, `*_psl*` and replay CSVs in its root: that is why the root has 60 CSVs.
   `build_masked_csv.py`, `build_psl_replay_csv.py`, `build_replay_csv.py` and `fewshot_cell.sh`
   should write/read `derived/` from v2 on; a small change, do it when v2 is first built.)
6. **Freeze = README + manifest + changelog entry**, nothing else. No renaming of run folders.

## Making a new version (checklist)

1. Add or update raw data under `_raw/` (new folder for anything changed). Update
   `configs/datasets/*.yaml` if a dataset's raw folder, keypoint map or exclusions changed.
2. Point `paths.yaml` `data_dir`/`results_dir` at `head-fixed-v<N>`.
3. `python scripts/convert_dataset.py --dataset <each>` and `python scripts/build_dataset.py ...`
   for the tags you need (n=1 / n-1 / n=all, see `docs/build_dataset.md`).
4. `python scripts/data_manifest.py` → writes `data/head-fixed-v<N>/MANIFEST.json`.
5. `python scripts/data_manifest.py --diff v<N-1>` → prints what changed vs the previous version
   (frames, sessions, keypoints, raw hashes per dataset). Paste it into `DATA_VERSIONS.md` under a
   new heading, add one line on *why*.
6. Create `results/head-fixed-v<N>/README.md` (data version, manifest hash, recipe) and train.

## Space

Frames never change, so `convert_dataset.py --link_frames` symlinks `labeled-data/<dataset>` to the
raw frames instead of copying (v1 holds copies, 2.3 GB; every later version is label CSVs only).

## Skills (repo-level, agent-agnostic)

Skills live in `mouse-pose/skills/<name>/SKILL.md` and are symlinked into the studio's
`.claude/skills/` and `.cursor/skills/`, so Claude Code, Cursor and any other agent read the same
file. `dataset-update` runs when a dataset changes; `train-plan` reports which recipe-of-record models
exist or are missing (and runs only what the user names); `eval-suite` is the post-training
evaluation battery and the folder convention for every qualitative deliverable
(`<results_dir>/qualitative/<date>-<topic>/README.md` with front matter, indexed into
`qualitative/INDEX.md` by `scripts/qualitative_index.py`); `pose-video` renders overlays.

## v1 (done 2026-09-18)

`data/head-fixed` → `data/head-fixed-v1`, `results/head-fixed` → `results/head-fixed-v1`, with
`head-fixed` symlinks left in both places so every existing path keeps working; `paths.yaml` names
the `-v1` directories explicitly. All five datasets registered as `<dataset>@v1`.
