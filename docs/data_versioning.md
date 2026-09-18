# Data versioning — how dataset iterations, built data and results stay together

The corpus will change many times (new datasets, re-labeled or re-split existing ones). The rule
that keeps this sane: **one data version = one built-data directory + one results directory, with
the same version suffix, and a changelog entry.** Nothing else changes: `paths.yaml` selects the
version, every script already routes through it.

## Layout

```
poseinterface/
  DATA_VERSIONS.md                 the changelog: one section per version (below)
  _raw/                            source datasets, APPEND-ONLY
    facemap/  ibl/  cheese-2d/  cazettes-side/  kondo/      as delivered
    ibl-v2/                        a CHANGED dataset gets a new raw folder; never edit a raw folder in place
    <new-dataset>/                 a new dataset is just a new folder
  data/
    head-fixed-v1/                 = today's data/head-fixed, frozen once results exist for it
    head-fixed-v2/                 next build (convert + build with paths.yaml pointing here)
      MANIFEST.json                auto-written by scripts/data_manifest.py: per-dataset counts,
                                   sessions, keypoints, views, raw source folder + hash of its CSVs
      CollectedData_<ds>_{train,test}.csv, CollectedData_<tag>_{train,test}.csv, labeled-data/, videos/
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
3. **Raw folders are never edited in place.** A re-labeled or re-split dataset arrives as a new raw
   folder (`ibl-v2/`); `configs/datasets/<name>.yaml` says which raw folder a dataset name reads
   from. The manifest records the raw folder and a hash of its CSVs, so "which labels did v3 use"
   is answered by the manifest, not by memory.
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

`labeled-data/` is 2.3 GB per version because `convert_dataset.py` copies images from `_raw`. For
datasets that did not change between versions the images are identical, so a later optimization is
to hard-link them (`cp -al`) or symlink the unchanged `labeled-data/<dataset>` folders to the
previous version. Not needed for the first few iterations.

## What happens to v1 (proposal, needs the go-ahead)

Rename `data/head-fixed` → `data/head-fixed-v1` and `results/head-fixed` → `results/head-fixed-v1`,
leave symlinks `head-fixed → head-fixed-v1` in both places so every existing path (analysis
scripts, the other session's worktrees, the catalogs) keeps working, set `paths.yaml` to the `-v1`
names explicitly. v1's `MANIFEST.json` and `DATA_VERSIONS.md` entry are generated from the current
inventory. Nothing is copied or deleted.
