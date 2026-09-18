# Head-Fixed Combined Pose Estimation Dataset

Current consolidated zero-shot research recipes, comparison metrics, and recovery instructions:
[zero-shot baseline](docs/zero_shot_baseline.md).

Pipeline for merging multiple labeled datasets into a single standardized training set for Lightning Pose.

## Installation

```bash
pip install -e .
```

After that, `mouse_pose` is importable from any script without path manipulation.

---

## Workflow

### 1. Add a new raw dataset

Place the raw dataset under `_raw/<dataset-name>/` with the standard DLC layout:

```
_raw/<dataset-name>/
  labeled-data/<session>/<frame>.png
  CollectedData.csv        ← train split
  CollectedData_test.csv   ← test split
```

Some datasets require a preprocessing step to generate pseudo-labels before conversion
(see [Preprocessing](#preprocessing) below).

### 2. Convert the dataset (run once per dataset)

```bash
conda run -n pose python scripts/convert_dataset.py --dataset <dataset-name>
```

Reads `configs/datasets/<dataset-name>.yaml`. Outputs to `data/head-fixed/`:
- `CollectedData_<dataset>_train.csv`
- `CollectedData_<dataset>_test.csv`
- `labeled-data/<dataset>/<session>/<frame>.png`  (images copied)

Re-running is safe — images are skipped if already present.

Two options for data versioning (`docs/data_versioning.md`): `--link_frames` symlinks each frame to the
raw file instead of copying it (frames never change between data versions, only label CSVs do), and a
`raw_folder: <name>-v2` key in `configs/datasets/<dataset>.yaml` reads the CSVs from a re-labeled raw
folder while the dataset keeps its name. Register a changed dataset first:
`python scripts/data_manifest.py --register <dataset> --note "what changed"`.

### 3. Build a merged training set (run as needed)

```bash
# every dataset contributes all of its available train frames
conda run -n pose python scripts/build_dataset.py --tag face+ibl --datasets facemap ibl --n_frames -1
```

Outputs to `data/head-fixed/`:
- `CollectedData_<tag>_train.csv`
- `CollectedData_<tag>_test.csv`

Every tag uses `--n_frames -1` (all frames). `--n_frames <N>` caps each dataset at `N` frames
instead; it isn't part of the current design, but if you use it, frame selection is reproducible —
the same `--seed` + dataset name always selects the same frames regardless of which other datasets
share the tag (default seed 42, moot at `-1`).

Single-dataset tags need no build step: `convert_dataset.py` already writes
`CollectedData_<dataset>_{train,test}.csv` with every frame, byte-identical to what a single-dataset
`--n_frames -1` build would produce.

**See [`docs/build_dataset.md`](docs/build_dataset.md) for which tags to build and why** — in short,
n=1 (each dataset alone), n−1 (leave-one-out), and n=all, rather than all `2^n − 1` combinations.

### 4. Train

```bash
# Dry run to preview commands
conda run -n pose python scripts/train_sweep.py --dry_run \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_face+ibl+cheese+caz_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2"

# Full sweep
conda run -n pose python scripts/train_sweep.py \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_face+ibl+cheese+caz_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino"
```

`--train_frames 1` is Lightning Pose's convention for "use every frame in the CSV" — not one frame.
Since every tag already contains all available frames, that's what every run uses; `--train_frames
<N>` would subsample from the CSV at training time, which the current design doesn't do. See
[`docs/train_sweep.md`](docs/train_sweep.md) for the two-phase plan (n=1 and n=all first, then the
n−1 ablations).

Results land at `results/head-fixed/<tag>/<losses>/tf<N>/<backbone>/seed<N>/`.
Evaluation runs automatically after each model against every per-dataset test CSV.

### 5. Train on Lightning AI (parallel, optional)

`scripts/train_sweep_lightning.py` takes the exact same CLI args as `train_sweep.py` above, but
launches every combo as an independent Lightning Job instead of looping sequentially:

```bash
pip install -e ".[lightning]"

# from within a Lightning AI studio
python scripts/train_sweep_lightning.py \
    --csv_files "CollectedData_facemap_train.csv;CollectedData_face+ibl+cheese+caz_train.csv" \
    --train_frames "1" \
    --seeds "0;1;2" \
    --backbones "vits_dino" \
    --machine L4
```

Each job command begins with a preflight that aborts if `lightning_pose` resolved to a released
site-packages build instead of the local editable clone — otherwise a job could train successfully
while silently ignoring local LP changes (`--allow_stock_lp` to opt out; see `make_preflight_command`).

Both scripts share their combo generation, naming, and `litpose train` command-building via
`mouse_pose/train.py` — they only differ in *how* a command gets executed (subprocess loop vs.
`Job.run`), so there's one place to change if the sweep logic itself needs to change. Because a
Lightning Job is a fresh remote process with no way to "come back" to it afterward like the local
loop does, evaluation can't happen in-process there — instead `mouse_pose/train.py` is itself
CLI-invocable (`python -m mouse_pose.train --output_dir ... --csv_file ...`) and gets chained onto
the training command with `&&` for each job.

**Getting data onto Lightning storage:** each Job is an isolated snapshot of the launching Studio's
filesystem, and syncing the ~10k individual label/image files in `data/head-fixed_vN` is slow. Instead,
archive it once (`tar -cf head-fixed_v2.tar head-fixed_v2`) and upload just that file to the
Studio. Every job then extracts it into place itself on first use if it's not there yet — safe to do
independently per job since there's no shared filesystem to race on (see `make_extract_command` in
`mouse_pose/train.py`, and `docs/train_sweep.md` for the full setup). `paths.yaml` is machine-specific
and gitignored, so the Studio needs its own copy: `data_dir` pointing at wherever the archive lives,
`results_dir` pointing at storage that's actually persistent/shared across jobs (e.g. a
teamspace-mounted drive) — the two have different persistence needs and don't have to be on the same
kind of storage.

---

## Preprocessing

Some datasets need custom work before the standard convert step (pseudo-label
generation, pulling frames from a non-DLC source format, etc.). Each one has its own
README under `scripts/preprocessing/`.

**Before starting a new one, see [`scripts/preprocessing/README.md`](scripts/preprocessing/README.md)**
— it has the checklist of decisions (new keypoints, laterality, multi-view merging,
train/test split) that need a human call rather than an inferred default.

---

## Currently converted datasets

Frame counts are what lands in `data/head-fixed` — i.e. *after* `exclude.sessions`. The raw CSV
count is deliberately not listed: it isn't what you train on, and carrying both numbers just gives
two things to go stale. Run `convert_dataset.py` and it prints `N frames excluded` for the split.

| Dataset       | Train frames | Test frames | Notes |
|---------------|--------------|-------------|-------|
| facemap       | 2400         | 100         | cam1 (left side, 1800 frames → left eye contour) + cam0 (right side, 600 frames → right eye contour); bilateral kps lateralized via `{side}` |
| ibl           | 5962         | 1446        | wrist + pupil_center + nose_tip + tongue; human-reviewed (July 2026), supersedes `ibl-paw`. Excludes 26 legacy `ibl-paw`-era sessions (1646 frames) — see `configs/datasets/ibl.yaml` |
| cheese-2d     | 665          | 291         | four views (L/R/BC/TC); custom visibility post-processing. No exclusions |
| cazettes-side | 830          | 217         | left-view; bilateral kps lateralized via `{side}`. No exclusions |

**What `exclude` means.** `exclude.sessions` drops whole sessions from *both* splits before anything
else runs; `exclude.keypoints` drops source keypoints entirely (e.g. `cheese-2d`'s `ref(head-post)`,
a rig fiducial rather than anatomy). Exclusion is about *provenance and view*, not label quality —
excluded frames are usually labeled just fine, they simply don't belong in this dataset's identity.
Note the difference from visibility: an excluded frame is absent from the CSV, whereas an unlabeled
keypoint in a kept frame is present with `visible=1`.

---

## Architecture

### Two-step pipeline

```
_raw/<dataset>/                    data/head-fixed/
  configs/datasets/<dataset>.yaml  ──convert──▶  CollectedData_<dataset>_{train,test}.csv
  CollectedData.csv                              labeled-data/<dataset>/...
  CollectedData_test.csv
                                   ──build──▶  CollectedData_<tag>_{train,test}.csv
```

**convert** (`scripts/convert_dataset.py`) is slow (copies images) and run once per dataset.
**build** (`scripts/build_dataset.py`) is fast (CSV only) and run freely for experimentation.

### Canonical keypoint vocabulary (`configs/keypoints.yaml`)

Single source of truth for all keypoint names and their ordering. Every output CSV — per-dataset
and merged — has columns in this order. Datasets that don't label a keypoint carry `visible=0` for it.
The current count changes as datasets are added — read `configs/model.yaml`'s `data.num_keypoints`
directly rather than citing a number here or anywhere else in docs; it goes stale immediately.

### Visibility convention

| Value | Meaning |
|-------|---------|
| `2`   | Keypoint is labeled in this frame |
| `1`   | Keypoint belongs to this dataset but is unlabeled in this frame (e.g. wrong-side view) |
| `0`   | Keypoint is not part of this dataset |

Lightning Pose's loss function is visibility-aware: `vis=0` frames are excluded from loss for that
keypoint. This means single-dataset and merged models all share the same LP config 
(`configs/model.yaml`).

### `configs/datasets/<name>.yaml` format

```yaml
exclude:
  keypoints: [kp1, kp2]   # source keypoints to drop entirely
  sessions: [sess1, ...]   # sessions to exclude from both splits

keypoints:
  <source_name>: <canonical_name>         # direct rename
  <source_name>: <canonical_{side}_name>  # lateralized: expands to _left + _right

sessions:
  <session_name>: left | right | null     # drives lateralization
```

**Lateralization:** if the canonical target contains `{side}`, `convert_dataset.py` expands it into
two columns (`_left`, `_right`). The session's `side` value determines which column gets real
coordinates; the other gets `NaN` / `vis=1`.

**`null` sessions** have no dominant side. Lateral keypoints are inapplicable; only midline keypoints
get filled. Useful for head-on camera views (cheese-2d BC/TC sessions).

### Per-dataset post-processing (`convert_dataset.py`)

Some datasets need custom visibility logic beyond the standard lateralization rules. These are
implemented as functions registered in `POST_PROCESS` at the top of `convert_dataset.py`:

```python
POST_PROCESS: dict[str, Callable] = {
    "cheese-2d": _post_process_cheese2d,
}
```

Each function receives the fully-processed DataFrame and the dataset config, and returns a modified
DataFrame.

**cheese-2d specifics:** Missing labels are annotation gaps, not occlusion. Post-processing promotes
`vis=1 → vis=0` for keypoints that should be visible given the session's viewpoint.

### Directory layout

```
mouse-pose/
  configs/
    keypoints.yaml              canonical keypoint vocabulary
    model.yaml                  LP model config; data_dir/csv_file overridden per-run by train_sweep*.py
    datasets/
      <dataset>.yaml            per-dataset conversion config

  scripts/
    convert_dataset.py          per-dataset conversion (run once)
    build_dataset.py            subsampling + merging (run freely)
    train_sweep.py              LP training sweep + evaluation, local/sequential
    train_sweep_lightning.py    same sweep, Lightning AI/parallel (see mouse_pose/train.py)
    preprocessing/
      ibl-face/                 iblvideo pseudo-label pipeline


  mouse_pose/
    paths.py                    path resolution from paths.yaml
    train.py                    sweep combo/naming/command logic shared by both
                                 train_sweep*.py scripts; also a standalone CLI
                                 (`python -m mouse_pose.train`) for evaluation only
    plots/
      plot_keypoints.py         keypoint overlay visualization

poseinterface/
  _raw/
    <dataset>/
      labeled-data/<session>/<frame>.png
      CollectedData.csv
      CollectedData_test.csv

  data/head-fixed/
    labeled-data/
      <dataset>/<session>/<frame>.png
    CollectedData_<dataset>_{train,test}.csv   per-dataset (from convert)
    CollectedData_<tag>_{train,test}.csv       merged (from build)

  results/head-fixed/
    <tag>/<losses>/tf<N>/<backbone>/seed<N>/
      eval/<dataset>/
        predictions.csv
        pixel_error.csv
```

Model checkpoints (`*.ckpt`) are deleted once evaluation completes — `eval/<dataset>/` is what's kept
long-term, not the trained weights. This happens in `mouse_pose.train.evaluate_model`, so it applies
whether a run finished locally or on Lightning AI.

### Adding a new dataset

Dataset onboarding is three stages — see
[`scripts/preprocessing/README.md`](scripts/preprocessing/README.md) for the full model,
what to ask before starting, and stage 1 (convert to LP format) in detail. This section
covers stages 2 and 3, which live in this repo's shared config rather than in
`scripts/preprocessing/`.

**Stage 2 — add to the corpus.** Only once you've confirmed the dataset should actually
be merged in:
1. Add new keypoints to `configs/keypoints.yaml` and `configs/model.yaml` (keep
   `data.num_keypoints` in sync — it's a plain count, not derived from the list) if the
   dataset introduces any
2. Append `<name>` to `configs/dataset_registry.yaml` — the list position is the dataset id
   (samplers, per-dataset heads and checkpoints identify datasets by index), so append at the
   end and never reorder. It drives `scripts/build_dataset.py`'s default `--datasets` set, the
   per-dataset evaluation in `mouse_pose/train.py` and `mouse_pose.datasets.ALL_DATASETS`
   (derived from it). It isn't derived from `configs/datasets/`, so it must be updated by
   hand or the new dataset silently won't be included
3. If custom visibility logic is needed, add a function to `POST_PROCESS` in `convert_dataset.py`
4. Run `conda run -n pose python scripts/convert_dataset.py --dataset <name>`

**Stage 3 — rebuild the combined dataset, as a new data version.** A new dataset (or new
keypoints) invalidates the whole tag set: the `n=all` tag now misses a dataset and every `n−1`
leave-one-out tag omits two. Register the change, point `paths.yaml` at the next
`data/head-fixed-v<N>` and rebuild every merged tag there (`docs/data_versioning.md`,
`skills/dataset-update`); results trained on the old tags stay valid for the old version.

### Renaming or deprecating a dataset

Dataset identity is a *name*, not a single file, and it's referenced in several places that
don't cross-check each other. When renaming (e.g. `ibl-face` → `ibl`) or deprecating one
(e.g. `ibl-paw`), update all of:

1. `_raw/<old-name>/` → `_raw/<new-name>/` (physical rename; `convert_dataset.py` resolves the
   raw directory as `<raw_dir>/<dataset-name>`, so these must match)
2. `configs/datasets/<old-name>.yaml` → `configs/datasets/<new-name>.yaml`
3. `ALL_DATASETS` in `mouse_pose/datasets.py`
4. Any hardcoded raw-dir constants inside preprocessing scripts (e.g.
   `scripts/preprocessing/ibl-face/create_ibl_face_dataset.py` had `IBL_FACE_DIR` hardcoded to
   `"ibl-face"` independent of the config filename)
5. This README and any `scripts/preprocessing/*/README.md` — grep for the old name
6. Existing `data/head-fixed_vN/` and `results/head-fixed_vN/` directories are a **frozen
   historical record** of whatever name was used at build time — don't rename files inside them
   to match; instead rebuild a new version (see below) once the rename is done upstream

A pipeline preprocessing folder name (e.g. `scripts/preprocessing/ibl-face/`) can reasonably stay
as-is even after the dataset itself is renamed — it names the *process* that produces the dataset,
not the dataset's identity in `configs/datasets/` or `_raw/`.

### Dataset versioning

`paths.yaml`'s `data_dir` and `results_dir` point at unversioned working paths
(`data/head-fixed`, `results/head-fixed`) — these are scratch locations, not meant to be committed
to or read from directly. The convention is:

1. Leave `paths.yaml` pointed at the unversioned path
2. Run the full convert → build (→ train) pipeline there
3. Once it succeeds, rename the directory to freeze it: `data/head-fixed` → `data/head-fixed_vN`,
   `results/head-fixed` → `results/head-fixed_vN`

**Keep `data/head-fixed_vN` and `results/head-fixed_vN` numbering in lockstep** — results trained
against `data/head-fixed_v1` belong in `results/head-fixed_v1`, not `results/head-fixed_v2`. A
mismatch here previously caused confusion about which data version a set of results actually came
from; if you bump the data version, bump the results version to match even if you haven't trained
anything against it yet.
