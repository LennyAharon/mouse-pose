# hantman-mv preprocessing

Builds `_raw/hantman-mv` (single-view) from the multi-view DLC project at
`_raw/_dlc/hantman-mv`.

**Status: stage 2 drafted, not run.** `configs/datasets/hantman-mv.yaml` exists and
validates cleanly, and `configs/keypoints.yaml`/`configs/model.yaml` already carry the
8 new canonical keypoints it needs (`d1_tip`–`d4_tip`, lateralized), but
`scripts/convert_dataset.py --dataset hantman-mv` has not been run — see
[`scripts/preprocessing/README.md`](../README.md) for what that means. Unlike `hantman`
(the older, 4-keypoint reaching dataset — see `../hantman-sleap/`), `hantman-mv` has a
fuller finger/paw/pellet(+face) skeleton and is a separate dataset entirely; nothing here
merges the two.

## Why a custom converter

The source is already standard DLC format, so no format translation is needed — but it
arrives as **two per-view CSVs** (`CollectedData_side.csv`, `CollectedData_front.csv`)
sharing an identical 17-keypoint schema, and the project's `project.yaml` declares
`view_names: [front, side]`. "Converting to single-view" means:

1. Concatenating the two CSVs — side and front session directories already encode the
   view in their names (e.g. `KPC188_20260110_v038_side` / `..._front`), so there's no
   naming collision and no remapping needed, just a row concat.
2. Writing a `project.yaml` with `view_names: []`, so the LP labeling app treats the
   result as one single-view project instead of a synchronized multi-view one.

No canonical-vocab mapping happens here — the output keypoint names are exactly the
source names (`d1_tip`, `d1_middle`, ..., `wrist`, `pellet`). That mapping is a stage-2
concern (`configs/datasets/hantman-mv.yaml`), not this script's job.

**Only CSV-referenced images are copied**, not whole `labeled-data/<session>/`
directories — each session directory on disk holds many more frames than are labeled
(unlabeled context frames around each label), so a wholesale copy would pull in far
more than what's needed (labeled-data here mixes ~2050 image files against 222 labeled
rows across both views).

## Design notes

- **Split:** subject-level (first `_`-delimited token of the session name, uppercased),
  same convention as `../hantman-sleap/`. Target is the **midpoint** of the requested
  10-15% range (0.125), not the upper edge — the greedy subject-accumulation split only
  ever overshoots its target (it stops as soon as the running total meets it), and with
  as few as ~17 subjects of uneven size, aiming at 0.15 regularly overshot past it in
  practice (one subject alone was 23% of all frames). Aiming at the midpoint leaves
  room for the overshoot while still landing in range.
- **`videos/`** is created empty in the output, matching the convention in other `_raw/`
  datasets (e.g. `kondo/`, `petersen-top/`) — the source's `videos/` has a handful of
  full session recordings (241MB) that aren't needed for LP training (only labeled
  frames are used), so they aren't copied.
- Source `labeled-data/` also contains stray `*.jpgZone.Identifier` files (Windows
  download artifacts) — ignored, not copied.

## Manual keypoint additions (2026-09-15)

`eye_back`, `eye_top`, `eye_front`, `eye_bottom`, `nose_tip`, `nose_bottom` were added
directly to `_raw/hantman-mv/project.yaml` and both `CollectedData*.csv` files as empty
(`NaN`) columns on every existing row — a schema extension to label via the LP app, not
new label data (no source provided any values for them).

**This is a manual edit to the stage-1 output, not something `convert_hantman_mv.py`
reproduces.** The raw source (`_raw/_dlc/hantman-mv`) still only has the original
17 finger/paw/pellet keypoints, and the script builds its output purely from that
source's CSV columns. Re-running the script from scratch would regenerate only those
17 and silently drop this extension — if that ever happens, redo this addition
afterward (or fold it into the script first, if these face keypoints should always be
part of the schema going forward).

## Computed keypoint: wrist_new (2026-09-15)

`wrist_new` was added directly to `_raw/hantman-mv/project.yaml` and both
`CollectedData*.csv` files as a new column, computed per-row (not sourced from the raw
DLC export, which has no such point) from three existing keypoints:

1. `mid = nanmean(d2_base, d3_base)`, x and y independently — the mean of whichever of
   the two is present; `NaN` only if both are `NaN`.
2. `wrist_new = (mid + hand_middle) / 2` — a plain average, so `wrist_new` is `NaN`
   whenever either `mid` (i.e. both `d2_base` and `d3_base`) or `hand_middle` is `NaN`.

192/192 train rows and 30/30 test rows were checked against this formula directly;
183 train rows and 24 test rows got a real value, the rest `NaN` per the rule above.

**Manually corrected afterward.** Some `wrist_new` values were hand-adjusted after this
computation (presumably where the formula placed the point somewhere visibly wrong), so
the current CSVs are not guaranteed to match the formula above exactly row-for-row — treat
the formula as how the column originated, not as a reproducible derivation of its current
values.

**Also not reproduced by `convert_hantman_mv.py`**, for the same reason as the manual
keypoint additions above — a re-run from the raw source would drop it. Maps to
`wrist_{side}` in `configs/datasets/hantman-mv.yaml` (stage 2), reusing the existing
canonical `wrist_left`/`wrist_right` — no new canonical keypoint needed for this one.

## Scripts

| Script | Env | Purpose |
|---|---|---|
| `convert_hantman_mv.py` | `pose` | Concats both view CSVs, splits, copies referenced images, rewrites `project.yaml` |

## Usage

```bash
conda run -n pose python scripts/preprocessing/hantman-mv/convert_hantman_mv.py

# different train/test split seed
conda run -n pose python scripts/preprocessing/hantman-mv/convert_hantman_mv.py --seed 1
```

## Stage 2 (drafted, not run — see status note above)

`configs/datasets/hantman-mv.yaml` keeps `d1_tip`/`d2_tip`/`d3_tip`/`d4_tip` (lateralized,
newly added to `configs/keypoints.yaml`/`configs/model.yaml`), `eye_back`/`eye_top`/
`eye_front`/`eye_bottom` (lateralized, already canonical), `nose_tip`/`nose_bottom`
(midline, already canonical), and the computed `wrist_new` (lateralized to
`wrist_{side}`, already canonical — see "Computed keypoint" above). All 54 sessions are
declared `right` (single hand, per the raw source). Everything else — the digit
`_middle`/`_base` joints, `hand_middle`/`hand_lateral`/`hand_medial`, source `wrist`,
`pellet` — is excluded.

Only one side was ever filmed, so the default per-split output would mark every `_left`
counterpart of a lateralized keypoint `visible=1` ("in dataset, unlabeled") rather than
`visible=0` ("not part of this dataset") — training on that would teach the model to
predict a suppressed heatmap for a side that was simply never assessed. A
`POST_PROCESS["hantman-mv"]` entry in `scripts/convert_dataset.py` forces every `_left`
column to `visible=0` after the standard split processing. Same pattern as `cheese-2d`'s
post-process function, simpler here since there's only one side/one scoring rule instead
of per-session left/right/null.

To actually run stage 2:
```bash
conda run -n pose python scripts/convert_dataset.py --dataset hantman-mv
python scripts/build_dataset.py --tag <tag> --datasets hantman-mv ...
```
