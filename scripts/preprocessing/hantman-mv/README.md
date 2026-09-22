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

## Changelog

### 2026-09-22 (MW)
- Ran `scripts/preprocessing/extract_clips.py` (see
  [`cazettes-side/README.md`](../cazettes-side/README.md) for where that script came
  from) on the four videos in `_raw/_dlc/hantman-mv/videos_test/`
  (`KPC190_20260214_v033_front`, `KPC190_20260214_v033_side`,
  `KPC200_20260323_v053_front`, `KPC200_20260323_v053_side`), saving to
  `_raw/hantman-mv/videos_test/` as labeling-candidate review clips.
- These containers report `1 fps` (5998 frames -> ffprobe reads the duration as ~100
  minutes), but the true capture rate is 500 fps (~12s actual duration) — the 1 fps tag
  is wrong metadata, not a real frame rate. This prompted two new options on
  `extract_clips.py`/`make_video_snippet`: `--from-start` (take the first
  `--clip-length` seconds instead of searching for the highest-motion window) and
  `--fps` (override the frame rate used for all time math, and pass `-r <fps>` to
  ffmpeg as an input option so it regenerates timestamps at the true rate instead of
  trusting the container's declared one).
- First tried `--clip-length 1 --skip-start 0 --from-start --fps 500` as a quick check
  (500 fps / 500 frames / 1.0s output, confirmed correct), then `--clip-length 12` for
  the full ~12s clip, before settling on `--clip-length 8` — verified via ffprobe that
  each final output clip is 500 fps / 4000 frames / 8.0s duration, h264/yuv420p/mp4.
  Raw-video review only, not a label change — `CollectedData*.csv` files are
  untouched.

### 2026-09-17 (MW)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints (all rows
  empty — ears are never visible in this view) to `project.yaml` + both
  `CollectedData*.csv` files, and mapped `ear_*: ear_*_{side}` in
  [`configs/datasets/hantman-mv.yaml`](../../../configs/datasets/hantman-mv.yaml). All
  four canonical names already existed in `configs/keypoints.yaml`/`model.yaml`.
- Added an exemption (`_HANTMAN_MV_LEFT_SUPPRESS_EXEMPT`) to `_post_process_hantman_mv`
  in `scripts/convert_dataset.py` so `ear_*_left` stays at the default `visible=1`
  (matching `ear_*_right`) instead of being swept into the blanket `_left → visible=0`
  rule described under "Stage 2" below. Without it the ear keypoints would've ended up
  asymmetric (right actively suppressed, left excluded from loss entirely), unlike the
  symmetric treatment in `cazettes-side`/`facemap`. The pre-existing `eye_*_left`/
  digit/wrist `_left` columns are unaffected and still forced to `visible=0`.

### 2026-09-15 (MW)
- Added `eye_back`, `eye_top`, `eye_front`, `eye_bottom`, `nose_tip`, and `nose_bottom`
  as new keypoints (all rows empty) to `project.yaml` + both `CollectedData*.csv`
  files — a schema extension to label via the LP app later, not new label data (no
  source ever provided values for them).
- Computed a new `wrist_new` column from three existing keypoints:
  `mid = nanmean(d2_base, d3_base)` (x/y independently), then
  `wrist_new = (mid + hand_middle) / 2` — `NaN` whenever either input is `NaN`.
  183/192 train rows and 24/30 test rows got a real value under this formula; some
  values were **manually corrected afterward**, so the CSVs no longer match the formula
  exactly row-for-row — treat it as how the column originated, not a reproducible
  derivation of its current values. Maps to `wrist_{side}` (reuses existing canonical
  `wrist_left`/`wrist_right` — no new vocab needed).

**Neither addition above is reproduced by `convert_hantman_mv.py`.** Both are manual
edits to the stage-1 output; the raw source (`_raw/_dlc/hantman-mv`) still only has the
original 17 finger/paw/pellet keypoints, and a from-scratch re-run would regenerate only
those and silently drop these additions.

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
`wrist_{side}`, already canonical — see the 2026-09-15 Changelog entry above). All 54 sessions are
declared `right` (single hand, per the raw source). Everything else — the digit
`_middle`/`_base` joints, `hand_middle`/`hand_lateral`/`hand_medial`, source `wrist`,
`pellet` — is excluded.

Only one side was ever filmed, so the default per-split output would mark every `_left`
counterpart of a lateralized keypoint `visible=1` ("in dataset, unlabeled") rather than
`visible=0` ("not part of this dataset") — training on that would teach the model to
predict a suppressed heatmap for a side that was simply never assessed. A
`POST_PROCESS["hantman-mv"]` entry in `scripts/convert_dataset.py` forces every `_left`
column to `visible=0` after the standard split processing, **except** the four
`ear_*_left` columns (see the 2026-09-17 Changelog entry above), which are
deliberately left at `visible=1` instead. Same pattern as `cheese-2d`'s post-process
function, simpler here since there's only one side/one scoring rule instead of
per-session left/right/null.

To actually run stage 2:
```bash
conda run -n pose python scripts/convert_dataset.py --dataset hantman-mv
python scripts/build_dataset.py --tag <tag> --datasets hantman-mv ...
```
