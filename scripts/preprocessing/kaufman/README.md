# kaufman dataset conversion

Stage 2 complete — kaufman is in the combined corpus. See
[`configs/datasets/kaufman.yaml`](../../../configs/datasets/kaufman.yaml) for the
current keypoint mapping.

## Source format

`_raw/kaufman/` arrived already in standard DLC layout (`labeled-data/<session>/`,
`CollectedData_Hank.csv`, `config_multiview_384.yaml`, `calibration/`) — a calibrated
**multi-view** Lightning Pose project: 104 `labeled-data/<timestamp>-cam[12]/` dirs (52
session timestamps x cam1/cam2), each with its own per-session
`CollectedData_Hank.csv`, sharing one `calibration.toml`/`calibrations.csv` and a
`model_type: heatmap_multiview_transformer` config.

27 keypoints, scorer `Hank`, identical schema across all 104 sessions (verified).

## Decisions

- **Video encoding**: the 24 sample videos in `videos/` are already h264/yuv420p/mp4 —
  no re-encoding needed.
- **Single-view, by explicit request**: despite the calibrated multi-view source,
  this is being built as a **single-view** dataset — cam1 and cam2 are kept as
  independent rows/sessions, not merged into a multi-view sample. `project.yaml`
  (hand-written, keypoint names copied from `config_multiview_384.yaml`) has
  `view_names: []`.
- **Keypoints**: stage 1 carries through all 27 source keypoint names as-is (laterality
  is already baked into the source names, e.g. `LFPm` vs `RHPm`). Stage 2 keeps only
  the 4 right-forepaw digit tips (`RFPf1`-`RFPf4` → `d4_tip`-`d1_tip`, note the reversed
  digit order — see `configs/datasets/kaufman.yaml`), all already-canonical, lateralized
  to `_right` since every session only ever assessed the right forepaw; the other 23
  keypoints are excluded.
- **Train/test split**: grouped by session **timestamp**, not by individual
  labeled-data dir or by subject. Video filenames carry subject IDs (e.g. `b8sSM5`),
  but only 2 of the 52 distinct labeled-data timestamps overlap with the 24 sample
  videos, so a subject can't be recovered for most of the 52 sessions. Grouping by
  timestamp keeps a session's cam1 and cam2 views (the same trial, viewed twice) on the
  same side of the split — treating them as independent would leak near-duplicate
  frames across train/test.

  Every video-backed timestamp is *forced* into test (so every delivered video ends up
  a labeling-review candidate in `videos_test/`) — not every test session has video,
  but every video's session is in test. The remaining 10-15% target is then filled by a
  normal greedy random split over the other timestamps, via
  `mouse_pose.subject_split.subject_split` (generic despite the name — it only needs a
  `{group: count}` dict), with the *remaining* target fraction rescaled so the forced
  frames still count toward the overall 10-15% (midpoint 0.125 used for the
  not-yet-reached portion). Only 2 of 52 timestamps are video-backed (108/2540 frames,
  4.3%), so the target is reached almost entirely by the random top-up. Result: 46
  train / 6 test timestamps, 2218 / 322 frames (12.7% test).
- **`videos_test/`**: the 2 video-backed test-session videos
  (`b8sSM5_20241213-125905_cam[12]...`, `b8sSM10_20250404-151615_cam[12]...`) were
  moved from `videos/` into `_raw/kaufman/videos_test/` so they sit alongside the test
  split they belong to. The other 10 sample videos (all train-session timestamps)
  remain in `videos/`.

## Running

```bash
conda run -n pose python scripts/preprocessing/kaufman/convert_kaufman.py
```

Concatenates all 104 per-session CSVs and writes `_raw/kaufman/CollectedData.csv` /
`CollectedData_test.csv`. No image copying needed — everything already lives at its
final `_raw/kaufman/` path. Spot-checked by overlaying keypoints from the first train
row on its source image; all 27 land in anatomically correct positions (nose/tongue at
the mouth, spout keypoints on the water spout, forepaw digit keypoints clustered on the
visible paw, hindpaw and left-paw points in the right spots).

## Stage 2

`configs/datasets/kaufman.yaml` maps the 4 right-forepaw digit tips to their canonical
names and excludes everything else; all already existed in
`configs/keypoints.yaml`/`configs/model.yaml`, so no vocab changes were needed. All 104
sessions are declared `right`.

Only the right forepaw was ever filmed, so the default per-split output would mark
every `d[1-4]_tip_left` `visible=1` ("in dataset, unlabeled") rather than `visible=0`
("not part of this dataset") — training on that would teach the model to predict a
suppressed heatmap for a side that was simply never assessed. A
`POST_PROCESS["kaufman"]` entry in `scripts/convert_dataset.py` forces every `_left`
column to `visible=0` after the standard split processing (same pattern as
`hantman-mv`, simpler here since there's no ear-keypoint exemption).

`kaufman` was added to `ALL_DATASETS` in `mouse_pose/datasets.py`.

```bash
conda run -n pose python scripts/convert_dataset.py --dataset kaufman
python scripts/build_dataset.py --tag <tag> --datasets kaufman ...  # not yet run
```
