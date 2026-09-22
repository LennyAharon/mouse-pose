# kaufman dataset conversion

**Status: stage 1 only.** `_raw/kaufman/` is a usable standalone LP project, not yet in
the combined corpus. See [`scripts/preprocessing/README.md`](../README.md) for what
stage 2 would involve; don't start it unless asked.

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
- **Keypoints**: all 27 source keypoint names carried through as-is (laterality is
  already baked into the names, e.g. `LFPm` vs `RHPm`) — no new-keypoint or laterality
  decision needed at stage 1. That mapping to the shared corpus vocabulary is a stage-2
  question, not yet done.
- **Train/test split**: grouped by session **timestamp**, not by individual
  labeled-data dir or by subject. Video filenames carry subject IDs (e.g. `b8sSM5`),
  but only 2 of the 52 distinct labeled-data timestamps overlap with the 24 sample
  videos, so a subject can't be recovered for most of the 52 sessions. Grouping by
  timestamp keeps a session's cam1 and cam2 views (the same trial, viewed twice) on the
  same side of the split — treating them as independent would leak near-duplicate
  frames across train/test. Used `mouse_pose.subject_split.subject_split` (generic
  despite the name — it only needs a `{group: count}` dict) with `TEST_FRACTION =
  0.125` (midpoint of the requested 10-15% range). Result: 46 train / 6 test
  timestamps, 2218 / 322 frames (12.7% test).

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
