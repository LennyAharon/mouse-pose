# cazettes-side dataset changelog

Side-view head-fixed mouse recordings (scorer: Behnaz), already in standard DLC
layout — no custom conversion script was needed, so this folder exists only to
track keypoint-level changes to the source labels over time. See
[`configs/datasets/cazettes-side.yaml`](../../../configs/datasets/cazettes-side.yaml)
for the current keypoint mapping.

## Changelog

### 2026-09-17 (MW)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints
  (all rows empty — ears are never visible in this side view) to
  `CollectedData.csv` + `CollectedData_test.csv`, `project.yaml`, and the
  `keypoints` mapping in
  [`configs/datasets/cazettes-side.yaml`](../../../configs/datasets/cazettes-side.yaml)
  (`ear_*: ear_*_{side}`). All four canonical names already existed in
  `configs/keypoints.yaml`/`model.yaml`, so no vocab changes were needed.

### 2026-08-07 (MW)
- Added the `pupilCenter` keypoint and manually labeled it in the Lightning Pose
  app — all 830 rows across `CollectedData.csv` + `CollectedData_test.csv` are
  labeled (no NaNs).
- Made small adjustments to the `rightPaw` and `leftPaw` labels.
- Removed the stale per-session `CollectedData_Behnaz.csv` files under
  `_raw/cazettes-side/labeled-data/<session>/` (2026-09-17) — they predated this
  change and had no `pupilCenter` column, out of sync with the top-level
  `CollectedData.csv`/`CollectedData_test.csv`.
