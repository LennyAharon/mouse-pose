# facemap dataset changelog

Single-camera orofacial mouse recordings from the public
[Facemap dataset](https://doi.org/10.25378/janelia.23712957) (Syeda et al. 2024),
already in standard DLC layout — no custom conversion script was needed, so this
folder exists only to track keypoint-level changes to the source labels over
time. See
[`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml)
for the current keypoint mapping.

## Changelog

### 2026-09-17 (MW)
- Added `ear_top`, `ear_tip`, `ear_bottom`, and `ear_base` as new keypoints
  (all rows empty — ears are never visible in this view) to
  `CollectedData.csv` + `CollectedData_test.csv`, `project.yaml`, and the
  `keypoints` mapping in
  [`configs/datasets/facemap.yaml`](../../../configs/datasets/facemap.yaml)
  (`ear_*: ear_*_{side}`). All four canonical names already existed in
  `configs/keypoints.yaml`/`model.yaml`, so no vocab changes were needed.
