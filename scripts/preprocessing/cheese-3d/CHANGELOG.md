# cheese-3d dataset changelog

Six-camera (L/R/TL/TR/BC/TC) multi-view mouse orofacial recordings, pseudo-labeled by
distillation from the single-view `cheese-2d` dataset and reprojected via multi-view
triangulation. Already in standard DLC layout — no custom conversion script was needed,
so this folder exists only to track keypoint-level changes to the source labels over
time. See
[`configs/datasets/cheese-3d.yaml`](../../../configs/datasets/cheese-3d.yaml)
for the current keypoint mapping.

Unlike `cheese-2d`, no `POST_PROCESS` override is registered for this dataset in
`scripts/convert_dataset.py` — every keypoint in the source CSVs is either labeled
(`visible=2`) or unlabeled/occluded (`visible=1`), so the framework default is correct
as-is.

## Changelog

### 2026-09-23 (MW)
- Started manual adjustment of ear labels.

### 2026-09-23 (MW)
- Dropped sessions from the original `cheese-3d` dataset: subjects `B31` and `B6` from
  train, and the `chew_temperature` sessions for `B32` and `B33` from test.

### 2026-09-22 (LA)
- Added pupil pseudo-labels (`pupil_center_left`/`pupil_center_right`) by running the
  fully trained MM model on the existing train/test images and keeping predictions with
  confidence >= 0.70. Candidates were visually reviewed, clear misplacements were
  refined using approximate pupil-center annotations plus local optical-flow
  propagation, and cases where reflections or occlusion prevented reliable localization
  were excluded. Appended to versioned CSV copies — existing annotations and the
  train/test split are unchanged, and the original raw data was not modified. These
  remain approximate pseudo-labels pending further manual refinement.

### Spring 2026 (LA)
- Created the `cheese-3d` dataset by distilling `cheese-2d`: an ensemble of three
  single-view transformer pose models (DINOv2-pretrained ViT-B backbone) was trained on
  665 instances labeled per-keypoint in a subset of views (depending on anatomical
  visibility), then run across full-length videos to get per-view 2D predictions.
- Predictions were filtered to require at least two views with median likelihood > 0.6
  for a keypoint to be eligible for triangulation; keypoints that never cleared this bar
  in any view for a session (from occlusion or a total lack of training labels across
  all views) were excluded from the per-frame acceptance criterion and left as `NaN`.
  A frame was kept only if every eligible keypoint was confidently predicted in at least
  two views simultaneously.
- Kept frames were triangulated into 3D using camera calibration, then reprojected back
  into all six views to produce geometrically consistent 2D pseudo-labels — including
  for cameras that lacked a confident direct prediction.
- k-means clustering on flattened per-session 3D keypoints selected ~50-55
  pose-diverse representative frames per session (cluster-centroid-nearest frame).
- A final reprojection-error quality filter (mean Euclidean distance between the
  ensemble's original 2D predictions and the triangulated/reprojected labels, averaged
  over confident keypoints/views) discarded the worst 25% of frames.
- Final train/test sets: 450/150 instances, all six views per instance, `NaN` where
  triangulation wasn't possible.
- See the appendix of Wang et al., 2026 (arXiv), "BEAST 3D: Animal behavioral analysis
  and neural encoding from multi-view video via Gaussian splatting" for full details.
