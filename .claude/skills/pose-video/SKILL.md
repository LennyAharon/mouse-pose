---
name: pose-video
description: Create high-quality overlay videos or figures of pose-estimation predictions and/or ground-truth labels, in any requested format — keypoint subsets, per-keypoint colors, marker sizes, names, legends, frame ranges, fps, multiview layouts. Use whenever asked to visualize pose predictions, labeled keypoint data, or prediction-vs-label comparisons, on any pose project (single- or multi-view).
---

# Pose overlay videos and figures

You are rendering keypoint overlays. Requests vary every time (colors, subsets, sizes,
layouts); the craft below stays constant. The skill is project-agnostic — a
project-specific adapter section at the end covers the repo this skill ships with, and
the same pattern (paths module + conventions doc) applies to any other project.

## Workflow

1. **Find the source of truth for what to draw.** Predictions live in DLC-style CSVs
   (see formats below), produced either by a prior evaluation or freshly by running the
   model. Ground truth lives in label CSVs. Never invent coordinates.
2. **Start from an existing renderer if the project has one**; otherwise write a small
   cv2 script in the scratchpad. For one-off format requests, copy-and-edit in the
   scratchpad — only fold a change into a repo script when it is generally reusable.
3. **Render → re-encode → spot-check → deliver.** Extract one frame with cv2, look at
   it, and verify markers/colors/legend match what was requested before handing over.

## Data formats (DLC-style, used by DeepLabCut / Lightning Pose and kin)

- Predictions CSV: MultiIndex header `(scorer, keypoint, x|y|likelihood)`; index is
  either image paths (labeled-frame evals) or frame numbers (video inference).
- Label CSV: header `(scorer, keypoint, x|y)` or `(scorer, keypoint, x|y|visible)`.
  With a `visible` column, a ground-truth point exists only where `visible == 2`
  (1 = occluded, 0 = unlabeled). Drop a stray first row when `index[0] == index.name`.
- Filter pseudo-columns from the keypoint list: `set`, `nan`, anything `Unnamed*`.
- Ask/check whether the project excludes specific channels from display or scoring
  (augmentation-only channels, deprecated keypoints); don't assume — grep the project's
  docs or config for exclusions.

## Rendering craft

- **Color design:** distinct color per keypoint via a golden-ratio hue wheel
  (`h = (offset + i * 0.618034) % 1`, high saturation/value); if red is reserved for
  ground truth, exclude the red band (`h < 0.06 or h > 0.94 → shift`). Keep the SAME
  keypoint→color map across every video/view/figure in a delivery, and emit a
  standalone `legend.png` (matplotlib swatches + names) instead of cluttering frames.
- **Ground truth vs prediction:** GT in one reserved color; draw a thin line from each
  GT point to its prediction — a visible per-keypoint error vector.
- **Uncertainty:** never silently hide low-confidence points (built-in renderers often
  drop below ~0.9 and confuse everyone). Draw them hollow/smaller below a stated
  threshold so uncertainty is visible.
- **Marker size and text:** scale to the frame; upscale small frames (shortest side
  ≳ 720 px) before drawing, multiplying coordinates by the same factor. Names, when
  requested: small font, dark outline behind the text, shortened tokens
  (`_left→_L`, `bottom→bot`). cv2 colors are **BGR**, not RGB.
- **Figure-quality stills:** for paper/advisor figures prefer matplotlib over cv2
  (dpi≥150, vector text), same color map as the videos.
- **Banner every frame** with the legend of what's shown (what red/hollow mean, conf
  threshold) plus frame id/path — videos travel without their context otherwise.

## Multiview

- Multiview projects have one CSV (or CSV column-block) and one video **per view**.
  Keep the keypoint→color map identical across views.
- Deliver per-view videos plus, when asked, a composite: resize views to a common
  height and `hconcat`/grid them per frame (pad with black when counts differ).
- View-specific keypoint sets are normal (a keypoint may be labeled in one view only);
  classify per view, not globally.

## Producing fresh predictions (when no CSV exists)

- Labeled frames: `Model.from_dir(run_dir).predict_on_label_csv(csv_file=..., data_dir=...)`.
- Video: `Model.from_dir(run_dir).predict_on_video_file(video_file=..., output_dir=...,
  compute_metrics=False, generate_labeled_video=False)` — needs a GPU and a run whose
  checkpoint still exists (many pipelines delete checkpoints after eval).
- **Only N frames wanted? Clip the video first** — far faster than predicting all of it:
  `ffmpeg -i in.mp4 -frames:v N -c:v libx264 -pix_fmt yuv420p clip.mp4`.
- Models with per-dataset heads or conditioning need a routing choice at inference
  (explicit source name, or a blind/combined mode) — check the project's model API.

## Output hygiene

- Write to a NEW descriptively-named folder under the project's results/qualitative
  area; never overwrite an existing video set.
- cv2's `mp4v` doesn't play in VS Code/browsers — always re-encode:
  `ffmpeg -y -i x.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`. If `ffmpeg` isn't on
  PATH: `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`.
- Get machine paths from the project's paths config, never hardcoded.

## Project adapter: mouse-pose (the repo this skill ships with)

- Paths: `from mouse_pose.paths import load_paths` (`data_dir`, `results_dir`);
  registry: `from mouse_pose.registry import load_registry`.
- Base renderer: `scripts/render_supermodel_videos.py`
  (`--style classes|perkp`, `--conf`, `--max_frames`, `--fps`) over a run's
  `eval/<dataset>/predictions.csv`. Blind-mode scoring: `scripts/blind_eval.py`.
- Which keypoints a dataset labels: `trainable` in `<data_dir>/dataset_inventory.json`.
- Project exclusion: `pupil_center_right` is hflip-trained only — excluded from scoring
  and usually from display (include it only if explicitly asked for all 36).
- **Catalog of every figure/video already delivered — `docs/qualitative_catalog.md`.**
  Read it before rendering: it lists what exists under `<results_dir>/qualitative/`,
  which model produced it, and the exact color/confidence conventions each delivery
  used (reuse them for consistency). Append new deliveries to it.
- Operational reference: `docs/operations.md`.
