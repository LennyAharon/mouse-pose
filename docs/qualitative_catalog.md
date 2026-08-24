# Qualitative outputs catalog — what each plot/video shows

Index of every figure and video delivered so far, so a fresh session (or person) knows
what already exists, which model produced it, and what the visual conventions mean —
**read this before re-rendering anything**. Everything lives under
`<results_dir>/qualitative/` unless noted. Current as of 2026-08-24; append new
deliveries here (folder, model, frames, conventions, caveats) as they are made.

Models referenced below:

| shorthand | run directory (under `<results_dir>`) |
|---|---|
| **full T=2** | `finetune-exp/trunk-sharedT2/seed0` — all-5-dataset, shared head, sampling T=2. This trunk copy is the one with a kept checkpoint; its `eval/` and `image_preds/` feed most renders |
| **leave-X T=2** | `<4-dataset-tag>_train/supervised/sampling-T2/tf1/vits_dinov3/seed0` — leave-one-out models, checkpoints kept |
| **scale A/B** | `scale-exp/face+cheese-T2-{dlc,zoomout}/seed0` — cheese+facemap only, stock vs zoom-out augmentation |

Global conventions (hold everywhere unless a folder says otherwise): RED is reserved
for ground truth; a thin line from GT to its prediction is the per-keypoint error
vector; per-keypoint colors come from a fixed golden-ratio wheel (red band excluded)
that is **identical across all deliveries** — each folder carries its own `legend.png`;
`pupil_center_right` is never drawn; `*_sm.mp4` is the same video re-encoded at CRF 28
only to fit the 30 MB chat-upload limit — prefer the non-`_sm` file for talks/figures.

## Per-dataset overlay videos (full model on test frames)

- **`supermouse-sharedT2/`** — one mp4 per dataset, "classes" style: RED = GT
  (+ error line), GREEN = prediction for a keypoint that dataset labels, BLUE =
  "new" keypoint the unified model adds, short names next to markers, in-frame
  banner. Hollow marker = confidence < 0.3. Made for the advisor's "I care about the
  *new* keypoints" framing.
- **`supermouse-sharedT2-perkp/`** — same frames, "perkp" style: per-keypoint colors,
  no names, `legend.png` maps color→name. Both produced by
  `scripts/render_supermodel_videos.py` (`--style classes|perkp`).

## Inference on user-provided IBL session videos

- **`video-inference-sharedT2/`** — full T=2 run on the first 400 frames of each video
  in `<data_dir>/videos` (IBL left/right cameras). Contains the raw prediction CSVs,
  `*_labeled.mp4` (Lightning Pose's built-in renderer — silently hides conf < 0.9),
  and `*_perkp.mp4` (our renderer, all predictions drawn, per-keypoint colors).
- **`video-inference-sharedT2-conf09/`** — the `*_perkp.mp4` style re-rendered with
  conf < 0.9 predictions **hidden entirely** (user-preferred), all sessions incl. the
  later-added ones. Caveat: right-camera videos are unflipped, but training used
  flipped right views — right-camera overlays are out-of-distribution by construction.

## Zero-shot vs full-model comparisons

- **`kondo-zeroshot-vs-full/kondo_gt_full_zeroshot.mp4`** and
  **`cazettes-zeroshot-vs-full/cazettes_gt_full_zeroshot.mp4`** — three overlays per
  test frame: RED = GT, GREEN = full T=2, BLUE = leave-that-dataset-out zero-shot;
  line length = pixel error; in-frame color legend, keypoint names drawn. Visual
  companion to the supported-keypoint zero-shot tables (scoring rule in
  `docs/operations.md`).

## Cross-dataset 12-view panels (`cross-dataset-panel/`)

The 12 canonical views: ibl left, ibl right(flipped), facemap cam0/cam1, cheese
BC/L/R/TC/TL/TR, cazettes side, kondo bottom.

- **`views_raw.png`** — one raw frame per view, no overlays (the scale/domain-gap
  figure; scale differences across datasets are the point).
- **`views_labeled.png`** — same layout with GT keypoints.
- **`views_inferred.png`, `views_inferred_draw1..5.png`** — full T=2 predictions on
  random frame draws (5 repeats with different frames).
- **`views_inferred_clean_draw1..5.png`** — same, "cleaned": low-likelihood predictions
  dropped AND wrong-side predictions removed (right-side keypoints on a left-facing
  view etc., per the view-side rule; midline keypoints always kept).
- **`panel.png`** — earlier single-frame-per-dataset composite (superseded by the
  views sheets).
- **`videos-clean/`** — one mp4 per dataset (~40-60 frames): all of that dataset's
  views tiled in one grid per frame, RED = GT, GREEN = prediction, conf ≥ 0.9 only,
  wrong-side hidden.
- **`views-12panel-video/`** — 4×3 grid videos, all 12 views advancing independently,
  RED = GT + error line, per-keypoint colors, conf < 0.9 hidden, `legend.png`:
  - `views_12panel.mp4` — 120 frames/view; facemap+cheese panels topped up with
    **train** frames (marked "(train)" in-panel; those overlays are optimistic).
  - `views_12panel_long[_sm].mp4` — 300 frames/view, more train top-up.
  - `views_12panel_test[_sm].mp4` — **test frames only** (the honest one); views with
    fewer test frames loop: ibl 300, cazettes 217, kondo 120, facemap cam0 75,
    cheese 41-51, facemap cam1 25.

## Scale experiment A/B (cheese+facemap, dlc vs zoom-out aug)

- **`scale-exp-ibl-AB/frame0..5.png`** — side-by-side stills on ibl test frames:
  left = A (dlc control), right = B (zoom-out), dots = predictions conf ≥ 0.5,
  red cross = GT nose_tip (the only supported keypoint for this model pair on ibl).
- **`scale-exp-AB-video/{ibl,kondo}.mp4`** — both models overlaid on the SAME frame:
  hollow = A, filled = B, connecting line = the A→B shift (thick when > 20 px),
  red cross = GT; hidden when both models < 0.5 conf. Shows the shift distribution is
  bimodal — most points barely move, big jumps cluster on ear/pad keypoints.

## Renderers

`scripts/render_supermodel_videos.py` is the committed base renderer; everything else
was rendered by one-off scratchpad scripts (ephemeral by design — recreate from the
conventions above, or generalize into the repo script if a format recurs). Fresh
predictions for frames outside a run's `eval/` come from
`Model.from_dir(run).predict_on_label_csv(...)` and land in the run's
`image_preds/<csv_name>/predictions.csv` — the full T=2 trunk already has these for
**all five** `_train` CSVs. Always H.264 re-encode (see the pose-video skill).
