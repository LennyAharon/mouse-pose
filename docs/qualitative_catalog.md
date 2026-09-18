# Qualitative outputs catalog — what each plot/video shows

## Facemap versus Mighty Mouse, all outputs (`facemap-two-row-all-keypoints/`)

`mightymouse_vs_facemap_100frames.mp4`: all 100 canonical Facemap test frames in CSV order,
3 fps (33.3-second sampled-frame slideshow). Top: all-data T2 zoom-in/out Mighty Mouse seed0,
same saved predictions as the Facemap pilot; bottom: official pretrained Facemap (all 15 outputs).
Green = Facemap tracker repertoire, including mirrored eye counterparts; blue = additional
Mighty Mouse outputs / transfer candidates. This is repertoire membership, not exact membership
in the reduced Facemap label set. All 35 Mighty Mouse outputs except augmentation-only
`pupil_center_right` are included. Hollow below confidence 0.60; no confidence-based hiding.
Sidebars list IDs/names/confidences, with IDs on confident image markers. No GT overlays.
The all-data model's known inaccurate Facemap pupil/ear transfer is intentionally retained
because this is an inspection of the benchmark model, not a curated transfer demonstration.
Includes `preview.png`, `legend.png`, and provenance. Renderer: studio-root
`render_facemap_comparison.py`; H.264/yuv420p, 1280x1834, all 100 encoded frames verified.

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

## Augmentation inspection (`aug-inspection/`, 2026-08-25)

- **`aug_panel.png`** — the adopted training augmentation seen directly: per dataset,
  one resize-only original + 4 draws through the real `model_zoomaug.yaml` pipeline
  (hflip off for clarity), labeled keypoints overlaid (off-frame ones hidden).
  Visual verification that per-dataset zoom bounds behave (facemap shrinks hard,
  ibl barely moves), keypoints track every warp, and padding is clean black.
- **`sharpness_by_view.txt`** — Laplacian-variance sharpness per view at network
  input scale (256px). Spread ~6x but content-confounded and NOT in the feared
  direction (ibl/cazettes sharpest, facemap softest) — no blur-augmentation gap;
  axis measured and closed without a run.

## Few-shot leave-one-out 12-panel (`fewshot-lr5-tf50-12panel/`, 2026-08-27)

Same 12-view 4×3 grid layout as `views-12panel-video/`, now on the **full-finetune
few-shot models**: `fewshot-exp-lr5/<dataset>/tf50-draw0` — the leave-that-dataset-out
trunk further finetuned on 50 labeled frames of the held-out dataset (arm `trunk5` in
the fewshot-curves figure), evaluated on that dataset's own test frames. Each panel's
model is the finetune specific to its dataset (all 6 cheese-2d panels share one model,
etc.). GT is drawn **GREEN** here (not the usual red — per-delivery override), with a
thin white error line to its prediction. Keypoints are colored by transfer category
relative to each panel's dataset, not by identity:

- **ORANGE — supported**: one of this dataset's own eval keypoints that at least one
  of the other 4 datasets also teaches during leave-one-out pretraining (zero-shot
  transfer target, sharpened by the 50-frame finetune).
- **MAGENTA — unsupported**: one of this dataset's own eval keypoints that NO other
  dataset teaches — only the 50 finetune frames supervise it, no pretraining transfer
  to lean on.
- **CYAN — rest**: a keypoint this dataset has no ground truth for at all (the shared
  36-kp head still predicts it; shown unscored, for a purely qualitative read on
  whether it lands somewhere plausible).

Per-dataset supported/unsupported/rest counts (of 35, excluding `pupil_center_right`):
facemap 7/1/27, ibl 4/2/29, cheese-2d 16/11/8, cazettes-side 7/1/27, kondo 11/0/24.

- `fewshot_tf50_12panel_allconf.mp4` — every prediction drawn regardless of confidence.
- `fewshot_tf50_12panel_conf09.mp4` — predictions with likelihood < 0.9 hidden entirely.
- `legend.png` — color key.
- 60 frames/view (evenly sampled from that view's test set, looped if shorter), 3 fps.
  Rendered by scratchpad `render_fewshot_12panel.py` (not folded into the repo — recreate
  from the conventions above if needed again).

## Few-shot leave-one-out 12-panel — DINO arm (`fewshot-dino-tf50-12panel/`, 2026-08-27)

Same layout, coloring, and script as `fewshot-lr5-tf50-12panel/` above, but sourced from
`fewshot-exp-dino/<dataset>/tf50-draw0` — the DINOv3-backbone-from-scratch arm (BLUE in
the fewshot-curves figure) instead of the leave-one-out-trunk full-finetune arm. Same
per-dataset supported/unsupported/rest counts (the categorization only depends on the
leave-one-out dataset composition, not the model). `fewshot_dino_tf50_12panel_allconf.mp4`,
`fewshot_dino_tf50_12panel_conf09.mp4`, `legend.png`.

## Leave-one-out 12-panel — zero-shot, no finetuning (`zeroshot-looT2-12panel/`, 2026-08-27)

Same layout/coloring again, this time on the **un-finetuned** leave-one-out trunks
(`<4-dataset-tag>_train/supervised/sampling-T2/tf1/vits_dinov3/seed0`, the "leave-X T2"
row in the models table at the top of this doc) evaluated directly on the held-out
dataset's test frames — no 50-frame adaptation step at all. This is the zero-shot
transfer baseline the two few-shot deliveries above are measured against: visibly
longer white GT→prediction error lines and more scattered ORANGE (supported) points,
since those keypoints have only ever been seen through cross-dataset transfer, never
through the target dataset's own frames. `zeroshot_looT2_12panel_allconf.mp4`,
`zeroshot_looT2_12panel_conf09.mp4`, `legend.png`.

Both deliveries rendered by scratchpad `render_12panel_lib.py` (shared with the lr5
delivery) + `run_dino_and_zeroshot.py` — not folded into the repo.

## pupil_center on facemap: 6-model comparison (`pupil-center-facemap-zeroshot-vs-finetune/`, 2026-08-27)

Narrow follow-up to the 12-panel deliveries: **both pupil_center keypoints**
(`pupil_center_left` CYAN, `pupil_center_right` YELLOW), **one dataset** (facemap),
2×6 grid = {cam0, cam1} × {zero-shot, finetune tf10-draw0, finetune tf50-draw0
(both `fewshot-exp-lr5`), LoRA r16 tf50-draw0 (`fewshot-exp-lora-r16-lr5e-5-head5e-4`,
the recommended few-shot recipe), anchor-LoRA-video tf10-draw0
(`fewshot-exp-anchor-lora-video-conf1` — video-anchored LoRA, confidence-weighted;
only tf10-draw0 exists on disk for facemap, no tf25/tf50), all-datasets (shared-head
T2 trunk trained on all 5 datasets — unlike the other columns it HAS seen facemap
frames during pretraining, just never facemap pupil_center labels)}. **Every column
reads from that run's `eval/facemap/predictions.csv`** — the held-out facemap test
set, never used for training by any of these runs (an earlier version of this
delivery had a 3rd row of predictions on the actual finetune *training* frames;
removed per user request to keep this strictly an evaluation-set comparison). No
confidence filter (every prediction drawn, likelihood printed next to each marker).

facemap labels neither keypoint at all — checked against `dataset_inventory.json`,
confirmed not in facemap's `eval` or `trainable` lists — so there is **no ground
truth to draw**; this is a purely qualitative "does the predicted point land inside
the visible pupil" check. `pupil_center_left` is a genuine cross-dataset-transfer
target (taught by the other 4 datasets); `pupil_center_right` is the one vocabulary
slot no dataset labels at all (hflip-augmentation signal only) — shown at the user's
explicit request despite the project's usual exclusion of it from scoring/display.
`_left` lands on the visible pupil across models with real confidence; `_right` sits
off in the fur at ~0.00 likelihood every frame, as expected for a keypoint with no
real supervision.

100 steps @ 4fps: all 75 cam0 test frames, cam1's 25 test frames looped to fill.
`pupil_center_facemap_6models.mp4` (47 MB; `..._sm.mp4` is the CRF-28 re-encode
delivered in-chat, prefer the full file for talks/figures). Rendered by scratchpad
`render_pupil_facemap.py`.

## ibl / cheese-2d: 5-method tf10 grid (`{ibl,cheese-2d}-5model-tf10-grid/`, 2026-08-27)

One video per dataset (ibl, cheese-2d), rows = that dataset's own camera views
(ibl: left/right; cheese-2d: BC/L/R/TC/TL/TR), columns = 5 few-shot methods, all
tf10-draw0, all leave-that-dataset-out pretraining, all evaluated on the dataset's own
held-out test set (`eval/<dataset>/predictions.csv` per run — never training frames):

| column | run |
|---|---|
| zero-shot | LOO T2 trunk, no finetune (`<4-dataset-tag>_train/supervised/sampling-T2/...`) |
| dino tf10 | DINOv3-from-scratch 10-frame arm (`fewshot-exp-dino`) |
| full-FT tf10 | full-finetune 10-frame arm (`fewshot-exp-lr5`) |
| LoRA tf10 | LoRA r16 10-frame arm (`fewshot-exp-lora-r16-lr5e-5-head5e-4`) |
| anchor-LoRA tf10 | confidence-weighted anchored LoRA 10-frame arm (`fewshot-exp-anchor-lora-conf1` — user picked this over the uniform-weighting `fewshot-exp-anchor-lora`, since `finetune_methods.html` flags conf-weighting as the fix for uniform anchoring underperforming plain LoRA on ibl) |

Same color/marker conventions as the 12-panel deliveries: GREEN=ground truth (+ white
error line), ORANGE=supported keypoint (taught by ≥1 of the other 4 LOO datasets),
MAGENTA=unsupported (only the 10 finetune frames teach it), CYAN=rest (this dataset
has no GT for it at all). No confidence filter — every prediction drawn, hollow when
likelihood < 0.9. ibl: supported=4 unsupported=2 rest=29; cheese-2d: supported=16
unsupported=11 rest=8 (same categories as the earlier 12-panel deliveries, since they
depend only on LOO composition). 60 (ibl) / 51 (cheese-2d) frames/view @ 3fps.
`ibl_5model_tf10.mp4`, `cheese2d_5model_tf10.mp4`, `legend.png` per folder. Rendered
by scratchpad `render_5model_grid.py`.

## facemap zero-shot: old zoom-out-only vs new zoom-in+out augmentation (`facemap-zeroshot-zoomaug-vs-zoominout/`, 2026-08-28)

2×2 grid: rows = facemap's own views (cam0, cam1), columns = the SAME leave-facemap-out
T2 trunk recipe, differing only in pretraining augmentation config — no finetuning,
pure zero-shot:

- **zoom-out only (old)**: `model_zoomaug.yaml` — per-dataset `CropAndPad` percent is
  `(-0.15, <dataset upper bound>)`; every source zooms OUT toward the corpus's
  smallest apparent scale (ibl) but never zooms further in.
- **zoom-in + out (new)**: `model_zoominout.yaml` — per-dataset `CropAndPad` percent is
  `[<dataset lower bound>, <dataset upper bound>]`; every source ALSO reaches the
  corpus's LARGEST apparent scale (facemap's own eye crop), not just the smallest.
  Requires `lightning-pose-dev >= 44b69c8` (pair support).

Run dirs: `zoom-aug-exp/ibl+cheese+caz+kondo-T2-{zoomaug,zoominout}/seed0` (both leave
facemap entirely out of pretraining). Only facemap has both variants trained and
evaluated on disk. Same conventions as the 12-panel/5-model deliveries: GREEN=ground
truth (+ white error line), ORANGE=supported, MAGENTA=unsupported, CYAN=rest
(supported=7 unsupported=1 rest=27 for facemap). No confidence filter, hollow when
likelihood < 0.9. **Result**: the new zoom-in+out model's eye-region keypoints land
inside the eye with short error lines; the old zoom-out-only model's same keypoints
land near the nose with long error lines — a clear qualitative win for bidirectional
zoom on facemap's out-of-corpus-scale eye. 60 (cam0) / 25 (cam1, looped) frames @
3fps. `facemap_zeroshot_zoomaug_vs_zoominout.mp4`, `legend.png`. Rendered by
scratchpad `render_facemap_zoomcompare.py`.

**pupil_center_left-only variant**: same 2×2 grid, restricted to just
`pupil_center_left` (crosshair + likelihood, no other keypoints, no
`pupil_center_right`) per user request — facemap has no ground truth for it either
way, so this isolates the transfer story to one keypoint. Confidence jumps
substantially on the new zoom-in+out model (cam0: 0.01→0.92, cam1: 0.54→0.82) and the
point visibly centers on the pupil rather than sitting at its edge.
`facemap_zeroshot_zoomaug_vs_zoominout_pupilcenterleft.mp4`. Rendered by scratchpad
`render_facemap_zoomcompare_pupil.py`.

## facemap, new zoom-in+out trunk: zero-shot vs LoRA vs anchor-LoRA vs anchor-LoRA-video (`facemap-zio-4model/`, 2026-08-28)

2×4 grid: rows = facemap's own views (cam0, cam1), columns = 4 methods that all trace
back to the SAME `model_zoominout.yaml` ("-zio") pretraining lineage (as opposed to
the earlier deliveries, which used the old zoom-out-only trunk):

| column | run |
|---|---|
| zero-shot | `zoom-aug-exp/ibl+cheese+caz+kondo-T2-zoominout/seed0` (no finetune) |
| LoRA tf10 | `fewshot-exp-lora-r16-lr5e-5-head5e-4-zio/facemap/tf10-draw0` |
| anchor-LoRA tf10 | `fewshot-exp-anchor-lora-conf1-zio/facemap/tf10-draw0` |
| anchor-LoRA-video tf10 | `fewshot-exp-anchor-lora-video-conf1-zio/facemap/tf10-draw0` |

All four LoRA/anchor variants are finetuned FROM the zoominout zero-shot checkpoint
(confirmed via each run's `config.yaml` `model.checkpoint` path) — this is a clean
"same augmentation lineage, different finetune method" comparison, distinct from the
earlier `-zio`-less deliveries which forked off the old zoomaug trunk.

Per user request: shows only `pupil_center_left` (CYAN crosshair + likelihood — no
GT, pure transfer target) plus facemap's own **supported** (ORANGE) and
**unsupported** (MAGENTA) keypoints (GT in GREEN + white error line); the "rest"
category (27 unscored keypoints) is dropped to keep the frame legible. facemap:
supported=7 (`eye_back_left, eye_bottom_left, eye_front_left, eye_top_left, lowerlip,
nose_tip, nose_top`), unsupported=1 (`mouth`). No confidence filter; hollow marker
when likelihood < 0.9.

**Result**: supported keypoints land accurately across all four columns (error lines
barely visible). `pupil_center_left` confidence is where the columns diverge sharply —
zero-shot/anchor-LoRA/anchor-LoRA-video all land confidently on the pupil (0.82–0.93),
but plain LoRA's confidence craters (0.05, 0.08) despite landing in roughly the right
place — a concrete qualitative case for anchoring: LoRA's few-shot adaptation seems to
suppress confidence on transfer-only keypoints that anchoring (teacher supervision on
trunk-known channels) preserves. 60 (cam0) / 25 (cam1, looped) frames @ 3fps.
`facemap_zio_4model.mp4`, `legend.png`. Rendered by scratchpad
`render_facemap_zio_4model.py`.

## Full model, new-keypoints-only 12-panel (`fullmodel-newkps-12panel/`, 2026-09-01)

Same 12-canonical-view 4×3 grid layout as `views-12panel-video/`, but a single model
this time — `zoom-aug-exp/face+ibl+cheese+caz+kondo-T2-zoominout/seed0` (the all-5-dataset
trunk, new zoom-in+out augmentation lineage) — and each panel shows **only the
keypoints that dataset does NOT itself label**, generalizing the old
`supermouse-sharedT2/` "BLUE = new keypoint" convention to per-keypoint colors and
every dataset/view at once. "New" is defined purely by label availability (ALL_KPS −
that dataset's own eval set), independent of the model — same definition as the
"rest" category in the earlier 3-way/4-model deliveries. No GT is drawn (these
keypoints have none, by construction). Per-keypoint colors from a fixed golden-ratio
hue wheel (same convention as `supermouse-sharedT2-perkp/`) — a keypoint's color is
identical in every panel it appears in; `legend.png` maps all 35 names. Hollow marker
when likelihood < 0.9; nothing hidden. Frame identifier on every panel (`step i/n |
filename`, per the qualitative-video-conventions memory).

New-keypoint counts per dataset: facemap 27, ibl 29, cheese-2d 8 (cheese-2d already
labels most of the 36-keypoint vocabulary itself, hence far fewer dots per panel),
cazettes-side 27, kondo 24.

**No ensemble-variance ellipses this round** — user asked for them "if possible";
checked and only 1 seed exists for this zoom-in+out trunk (`seed0`), so no ensemble
is computable. The only all-5-dataset trunk on disk with a real 3-seed ensemble
(`face+ibl+cheese+caz+kondo_train/supervised/tf1/vits_dinov3/seed{0,1,2}`) is a
different recipe entirely (stock DLC augmentation, no zoom, no T2 sampling) — user
chose to keep the zoom-in+out trunk for consistency with the rest of this delivery
set over switching recipes for the ellipses. Revisit if a second/third seed of the
zoom-in+out trunk ever gets trained.

60 frames/view (evenly sampled, looped if a view has fewer) @ 3fps.
`fullmodel_newkps_12panel.mp4`, `legend.png`. Rendered by scratchpad
`render_fullmodel_newkps_12panel.py`.

## cazettes-side individual model vs full model, wrist_left + tongue (`cazettes-side-individual-vs-full-wristtongue/`, 2026-08-31)

1×2 grid on cazettes-side's held-out test frames (never trained on by either model),
focused on 3 of cazettes-side's OWN labeled keypoints — `wrist_left`, `tongue_tip`,
`tongue_center` — all with real GT both ways, unlike the earlier transfer-focused
deliveries:

| column | run |
|---|---|
| individual cazettes model (n=1) | `cazettes-side_train/supervised/tf1/vits_dinov3/seed0` — single-dataset, stock DLC augmentation (confirmed via `config.yaml`: no `dataset_names`/zoom config) |
| full model (zoom-in+out, 5 datasets) | `zoom-aug-exp/face+ibl+cheese+caz+kondo-T2-zoominout/seed0` — same all-5-dataset trunk used in the 4-model deliveries above |

GT GREEN + white error line; predictions per-keypoint colored (`wrist_left` ORANGE,
`tongue_tip` MAGENTA, `tongue_center` CYAN). No confidence filter; hollow marker when
likelihood < 0.9. **Result**: `wrist_left` is essentially perfect (1.00 conf, GT
overlap) in both models — no surprise, it's a large, easy landmark. The tongue
keypoints tell a different story: the individual model reliably nails `tongue_tip`/
`tongue_center`, while the full 5-dataset model's `tongue_center` confidence
frequently collapses toward 0.00 even on the same frame — a case where joint
training on 5 datasets appears to cost this dataset something on a fine, easily
confused landmark, opposite of the transfer wins seen elsewhere in this catalog.
150 frames (of cazettes-side's 217-frame test set, evenly subsampled) @ 3fps.
`cazettes_individual_vs_full_wristtongue.mp4`, `legend.png`. Rendered by scratchpad
`render_cazettes_ood_wristtongue.py`.

## facemap/ibl/cazettes-side/kondo 4-model transfer, +full-model column (2026-08-31)

Added a 4th column to all four 3-way deliveries below: **full model (all 5
datasets)** — `zoom-aug-exp/face+ibl+cheese+caz+kondo-T2-zoominout/seed0`, the
all-5-dataset shared-head T2 trunk on the SAME zoom-in+out augmentation lineage as
columns 2–3, so it's an apples-to-apples 4th point of comparison, not a different
recipe. Unlike the first 3 columns this model DOES see the target dataset's own
frames during pretraining (just never that dataset's missing-keypoint labels, e.g.
still no facemap `pupil_center_left` labels) — the in-domain-images-but-no-label
case, distinct from the leave-one-out columns' zero-label-and-zero-image case.
`facemap_4model_transfer.mp4`, `ibl_4model_transfer.mp4`, `cazettes_4model_transfer.mp4`,
`kondo_4model_transfer.mp4` in their respective folders (same rendering scripts, now
with 4 `MODEL_RUNS` entries). Result: full-model pupil confidence on facemap is the
highest of all four columns (0.94–0.96 vs anchor-LoRA's 0.43–0.83) — in-domain images
alone, even without direct supervision, help substantially. kondo's wrist/nose
predictions also visibly tighten from column 1→4, following the same pattern despite
kondo's weaker overall transfer picture (see the kondo 3-way note below).

**Frame identifier added to every panel** (bottom bar, all four `_4model_transfer.mp4`
files): `step <i>/<n>  |  <source filename>` — `<i>/<n>` is that panel's position in
its own (possibly looped) frame sequence, `<source filename>` is the actual image name
under `labeled-data/<dataset>/...` for full traceability back to a specific frame.

## facemap 3-way leave-one-out transfer, with bonus never-labeled keypoints (`facemap-3way-transfer/`, 2026-08-31)

2×3 grid: rows = facemap's own views (cam0, cam1), columns = zero-shot (old
zoom-out-only aug, `zoom-aug-exp/ibl+cheese+caz+kondo-T2-zoomaug/seed0`) | zero-shot
(new zoom-in+out aug, `.../−T2-zoominout/seed0`) | anchor-LoRA tf10 finetuned from the
zoom-in+out checkpoint (`fewshot-exp-anchor-lora-conf1-zio/facemap/tf10-draw0`) — all
three leave facemap entirely out of pretraining. Shows facemap's supported (ORANGE) /
unsupported (MAGENTA) keypoints with GT (GREEN + white error line), `pupil_center_left`
(CYAN crosshair, no GT), and **two bonus transfer keypoints facemap never labels at
all**: `nose_bottom` and `pad_center` (YELLOW diamond, no GT).

These two were picked by inspection, not guessed: a diagnostic scatter of all 27
"rest" keypoints on a sample frame (scratchpad `diag_zoomaug.png` / `diag_zoominout.png`)
plus a mean-confidence check over the full facemap test set showed `pad_center`
(mean conf 0.82–0.87, landing right at the visible whisker-pad edge) and `nose_bottom`
(mean conf 0.92–0.93, landing just under the nose bridge) are consistently confident
AND anatomically correct across both augmentation variants — real transfer, not noise.
By contrast `ear_*`/`tongue_*`/`wrist_*` sit at ~0.00 confidence throughout (correctly
signaling "not visible in this crop", not transfer). `pad_side_left`/`pad_top_left`/
`upperlip_left` were mid-confidence (0.15–0.78, inconsistent across the two aug
variants) and were left out to keep the frame legible.

No confidence filter on the drawn keypoints; hollow marker when likelihood < 0.9 (for
the supported/unsupported category only). 60 (cam0) / 25 (cam1, looped) frames @
3fps. `facemap_3way_transfer.mp4`, `legend.png`. Rendered by scratchpad
`render_facemap_3way_transfer.py`.

**tf50 variant** (`facemap_3way_transfer_tf50.mp4`): same 3 columns, 3rd column
swapped for `fewshot-exp-anchor-lora-conf1-zio/facemap/tf50-draw0`. Notable: unlike
tf10 anchor-LoRA (pupil confidence 0.67–0.87, matching zero-shot), **tf50 anchor-LoRA's
`pupil_center_left` confidence drops to 0.05–0.13** — the same collapse seen earlier
with plain LoRA (see the `facemap-zio-4model/` delivery above) — even though the point
still lands in roughly the right place. Worth a closer look: more finetune data seems
to erode confidence on this transfer-only keypoint despite anchoring, at least at
N=50. `pad_center`/`nose_bottom` stay confident throughout.

## ibl 3-way leave-one-out transfer, with bonus never-labeled keypoints (`ibl-3way-transfer/`, 2026-08-31)

Same structure as the facemap 3-way delivery above, adapted for ibl: 2×3 grid, rows =
ibl's own views (left, right), columns = zero-shot (old zoom-out aug,
`zoom-aug-exp/face+cheese+caz+kondo-T2-zoomaug/seed0`) | zero-shot (new zoom-in+out
aug, `.../−T2-zoominout/seed0`) | anchor-LoRA tf10 finetuned from the zoom-in+out
checkpoint (`fewshot-exp-anchor-lora-conf1-zio/ibl/tf10-draw0`, checkpoint lineage
confirmed via `config.yaml`) — all three leave ibl entirely out of pretraining.

ibl's own eval set: supported = `nose_tip, pupil_center_left, wrist_left, wrist_right`
(note `pupil_center_left` has REAL ibl ground truth here, unlike facemap where it was
a pure transfer target — no special crosshair treatment needed); unsupported =
`tongue_end_left, tongue_end_right`. GT in GREEN + white error line, ORANGE/MAGENTA
prediction dots.

**Bonus transfer keypoints — much richer than facemap.** ibl's wider face-on camera
shows almost the entire left side of the face, so the same diagnostic scatter +
mean-confidence check (scratchpad `diag_ibl_zoomaug.png` / `diag_ibl_zoominout.png`)
found **12** "rest" keypoints (ibl labels none of them) at ≥0.9 mean confidence in
BOTH augmentation variants: `eye_back_left, eye_bottom_left, eye_front_left,
eye_top_left, lowerlip, mouth, nose_bottom, nose_top, pad_center, pad_side_left,
pad_top_left, upperlip_left`. Together they trace the whole visible face outline —
drawn as small unlabeled YELLOW diamonds (≥0.5 conf only; no per-point text, 12
labels would clutter the frame) so the pattern reads as a silhouette rather than a
list. `ear_*`, `pad_top_right`, `upperlip_right`, `tongue_center`, `tongue_tip` were
mid-to-low confidence (0.35–0.66, or the wrong side of the face for this camera
angle) and left out.

60 (left, capped) / 60 (right, capped) frames @ 3fps (ibl's actual test set is 735/711
frames per view — evenly subsampled to 60). `ibl_3way_transfer.mp4`, `legend.png`.
Rendered by scratchpad `render_ibl_3way_transfer.py`.

**tf50 variant** (`ibl_3way_transfer_tf50.mp4`): same 3 columns, 3rd column swapped
for `fewshot-exp-anchor-lora-conf1-zio/ibl/tf50-draw0`. The bonus-keypoint face
outline (yellow) still traces cleanly at N=50 — unlike facemap's `pupil_center_left`
collapse (see the facemap tf50 note above), ibl's `pupil_center_left` is a supported
keypoint with real ibl ground truth, so it isn't the transfer-only case that showed
the confidence drop.

## kondo 3-way leave-one-out transfer, with bonus never-labeled keypoints (`kondo-3way-transfer/`, 2026-08-31)

Same structure as the facemap/ibl 3-way deliveries above, adapted for kondo: 1×3 grid
(kondo has only one camera view, "bottom" — a from-above/behind body shot of the
mouse gripping a bar), columns = zero-shot (old zoom-out aug,
`zoom-aug-exp/face+ibl+cheese+caz-T2-zoomaug/seed0`) | zero-shot (new zoom-in+out aug,
`.../−T2-zoominout/seed0`) | anchor-LoRA **tf50** finetuned from the zoom-in+out
checkpoint (`fewshot-exp-anchor-lora-conf1-zio/kondo/tf50-draw0`, checkpoint lineage
confirmed via `config.yaml`) — all three leave kondo entirely out of pretraining.

kondo's own eval set (11 keypoints: `ear_bottom_right, ear_tip_right, eye_back_right,
eye_front_right, lowerlip, nose_bottom, nose_top, nose_tip, tongue_tip, wrist_left,
wrist_right`) is **entirely supported** (unsupported=0) — every keypoint kondo scores
is also taught by at least one other dataset.

**Bonus transfer keypoints — more modest than facemap/ibl.** kondo's camera shows
mostly the mouse's back/body, not a face close-up, so much less of the face is
visible at all. A diagnostic scatter (scratchpad `diag_kondo_zoomaug.png` /
`diag_kondo_zoominout.png`) plus mean-confidence check over the full kondo test set
found only 4 "rest" keypoints (kondo never labels any of them) with real,
visually-plausible signal, and none reach the 0.9+ bar facemap/ibl's best candidates
did: `eye_top_right` (0.58–0.67), `eye_bottom_right` (0.68–0.72), `pad_top_right`
(0.59–0.68), `pad_center` (0.47–0.67, the weakest and least consistent of the four
between augmentation variants, kept because it still landed in the right spot on
inspection). Everything else — `ear_base_right`, `ear_top_left/right`, `mouth`,
`upperlip_*`, `tongue_center/end_*`, `pupil_center_left` — sat near 0.00, genuinely
not visible from this angle rather than a transfer failure. Drawn as YELLOW diamonds
at a lower ≥0.4 confidence bar than facemap/ibl (0.5), since kondo's best signal here
is weaker to begin with — worth knowing before reading too much into any single
diamond.

100 frames (of kondo's 120-frame test set, evenly subsampled) @ 3fps.
`kondo_3way_transfer.mp4`, `legend.png`. Rendered by scratchpad
`render_kondo_3way_transfer.py`.

## cazettes-side 3-way leave-one-out transfer, with bonus never-labeled keypoints (`cazettes-side-3way-transfer/`, 2026-08-31)

Same structure as the facemap/ibl/kondo 3-way deliveries above, adapted for
cazettes-side: 1×3 grid (one view, "side"), columns = zero-shot (old zoom-out aug,
`zoom-aug-exp/face+ibl+cheese+kondo-T2-zoomaug/seed0`) | zero-shot (new zoom-in+out
aug, `.../−T2-zoominout/seed0`) | anchor-LoRA tf50 finetuned from the zoom-in+out
checkpoint (`fewshot-exp-anchor-lora-conf1-zio/cazettes-side/tf50-draw0`, lineage
confirmed via `config.yaml`) — all three leave cazettes-side entirely out of
pretraining.

cazettes-side's own eval set: supported = `lowerlip, nose_bottom, nose_tip,
pupil_center_left, tongue_tip, wrist_left, wrist_right` (7); unsupported =
`tongue_center` (1).

**Bonus transfer keypoints — as strong as ibl.** cazettes-side's camera is a
face-on side view similar to ibl's framing, and the transfer picture matches: a
diagnostic scatter (scratchpad `diag_caz_zoomaug.png` / `diag_caz_zoominout.png`)
plus mean-confidence check over the full test set found `eye_top_left,
eye_front_left, eye_back_left, eye_bottom_left` all at **0.99–1.00** mean confidence
in both augmentation variants — the model traces a near-perfect eye ring around a
keypoint cazettes-side never labels at all — plus `nose_top` (0.99–1.00),
`pad_center` (0.51–0.66), and `upperlip_left` (0.62–0.72). `ear_tip_left`,
`ear_bottom_left`, `pad_top_left` were borderline/inconsistent (0.4–0.73 across the
two variants) and left out; the right side of the face and `mouth`/`tongue_end_*`
stayed low since the camera only shows the left side.

100 frames (of cazettes-side's 217-frame test set, evenly subsampled) @ 3fps.
`cazettes_3way_transfer.mp4`, `legend.png`. Rendered by scratchpad
`render_cazettes_3way_transfer.py`.

## Few-shot curves (`fewshot-curves/`, 2026-08-26)

- **`fewshot_curves.png`** — the headline few-shot figure: 2 rows × 5 datasets. x = labeled
  frames N ∈ {10, 25, 50} (log axis), y = pooled mean pixel error over the held-out dataset's
  test (frame, keypoint) cells. Top row scores the *supported* keypoint set (zero-shot rule),
  bottom row every keypoint the dataset labels. BLUE = DINOv3 backbone + N labels
  (`fewshot-exp-dino/`), ORANGE = super-mouse leave-that-dataset-out trunk + the SAME N
  frames (`fewshot-exp-lr5/`, arm `trunk5`) — identical protocol, only the initialization
  differs; GREEN = the same trunk adapted with LoRA r16 (adapters 5e-5, head 5e-4, backbone
  frozen; `fewshot-exp-lora-r16-lr5e-5-head5e-4/`), the recommended few-shot recipe. Solid lines = frame draw 0; hollow markers = draws 1–2 (frame-draw spread, one
  model seed). Dashed = zero-shot n−1 trunk; dotted = dedicated model on all labels.
  `pupil_center_right` excluded. Regenerate with scratchpad `fewshot_plot.py`; numbers from
  `fewshot_score.py`. The lr-1e-5 trunk arm (`fewshot-exp/`) is deliberately not drawn (it
  cannot learn novel keypoints — see plan §15).

## Few-shot model comparison videos (31 Aug 2026)

- **`fewshot-model-comparison/`** — side-by-side composites on 300-frame test clips,
  per-keypoint colors (global wheel, `legend.png`), hollow = conf < 0.3, banner per panel:
  - `facemap_D3_comparison.mp4` — 2×2: zero-shot zoom-out trunk / zero-shot zoom-in/out trunk /
    plain LoRA (N=50, zio) / anchored LoRA (N=50, zio) on `cam0_D3` (unlabeled test video).
    Shows the pupil appearing correctly only in the zoom-in/out trunk and surviving fine-tuning
    only under anchoring.
  - `cheese_L_comparison.mp4`, `cheese_BC_comparison.mp4` — 1×3 (zero-shot zio trunk / LoRA /
    anchored, N=50) on the side and bottom cameras of `20231031_chew_record`.
  - `clips/` holds the ffmpeg-clipped sources; `preds/<clip>__<model>/` the raw prediction CSVs.
  Models: `zoom-aug-exp/*-T2-{zoomaug,zoominout}` trunks and the `*-zio` N=50 draw-0 cells.
  - `facemap_testframes_comparison.mp4` — the same 2×2 over the 100 LABELED test frames
    (4 fps), RED = GT + error line. **This is the honest facemap comparison**: on the
    unlabeled `videos_test` sessions (D3/D4/D7-D9, 2021_11 — none overlap the labeled
    sessions) ALL models' eye-region keypoints (pupil, eye contour, mouth) collapse to
    conf ≈ 0 while nose/wrists stay confident, stretched or square-padded alike — those
    clips are eye-OOD, so `facemap_D3_comparison.mp4` shows hollow (low-conf) eye markers
    for every panel and should not be read as a model ranking on the eye.

## Architecture figure

- **`architecture/supermouse_architecture.{png,pdf}`** — two panels: (A) trunk training
  (five labs → 36-kp vocabulary → DINOv3 + shared head, T=2 + zoom-in/out aug),
  (B) anchored LoRA adaptation (frozen teacher distills unlabeled channels, LoRA student).
  Colors match the atlas: amber = trainable, grey = frozen, violet = teacher, teal = labels.

## "What the anchor buys" videos (31 Aug 2026)

- **`anchor-help/`** — per dataset, two panels (plain LoRA | anchored LoRA), both fine-tuned at
  N=50 with the target keypoint(s) REMOVED from the training labels (masked-label protocol);
  test frames carry the held-back GT (red + error line), the hidden keypoint drawn large
  (pupil yellow, wrists green/orange), all other keypoints as faint grey context dots.
  `ibl_pupil_hidden.mp4`, `cazettes_pupil_hidden.mp4` (old-trunk masked runs, draw 0);
  kondo/ibl wrists and zio-trunk versions re-render from the 31 Aug overnight masked grid
  via scratchpad `render_anchor_help.py`.

## Notebook-style figures & per-keypoint tables (31 Aug 2026)

- **`figs-notebook-style/`** — `plot_head_fixed_updated.ipynb`-style pixel-error-vs-shared-ensemble-std
  curves (draws 0–2 as the ensemble, 95/50/5% percentile markers), one figure per dataset per set:
  `<ds>__anchored_vs_{fullft,dino,lora}.png`. Series: zoom-in/out LOO trunk zero-shot (grey),
  dedicated model (green), anchored LoRA N=10/25/50 (purple shades), comparison arm N=10/25/50
  (orange/blue/teal shades). Generator: scratchpad `notebook_style_figs.py` (execs the notebook's
  cell-3 machinery).
- **`tables/per_keypoint_<ds>.csv`** — per-keypoint mean pixel error, all arms × N, 3-draw means,
  zoom-in/out trunks; rendered with best-arm highlighting at
  https://claude.ai/code/artifact/16229353-0a3d-46b5-85e1-10d4fd2038d6

### anchor-help/ibl_pupil_hidden_zio.mp4 (2026-08-31)
Four panels, same 100 ibl test frames, pupil_center_left HIDDEN during N=50 fine-tuning
(masked protocol, zio trunks, draw 2 — the draw where full FT collapsed): ZERO-SHOT teacher
(`zoom-aug-exp/face+cheese+caz+kondo-T2-zoominout`) / FULL FT (`fewshot-exp-lr5-zio-mask-…`,
93.9 px this draw, 13–94 across draws) / PLAIN LORA (`…lora-r16-lr5e-5-head5e-4-zio-mask-…`,
94.3 px every draw) / ANCHORED LORA (`…anchor-lora-conf1-zio-mask-…`, 2.0 px). Conventions:
hidden kp large yellow (hollow = conf<0.3), RED = GT + error line, grey = other keypoints.
Same renderer + conventions as the earlier ibl/cazettes hidden videos (render_anchor_help.py).

### ibl-facecrop/ibl_facecrop_zoom.mp4 (2026-08-31)
Advisor's face-crop test as footage: leave-ibl-out zio trunk (`zoom-aug-exp/face+cheese+caz+kondo-T2-zoominout`),
zero-shot on 30 ibl test frames shown as the crop the model saw at 1x/1.5x/2x/3x (face-centered
on GT nose+pupil midpoint). Orange = nose_tip, yellow = pupil_center_left (hollow = conf<0.3),
RED = GT + error line; per-panel caption shows live px errors + pupil conf. Finding: ~2 px at
full frame already; crops add ~nothing; 3x degrades (conf drops). Check frame: ibl_facecrop_zoom_check.png.

## SA pseudo-label replay vs anchored LoRA (3 Sept 2026)

- **`psl-vs-anchor-cazettes-pupil/psl_vs_anchor_cazettes_pupil.mp4`** — 3-panel comparison on
  all 217 cazettes-side test frames with pupil GT: frozen LOO teacher /
  `fewshot-exp-psl-replay-thr0.6-zio-mask-pupil_center_left` (SuperAnimal-style pseudo-label
  replay, full FT, threshold 0.6) / `fewshot-exp-anchor-lora-conf1-zio-mask-…` (draw 0).
  Conventions as the hidden-pupil videos: masked pupil large yellow (hollow = conf<0.3),
  RED = held-back GT + error line, grey = the model's other keypoints; per-panel banner shows
  this-frame error + conf; footer carries frame i/n | filename. Shows the teacher's systematic
  up-left pupil offset on this rig, corrected by SA's full-FT domain adaptation and faithfully
  preserved by anchoring. Renderer: scratchpad `render_psl_compare.py`.
- **`psl-vs-anchor-cazettes-pupil-v2/anchor_vs_SA_replay_cazettes_pupil_classes.mp4`** —
  2-panel (anchored LoRA | SA memory replay), all 217 cazettes-side pupil-GT test frames,
  keypoints colored by class: masked pupil large YELLOW, supported CYAN, new ORANGE (hollow =
  conf<0.3), RED = GT + error line for every labeled keypoint; per-panel masked-pupil error in
  the banner, frame i/n | filename in the footer. Renderer: scratchpad render_psl_compare_v2.py.
- **`psl-vs-anchor-cazettes-pupil-v3/GT_vs_anchor_vs_SA_overlay_cazettes_pupil.mp4`** —
  single-panel overlay, all 217 cazettes-side pupil-GT test frames: YELLOW = ground truth
  (large, keypoint name beside each), RED = anchored LoRA, GREEN = SA memory replay
  (psl-replay-thr0.6), error line from GT to each prediction in the model's color; banner
  carries both models' masked-pupil error, footer frame i/n | filename. Renderer:
  scratchpad render_psl_compare_v3.py.
- **`psl-vs-anchor-cazettes-pupil-v4/GT_vs_anchor_vs_SA_overlay_with_class_errors.mp4`** —
  v3 overlay plus per-frame class errors in the banner: for each of anchored LoRA (red line)
  and SA memory replay (green line), the frame's masked-pupil, pooled supported, and pooled
  new keypoint error. Same colors/labels as v3. Renderer: scratchpad render_psl_compare_v4.py.

## Paper supplementary videos (`paper-supp-videos/`, 2026-09-04)

Four videos for the NeurIPS workshop supplementary, one per pillar of the paper. All
legends are drawn INTO each video's top banner (no legend.png, per user request); every
panel carries the frame identifier; metadata stripped at encode (`-map_metadata -1`)
for double-blind submission; CRF 26. Rendered by `render_supp_videos.py` kept IN this
folder (scratchpad copies get wiped by /tmp cleanup).

- **`v1_masked_ibl_pupil.mp4`** (2.5 MB) — the Figure-5 story in motion. 4 panels
  (zero-shot base / full FT / LoRA / anchored), 100 ibl test frames,
  `pupil_center_left` hidden from the N=50 adaptation labels (zio trunks, subset 0:
  FT 31.4 px, LoRA 94.2, anchored 2.0 vs teacher 2.0). YELLOW = hidden-pupil estimate
  (hollow = conf < 0.3, always drawn), RED = held-back GT + error line, LIGHT BLUE =
  the model's other keypoints, drawn only at conf >= 0.7 (user: grey/low-conf was a mess).
- **`v2_facemap_pupil_transfer.mp4`** (3.9 MB) — the Figure-6 story in motion: the
  genuine (un-constructed) transfer keypoint. 78 facemap test frames with a visible
  labeled left-eye contour and base-conf > 0.5, cropped to the eye (GT-contour box,
  1.7x margin), same 4 arms at N=50 (unmasked runs). YELLOW = inherited pupil estimate
  (hollow = conf < 0.5), RED squares = the lab's own eye-contour labels; live conf
  readout per panel.
- **`v3_full_vocabulary_12panel.mp4`** (5.3 MB) — all-data zio trunk (seed0), 12
  canonical views, ALL 35 drawn keypoints colored by GROUP (eye cyan, pupil yellow,
  ear magenta, nose orange, whisker-pad green, mouth/lips blue, tongue pink, wrist
  white — groups few enough to explain in the banner, replacing the per-kp hue wheel
  of earlier 12-panels), small RED dots = that dataset's own GT; only conf >= 0.7
  predictions drawn, all solid (no hollow markers).
- **`v4_transfer_high_conf_12panel.mp4`** (4.9 MB) — same grid/model/colors but ONLY
  keypoints the dataset never labels (inherited; count in each panel header) and only
  conf >= 0.5 drawn. High-confidence transfer at a glance; cheese views are sparse by
  construction (8 inherited kps). Facemap's confident-but-misplaced inherited ear
  points are visible — consistent with fig_transfer's caveat about facemap's
  inherited group.

## Transferred keypoints over time: 200-frame traces (`transfer-traces-200f/`, 2026-09-14)

Question: are the transferred keypoints smooth in time and accurate? All-data zio model
(`zoom-aug-exp/face+ibl+cheese+caz+kondo-T2-zoominout/seed0`) on 200 consecutive frames from the
middle of each available video: facemap cam0, cheese-2d BC/L/R/TC/TL/TR (TR has no confident
transferred kps, no video), ibl left, ibl right (hflipped to match training). Left: video with
dots (conf >= 0.7) and an 8-frame trail. Right: one row per keypoint, x solid / y dashed, px from
the clip median, gaps where conf < 0.7, per-row median px/frame and count of >10 px jumps; legend
is in the banner and row labels. "Transferred" = not labeled by the dataset on EITHER side (eval
set plus its left/right mirror partners, since hflip supervises the partner); a keypoint is shown
only if its median conf over the clip >= 0.7. Where a view has several sessions (facemap 6, ibl 3)
the one with the most confident transferred kps is rendered (said in the banner). Script, clips
and per-clip prediction CSVs live in the folder (`make_transfer_traces.py predict|render`,
`_clips/`); `summary.csv` has the per-video numbers.

Findings: eye contours and nose points are smooth (0.1-0.4 px/frame, no jumps) and on target.
Jumps concentrate in few keypoints and are periodic with licking on ibl (upperlip_left ~6 Hz,
pad_center). Confidence >= 0.7 does NOT imply accuracy: on ibl right, ear_top_left and
upperlip_left sit on the paw (within 5-11 px of the predicted wrist), smooth and wrong; on facemap
cam0 (600 cam0 training frames exist, labeled as the right eye), ear and upperlip/pad_side points land on the
eye contour. cheese-2d mouth in view R oscillates ~5 Hz, consistent with chewing (chew_record).

### Learned vs transferred glitchiness (same folder, 2026-09-14)
`compare_transfer_vs_learned.py` on all 18 cached clips. Learned = the dataset's labeled kps;
transferred = neither labeled nor a mirror partner of one; both gated at median conf >= 0.7.
Metrics: jitter = median |p_t - midpoint of neighbours| (% frame width), jump rate = % of steps
> 3% frame width, plus % of steps where both frames are confident. Outputs:
`learned_vs_transferred_summary.png` (one dot per keypoint, per dataset), four
`traces_learned_vs_transferred_*.png` (shared ±6%-of-width y-scale, jumps clipped), and CSVs.
Result: transferred are NOT glitchier in jitter or jump rate (medians equal or lower in all three
datasets); glitches are keypoint-specific (ibl upperlip_left, facemap ear_tip_right). The real
difference is dropout: transferred kps fall below 0.7 far more often (confident steps, median:
facemap 77% vs 99.5%, cheese 82% vs 100%, ibl 100% vs 100% but mean 92% vs 99%). Facemap learned
nose points oscillate ~7 Hz (sniffing at 25 fps), which the jitter metric counts as jitter.

### Facemap transferred pupil video (same folder, 2026-09-14)
`facemap_pupil_transfer_traces.mp4` via `make_facemap_pupil_video.py predict|render`. No left-side
(cam1) facemap video exists, so camera-0 clips are MIRRORED to a left-side view, routing the eye to
the *_left channels as in training. All-data model, session D7 (highest median pupil conf of the
6 mirrored sessions: 0.44-0.92). Yellow pupil (hollow below 0.7), white = facemap's own eye-outline
predictions, other transferred kps at conf >= 0.7, 3x eye zoom under the video, pupil trace first.
Finding: the pupil is smooth (0.32 px/frame, no jumps) and confident (0.92, 100% of frames >= 0.7)
but sits at the FRONT CORNER of the eye in every frame (median 9.7 px from eye_front_left vs
20.8 px from the eye-outline centre, eye 56 px wide), i.e. the known all-data-model facemap pupil
failure, not the leave-facemap-out teacher used in fig_facemap.

### Three all-data seeds on the video clips (same folder, 2026-09-14)
Seeds 1 and 2 predicted on every cached clip (`_clips/seed1/`, `_clips/seed2/`; seed0 CSVs sit in
`_clips/`) by `seed_consistency.py predict`. Seeds differ in rng_seed_data_pt only. Confidence
threshold 0.6 throughout (user). Outputs:
- `ensemble_median_<ds>_<clip>.mp4` (9 clips, same as the transfer videos): transferred kps only,
  no traces; dot = per-frame median of the 3 seeds, error bars = +-1 SD across seeds (x, y, to
  scale); side panel lists live variance (var_x + var_y, px^2) and SD per keypoint plus clip
  medians. `ensemble_variance_per_keypoint.csv`. (`seed_consistency.py median`)
- `seed0_vs_seed1_ibl_{left,right}.mp4`: seed0 ring vs seed1 dot + traces; keypoint kept if either
  seed is confident, so one-seed-only channels are visible. (`pair`)
- `seed_disagreement_summary.png` + CSVs: per-keypoint seed disagreement, transferred vs learned. (`stats`)
Findings: seeds agree to ~0.5-1 px on most transferred kps; they share the systematic errors
(facemap pupil at the eye's front corner in all 3 seeds, ibl upperlip_left on the paw in all 3),
so seed variance does not flag those. Where seeds differ it is mostly in WHICH channel is
confident (ibl right ear_top_left: seeds 0 and 2 on the paw, seed1 conf 0).

## IBL face crop vs transferred face keypoints (`ibl-facecrop-transfer/`, 2026-09-14)
`facecrop_transfer.py predict|analyze`: all-data zio model seeds 0-2 on all 1446 IBL test frames,
full frame vs face crops at 1.5/2/2.5/3x (5:4 aspect kept; centred at 0.6*pupil + 0.4*nose_tip
from seed0's own full-frame predictions; crops written as PNGs, predictions shifted back).
Label-free checks for the transferred face kps against IBL's labels: pupil inside predicted eye
outline, confident face kps within 12 px of a labeled wrist, confidence, seed SD; plus supported
pupil/nose error. Outputs `facecrop_results.csv` (seed mean [range]), `facecrop_results_full.csv`,
`facecrop_per_keypoint.csv`, `facecrop_examples.png` (4 paw-failure frames, full/2x/3x).
Finding: no real gain in transfer. Eye outline, nose top/bottom, pad_center already fine at full
frame (pupil inside eye 98.7%) and unchanged by crops. The paw failures (ear_tip, ear_top,
upperlip, mouth confidently on a wrist) vanish at 2x (13% -> 2% of confident face predictions
on a paw) mostly because those channels lose confidence once the paw is cropped out, not because
they become correct; at 3x ear_tip regains confidence but lands on the eye. The crop centre
excludes the ear. Supported pupil/nose error 0.94 -> 0.80 px at 1.5-2x, 1.10 px at 3x.
- Videos (same folder): `facecrop_ibl_left.mp4`, `facecrop_ibl_right.mp4` via `facecrop_video.py
  predict|render`. 200 frames of the cached session-5c0c IBL clips (right camera mirrored), three
  panels: full frame / 2x / 3x face crop, all drawn on the full frame with the (clip-fixed) crop box
  dashed; face keypoints by group colour (eye, ear, nose, whisker pad, lips/mouth) at conf >= 0.6,
  legend in the banner, per-panel count of confident face keypoints. All-data model seed0.

## Whisker-pad heatmaps on facemap / cazettes (`whisker-heatmaps/`, 2026-09-17)
Raw heatmaps of the all-data zio model (seed0) for pad_center / pad_side_left / pad_top_left
(labeled only by cheese-2d, so pure transfer on facemap and cazettes) with nose_tip as reference.
Heatmaps come from `predict_step(return_heatmaps=True)` on the standard prediction data path;
reported points match the saved eval predictions exactly (0.000 px) on all test frames.
- `whisker_heatmaps.py predict|figures`: every test frame of facemap (100), cazettes (217),
  cheese-2d (291, reference). Per (frame, kp): LP conf, raw peak, number of bumps, gap between the
  reported soft-argmax point and the hottest cell (baseline = heatmap cell size, nose_tip shows 5-10 px),
  spread. `whisker_heatmap_summary.csv`, `whisker_heatmap_stats.csv`, `heatmaps.npz`,
  `heatmaps_{facemap,cazettes-side,cheese-2d}.png` (4 frames each: 2 random, 2 largest gap; cheese
  frames restricted to ones with the left pad labeled; fixed overlay scale 0.05).
- `whisker_heatmap_video.py predict|render`: `whisker_heatmaps_facemap.mp4` (100 CONSECUTIVE frames,
  cached cam0 D7 clip mirrored to the left-side view; matches the clip's saved predictions to 0.25 px
  median) and `whisker_heatmaps_cazettes-side.mp4` (first 100 test frames in file order; no cazettes
  video exists). 2x2 panels, fixed overlay scale 0.03, orange X = reported point, cyan + = hottest cell.
Findings: on facemap, pad_top_left is confident (92% of test frames) but its strongest bump sits on
the eye's front corner, with only weak bumps on the real whisker pad (cheese labels put it 0.33 of
the way from eye corner to nose tip; facemap predictions sit at 0.02); pad_center and pad_side_left
are mostly flat maps (56% and 92% of test frames), whose soft argmax lands mid-image at conf ~0.
On cazettes, pad_center is a clean single bump like nose_tip (100% confident) and pad_top_left is
usually on the snout with a secondary eye-corner bump; pad_side_left is flat in 19% of frames and
otherwise weak, sometimes hottest on a paw. Right-side pads are flat everywhere, as expected.

## ViT-B transfer-only 12-panel (`vitb-transfer-12panel/`, 2026-09-17)

`vitb_transfer_12panel.mp4`: same grid, frame sampling (<= 60 held-out test frames per view, 3 fps),
group colors and banner style as `paper-supp-videos/v4`, but the model is the ViT-B DINOv3
all-data run (`experiments/zero-shot-backbones/results/all-data/vitb_dinov3/seed0`, eval
predictions), the floor is conf >= 0.7 (solid markers, nothing below drawn), and "transfer" is
stricter than v4: not in the dataset's `trainable` set (labels plus hflip partners), so
facemap's right-eye channels and ibl's hflip-supervised pupil are excluded. `tongue_tip` is
drawn in every view; cazettes-side and kondo label it themselves (panel header says
"+ own tongue tip"). Updated same day: `pupil_center_right` added in every view as a yellow
RING (normally never drawn; flip-trained only). It counts as inherited for facemap, cheese-2d
and kondo; for ibl and cazettes-side the flip supervises it ("+ own R pupil"). At conf >= 0.7
it appears only in cheese-2d R/TC/TR and kondo, on the eye; never in ibl, facemap, cazettes. No GT. Renderer `render_vitb_transfer.py` and `check.png` in the folder.
Caveat visible in the video: facemap's inherited ear points (magenta) and a wrist (white) sit
on the eye contour at conf >= 0.7, the same confident-but-misplaced facemap transfer noted for v4.
- **Ears on IBL** (2026-09-17): `ears_ibl/` = same extraction for ear_*_left + eye_front_left on all 1446
  IBL test frames (stats in `ears_ibl/whisker_heatmap_stats.csv`); `whisker_heatmaps_ibl-ears.mp4` =
  100 consecutive left-camera frames (session 5c0c) via `whisker_heatmap_video.py predict|render ibl-ears`
  (script now takes a job name; facemap job unchanged). Test-frame result: ear_base_left flat 99%;
  ear_tip_left / ear_top_left confident in 37% / 63% of frames, and 87% / 80% of those sit on a labeled
  wrist; eye_front_left (also transfer) confident 99%, 99% at the labeled pupil. Ear labels: cheese-2d
  (left ear from view L/TL only, ~270 frames) + kondo right ear (bottom view); 2,705 of 57,986
  observations; no side-view rig labels ears. Eye contours: 12,944 observations incl. facemap's
  9,566 from the same close side view as IBL.
- `ear_heatmaps_ibl_{left,right}_12panelframes.mp4` (`ear_heatmaps_12panel_frames.py`): the same ear
  heatmap panels on exactly the 60 IBL test frames per view that the 12-panel supplementary videos
  use (sorted by path, 60 evenly sampled), 3 fps, so heatmaps line up with paper-supp-videos v3/v4.
- **Why ears fire on IBL paws** (`whisker-heatmaps/ear_why/`, 2026-09-17, `ear_why.py`): (A) occlusion
  on 150 IBL test frames where an ear channel fires on a labeled wrist: hiding the paws (box filled
  with frame median) drops on-paw from 92-94% to 0% and ear_tip/ear_top go flat in 54-60% of frames
  (confident 55-86% -> 24-25%); hiding a same-size box elsewhere changes nothing; eye_front unaffected.
  (B) cross-dataset: ear_tip_left fires on the paw in 41% of its confident frames on cazettes too;
  on kondo (bottom view, right ear labeled) right-ear channels are 99% confident and 0% on paws while
  left-ear channels are flat. `ear_patches.png`: cheese ear-tip labels = bright rounded lit blob
  against dark background; IBL paws = the same low-level pattern. `occlusion_stats.csv`.
- `ear_why/occlusion_test.mp4` (150 frames, 3 fps): original / paws hidden / control box, ear_tip_left
  and ear_top_left heatmaps with conf + raw peak per panel. `ear_heatmaps_ibl_{left,right}_12panelframes_gate0.04.mp4`:
  same 12-panel test frames with a raw-peak gate (grey ring = conf >= 0.6 but raw peak < 0.04).
  Gate numbers (IBL test, on top of conf >= 0.6): at peak >= 0.04 ear_tip keeps 10.5% of its confident
  detections (59% of survivors still on a paw) while keeping 90% of the real-ear detections on the
  5c0c clip and 99.4% of eye_front; ear_top keeps 16% with 89% of survivors on the paw. So the gate
  removes ~93% (tip) / ~82% (top) of paw hits but the strongest paw responses overlap real ears.

## Ear label conventions, cheese-2d vs kondo (`ear-label-conventions/`, 2026-09-17)
`ear_labels_video.py`: `cheese-2d_ear_labels.mp4` (6-panel, 100 GT frames per view, train+test) and
`kondo_ear_labels.mp4` (bottom view, 100 frames). 8 ear keypoints large + named with fixed colors,
other labels small grey. Geometry (median, in eye-to-nose units): cheese left ear tip 1.58 from the
eye at 172 deg from the eye->nose direction, bottom 1.44, tip-bottom span 0.42, base always labeled;
kondo right ear tip 1.36 at 162 deg, bottom 1.42, span 0.35, no base/top labels. Same anatomy
(tip = pinna apex, bottom = lower margin), but kondo sees the ear from below and labels only 2 of 4.
- `cheese_vs_kondo_ear_labels.mp4` (`ear_labels_combined_video.py`): both datasets in ONE 4x2 video,
  cheese views BC/TC/TL/TR/L/R + kondo bottom, same 100-frame index per panel (not the same moment),
  8th slot = text legend of the two labeling conventions. `ear_labels_video.py` now exposes its
  helpers (render_separate() behind __main__).

## Ear fine-tune probe (`whisker-heatmaps/ear_finetune/`, 2026-09-17)
Test of "a few side-view ear labels fix the paw confusion": anchored LoRA (paper recipe, 2000 steps)
from the ALL-DATA zio trunk seed0 on 20 frames of the ibl 5c0c left-camera clip (`data/`), with
PSEUDO-labels from the base model's own predictions where conf >= 0.9 and raw peak >= 0.04:
ear_tip_left 18 frames, ear_bottom_left 6, nothing else labeled (anchor distills the rest).
`train_ear_anchored.sh`, run `run_anchor_lora_ears20/` (ckpt kept), `score_ear_finetune.py`,
heatmaps `ears_ibl_finetuned/`, videos `../ear_heatmaps_ibl_{left,right}_12panelframes_finetuned.mp4`,
`moved_off_paw.png`. On the 1446 IBL test frames (other sessions):
ear_tip_left confident-on-paw 87% -> 23%, in the ear zone 12% -> 71%; ear_bottom_left on-paw
26% -> 4%, confident 20% -> 79%; ear_top_left (NO labels given) unchanged 80% -> 76%;
supported kps within 0.08 px of base, eye contour unchanged. Works in 16/19 view-sessions; three
left-camera sessions (26aa51ff, 30e5937e, ebe090af) stay on the paw. Caveat: pseudo-labels from one
session, not human labels.
- `ear_finetune/ear_finetune_before_after.mp4` (`before_after_video.py`): base model | fine-tuned side by
  side on the 120 12-panel IBL test frames (60 left + 60 right), rows ear_tip / ear_bottom (labeled)
  and ear_top (no labels, control); labeled paws as white squares. The one to show.
- `cheese_ears_on_kondo.mp4` (`cheese_ears_on_kondo.py`): do cheese-3d's ear labels transfer to kondo's
  labeled ears? 100 kondo TEST frames, 3 panels: cheese-only model (stock recipe), leave-kondo-out zio
  model (ears from cheese only), all-data model (reference). Green = kondo GT (ear_tip_right,
  ear_bottom_right), orange X = prediction, red line = error. No training needed (existing evals).
  Median error: cheese-only 18.8 / 8.4 px (tip / bottom), leave-kondo-out 17.0 / 9.8, all-data 7.7 / 5.2;
  cheese-trained tip has a heavy tail (p90 > 330 px, i.e. sometimes not on the ear at all).
- `cheese_ears_on_kondo_zoom.mp4` (`cheese_ears_on_kondo_zoom.py`): same comparison zoomed on Kondo's
  labeled ear (crop = 2x tip-bottom span) + a full-frame locator; dashed Kondo tip->bottom axis and the
  prediction's offset along it (units of the span) printed per frame. Cheese-trained tip sits a steady
  +0.17..0.19 span toward the bottom (IQR 0.12..0.26), bottom ~0: a convention difference, not noise.

## Cleanup note (2026-09-18)
Regenerable intermediates were deleted by `scripts/cleanup_results_v1.py`: `transfer-traces-200f/_clips/*.mp4`
(prediction CSVs kept), `ibl-facecrop-transfer/_work`, `whisker-heatmaps/{_video,_video_ibl-ears,heatmaps.npz,
ears_ibl/heatmaps.npz,ear_why/_occl,ear_why/_ibl__occl,ear_finetune/data,ear_finetune/ears_ibl_finetuned/heatmaps.npz}`.
Each folder's script rebuilds them with its `predict` step (GPU). Delivered videos, figures, stats CSVs and
scripts are untouched. Superseded run roots referenced above now live under `<results>/_archive/`.
