---
name: pose-video
description: Create high-quality overlay videos or figures of pose-estimation predictions and/or ground-truth labels, in any requested format — keypoint subsets, per-keypoint colors, marker sizes, names, legends, frame ranges, fps, multiview layouts. Use whenever asked to visualize pose predictions, labeled keypoint data, or prediction-vs-label comparisons, on any pose project (single- or multi-view).
---

# Pose overlay videos and figures

You are rendering keypoint overlays. Requests vary every time (colors, subsets, sizes,
layouts); the craft below stays constant. The skill is project-agnostic — a
project-specific adapter section at the end covers the repo this skill ships with, and
the same pattern (paths module + conventions doc) applies to any other project.

## House style for mouse-pose — MANDATORY, overrides the generic craft below

Every video or figure made in this workspace, by any agent (Claude, Codex, Cursor), looks the
same. The generic "Rendering craft" section further down is for other projects; where it differs,
**this section wins.** Do not invent a new layout: copy the closest reference script below into
the new delivery folder and change only what the request needs.

1. **Colours = the standing GROUP palette, never one hue per keypoint.** One saturated BGR hue
   per keypoint group, none white or grey: eye (255,255,0), pupil (0,255,255), ear (255,0,255),
   nose (0,140,255), whisker pad (150,190,0), mouth/lips (255,130,60), tongue (170,110,255),
   wrist (255,0,127), digit tips R (0,255,120), digit tips L (0,0,255). The legend is ONE row of
   group swatches in the banner. No per-keypoint numbered legends, no golden-ratio wheel.
2. **Banner** (top of every frame, ~62 px): the group legend row, then one auto-fitted line
   saying which model(s), what is drawn, the confidence gate, any bar scale, and whether ground
   truth is drawn.
3. **Every panel** has a black header (view name, keypoint count, one summary number) and a
   black footer `step i/n | frame | source file` (see [[qualitative-video-conventions]]).
4. **Uncertainty = a +-1 SD cross** (x and y bars separately) centred on the dot, plus a side
   **strip** beside each panel listing, per drawn keypoint, SD x / y in true pixels (and
   confidence). Bars are true length; if they are sub-pixel (typical of EKS posteriors), scale
   them by a stated factor (x10) and say so in the banner; the strip keeps true pixels.
5. **Confidence gate 0.7** unless the user says otherwise; say it in the banner.
6. **Output hygiene** (below) applies unchanged: `videos/` + `csv/`, NO PNGs, file name
   `<what>__<model tokens>[_nogt].mp4`, README maps tokens to run dirs, one video per request,
   no ground truth unless asked. Cheese-2d is left out unless the user names it.

Reference scripts (copy, don't rewrite):

| video kind | reference |
|---|---|
| multi-view trunk panel (transfer / all keypoints) | `results/head-fixed-v9/qualitative/trunk-panels-09-29/render_panel.py` |
| model comparison, rows of models | `results/head-fixed-v9/qualitative/ibl-adapt-09-29/render_compare.py` |
| ensemble median + variance crosses | `results/head-fixed-v9/qualitative/ibl-adapt-09-29/ensemble_variance_rows.py` |
| EKS output with posterior-variance crosses (continuous clip) | `results/head-fixed-v9/qualitative/ibl-singleview-eks-09-30/render_sv_eks.py` |

## Workflow

1. **Find the source of truth for what to draw.** Predictions live in DLC-style CSVs
   (see formats below), produced either by a prior evaluation or freshly by running the
   model. Ground truth lives in label CSVs. Never invent coordinates.
2. **Start from an existing renderer if the project has one**; otherwise write a small
   cv2 script in the scratchpad. For one-off format requests, copy-and-edit in the
   scratchpad — only fold a change into a repo script when it is generally reusable.
3. **Render → re-encode → spot-check → deliver.** Extract one frame of the finished mp4 with
   cv2 into the SCRATCHPAD (never into the delivery folder), look at it, and verify
   markers/colors/legend match what was requested before handing over.

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

- **Left vs right must be tellable apart** whenever a lateral pair is drawn as a group (digit
  tips, wrists, ears): give the two sides two hues, never one hue for the pair.
- **Color design (other projects only; mouse-pose uses the group palette above):** distinct color
  per keypoint via a golden-ratio hue wheel
  (`h = (offset + i * 0.618034) % 1`, high saturation/value); if red is reserved for
  ground truth, exclude the red band (`h < 0.06 or h > 0.94 → shift`). Keep the SAME
  keypoint→color map across every video/view/figure in a delivery, and put the legend in
  the banner of every frame (no separate legend image).
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

- Write to a NEW folder `<results_dir>/qualitative/<topic>-<MM-DD>/` with a README.md whose
  front matter (title, date, data_version, question, models, outputs, finding) feeds
  `scripts/qualitative_index.py`; never overwrite an existing video set (`eval-suite` skill).
- **Folder layout (user, 2026-09-30):** videos go in `<folder>/videos/`, CSV tables in
  `<folder>/csv/`; scripts, README and .md tables stay at the top. **Never write PNGs** (no
  `_check.png`, no stills, no legend images) unless the user asks for an image.
- **Every file name says which model(s) produced it**, `<what is shown>__<model tokens>.mp4`
  (e.g. `zeroshot__LOO-ibl-vits12k.mp4`, `rows-allkps__zeroshot-LOO-ibl-vits12k_vs_alldata-vits12k.mp4`,
  `ensemble-mixture__aLoRA-r16-N2000-from-4ds-vits12k-3draws.mp4`): zero-shot vs trained-on-it,
  which trunk (leave-X-out / named subset / all-data), backbone and steps, fine-tuning method
  (anchored LoRA rank/lr, N frames, draw) and ensemble type. The README has a table mapping
  every token to its run directory, and the banner of every frame names the model per panel.
- **One video per request; ground-truth markers only when the user asks for them** (user,
  2026-09-30): default render with `--no_gt` (file suffix `_nogt`), and a labelled version only on
  request. This governs what is drawn; pixel-error evaluation always uses the labels.
- cv2's `mp4v` doesn't play in VS Code/browsers — always re-encode:
  `ffmpeg -y -i x.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4`. If `ffmpeg` isn't on
  PATH: `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`.
- Get machine paths from the project's paths config, never hardcoded.

## Project adapter: mouse-pose (the repo this skill ships with)

- Paths: `from mighty_mouse.paths import load_paths` (`data_dir`, `results_dir`);
  registry: `from mighty_mouse.registry import load_registry`.
- Base renderer: `scripts/render_supermodel_videos.py`
  (`--style classes|perkp`, `--conf`, `--max_frames`, `--fps`) over a run's
  `eval/<dataset>/predictions.csv`. Blind-mode scoring: `scripts/blind_eval.py`.
- Which keypoints a dataset labels: `trainable` in `<data_dir>/dataset_inventory.json`.
- Project exclusion: `pupil_center_right` is hflip-trained only — excluded from scoring
  and usually from display (include it only if explicitly asked for all 36).
- **Which run to render from — `docs/results_catalog.md`.** Read it BEFORE choosing a
  model: it decodes every folder name under `<results_dir>` (arm, trunk suffix, masked
  protocol), marks which roots are CURRENT vs superseded, and lists the five model
  roles (zero-shot trunk / anchored LoRA / plain LoRA / full FT / dedicated) with the
  exact paths and whether checkpoints survive for fresh predictions.
- **Index of deliveries:** `<results_dir>/qualitative/INDEX.md` (generated; per-folder READMEs are the
  source). v1's history is `docs/qualitative_catalog.md`.
  Read it before rendering: it lists what exists under `<results_dir>/qualitative/`,
  which model produced it, and the exact color/confidence conventions each delivery
  used (reuse them for consistency). Append new deliveries to it.
- Operational reference: `docs/operations.md`.

## Standing deliverable: the transfer panel (run this for every new trunk)

The video the user asks for whenever a new all-data trunk finishes: one panel per camera view
in the corpus, showing only the keypoints that view's dataset never supervises, so the whole
transfer class is inspectable at a glance. Do not rebuild it from scratch — reuse

    poseinterface/results/head-fixed-v9/qualitative/trunk-panels-09-29/render_panel.py
    (latest copy; imports mighty_mouse, the renamed package; default --mode transfer)
    python render_panel.py --model <run dir with eval/> --out <name>.mp4 \
        [--mode transfer|all|own] [--datasets a,b] [--exclude a,b] [--data_dir <corpus>] \
        [--conf 0.7] [--frames 60] [--panel 300] [--cols 5] [--fps 3]

copied into the new delivery folder (`<topic>-<MM-DD>/`) so each delivery keeps its own script.
Conventions it encodes, which the user has asked for repeatedly:

- **Cheese-2d is left out of every video unless the user names it** (standing preference,
  2026-09-25). `--exclude` defaults to `cheese-2d`; pass `--exclude ""` only when asked.

- **Transfer = not in `trainable`** for that dataset (its own labels *and* the lateral partners
  horizontal flips supervise). Looser definitions leak flip-supervised channels into the panel.
- **Confidence floor 0.7**, markers solid, nothing below the floor drawn.
- **`tongue_tip` and `pupil_center_right` are drawn in every view**, even where supervised; the
  panel header then says "+ own tongue tip / R pupil". `pupil_center_right` is a **ring**, since
  no dataset labels it and flips alone train it. This is the one sanctioned exception to the
  project rule that it is never drawn.
- **One saturated hue per keypoint group, never white or grey** (white markers on greyscale mouse
  video are unreadable): eye cyan, pupil yellow, ear magenta, nose orange, whisker pad teal,
  mouth/lips blue, tongue pink, wrist violet. **Digit tips are split by side: right green, left
  red** (user request 2026-09-22: a transfer video must show at a glance whether a left or a right
  digit channel fired; red is free here because this panel draws no ground truth). Apply the same
  rule to any other lateral pair the user asks to tell apart. The banner auto-shrinks to the
  canvas width, so a long run path never runs off the edge.
- **Every panel carries its view name, its inherited-keypoint count, and a step/frame footer**;
  the banner names the run, the floor and the colour key. No separate legend.png.
- `VIEW_ORDER` in the script lists the (dataset, view) panels and `view_of()` maps a session
  directory to its view. **A new dataset or camera means adding both**; views present in the data
  but missing from `VIEW_ORDER` are appended at the end and reported, never dropped silently.

Variants the user also asks for, same script:

- `--mode all`: every channel, own and inherited ("show all the high-confidence keypoints").
  Headers read "N labelled + M other".
- `--mode own`: only the dataset's own labelled keypoints (`direct` in the inventory, not the
  flip-supervised partners). Paired with `--mode all` on the same `--datasets` and frames, the
  difference between the two videos is exactly what the trunk inherits for that dataset.
- `--datasets` / `--exclude` restrict the panels; for one or two views use `--cols 2 --panel 640
  --frames 120`.
- **`--data_dir` pins the corpus.** `paths.yaml` always points at the newest version, so when
  rendering an older version's model, pass that version's data dir or the inventory and images
  will not match the model.
