---
name: eval-suite
description: Standard evaluation of trained models for a corpus version, and the house convention for every qualitative deliverable (tables, figures, videos). Run after training finishes or when the user asks to evaluate, compare or visualize models. Never trains anything.
---

# Evaluation suite and qualitative deliverables

Every evaluation, standard or ad hoc, lives in its own folder
`<results_dir>/qualitative/<topic>-<MM-DD>/` with a `README.md` whose YAML front matter has
`title, date, data_version, question, models, outputs, finding` (see `scripts/qualitative_index.py`
docstring) and the script that produced the outputs. `python scripts/qualitative_index.py` rebuilds
`qualitative/INDEX.md` from those READMEs — run it after every delivery. Nothing is written into a
run's folder; runs are read-only inputs. Videos and figures follow the `pose-video` skill
(colors, legend in the banner, frame identifier per panel, re-encode with ffmpeg, no separate
legend PNG); its project adapter now names this convention. Folder layout (user, 2026-09-30):
`videos/` for mp4s, `csv/` for tables, NO PNG files at all; every file name states the model(s) it
comes from (zero-shot trunk / leave-X-out / subset trunk / all-data trunk / dedicated / anchored LoRA
from which trunk, N, draw) and the README maps those names to run directories. VIDEOS: one video per
request, and draw the ground-truth markers in it only when the user asks for them (default `--no_gt`).
This is about what is DRAWN in videos only: every evaluation (pixel error, detection, per-keypoint tables)
always scores against the ground-truth labels as before — see `pose-video`
"Output hygiene" and `results/head-fixed-v9/qualitative/ibl-adapt-09-29/README.md` as the example.

## The standard battery (run once per corpus version, after `train-plan` reports `done`)

Ask the user which parts to run; suggest all of them, in order, and never launch training.
Default models to compare (user, 2026-09-29): the all-data trunk as ViT-S 12k AND ViT-B 24k.

1. **Tables:** `python scripts/eval_suite.py` — pooled and per-keypoint pixel error of every
   finished trunk and dedicated model on every dataset's test set (`summary.csv`,
   `per_keypoint.csv`, README table). Fill in `finding:` in the README with one sentence.
2. **Previous-version comparison:** for datasets whose test set did not change (same
   `dataset_version` in both manifests), add the previous corpus version's runs with
   `--run prev-all=<path>` so the table shows old vs new side by side. Say clearly when a
   comparison is not valid (changed test labels, changed vocabulary). When only a dataset's LABELS
   changed but its frames did not, re-score the old model's saved `eval/<ds>/predictions.csv`
   against the new labels (same image index) — that makes the comparison valid without new
   inference. When a dataset gained keypoints, report them separately from the old ones.
3. **Transfer videos:** with `pose-video`, the 16-view panel (`render_panel.py --mode all
   --conf 0.7`, cheese-2d excluded) of each all-data model on every dataset's test frames; also
   `--mode transfer` (only keypoints the dataset does not label) when transfer is the question.
   These are the videos the user looks at first.
4. **Transfer proxy without labels:** for every dataset and every keypoint group it does not
   label, (a) the share of test frames with a confident (>= 0.7) prediction and (b) the share of
   those that land within ~10 px of a labelled keypoint of a DIFFERENT body part. (b) is a
   wrong-placement signal only for parts that are not anatomically adjacent (digits / wrist /
   tongue on a face = wrong; pupil inside the eye corners, pad next to the nose = normal). Do
   not propose raw-heatmap diagnostics — the user dropped them (2026-09-27).
5. **Leave-one-out zero-shot (the labelled transfer measure):** for each leave-X-out trunk, score
   the left-out dataset on BOTH its train and test frames (more frames): test predictions are in
   `<run>/eval/<X>/predictions.csv`, train predictions in `<run>/zeroshot/<X>_train_predictions.csv`
   (written by the training queue; otherwise predict `CollectedData_<X>_train.csv` with the run's
   `*-best.ckpt`). Per keypoint: median px error, detection rate (>= 0.7), confident-but-wrong rate
   (> 25 px or a dataset-appropriate scale), ViT-S vs ViT-B side by side, and the all-data trunk
   (which saw the labels) as the upper-bound reference.

## Fine-tuning a trunk on one dataset (anchored LoRA)

`scripts/anchor_ft.sh <trunk> <ds> <N|all> <steps> <out>` (recipe of record, S or B trunk).
**Judge it ONLY on the dataset it was fine-tuned on** (user, 2026-09-29): the fine-tuned model is
that lab's model, so other datasets' test sets are irrelevant — do not report or weigh retention on
other datasets. Report, on that dataset's held-out (OOD) test frames, per labelled keypoint: mean /
median px, detection (>= 0.7) and confident-but-wrong, before (the trunk, or zero-shot for a
leave-X-out trunk) vs after each N. On the same frames, the keypoints the dataset does NOT label
(the anchored ones) may be checked for staying put (confident share, shift vs the trunk). Video:
before | after columns on that dataset's test frames, same frame per column, conf >= 0.7, its
labels as x markers (template: `results/head-fixed-v9/qualitative/ibl-adapt-09-29/render_compare.py`).

## Ad-hoc investigations

Same folder shape and README; state the question first, the models and data version used, and
end with the finding. Prefer reading existing `eval/<dataset>/predictions.csv` over new
inference; when new inference is needed (videos, heatmaps, crops), say which checkpoint and
verify the reported keypoints reproduce the saved predictions before drawing conclusions.
