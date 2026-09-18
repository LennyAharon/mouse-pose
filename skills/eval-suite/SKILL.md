---
name: eval-suite
description: Standard evaluation of trained models for a corpus version, and the house convention for every qualitative deliverable (tables, figures, videos). Run after training finishes or when the user asks to evaluate, compare or visualize models. Never trains anything.
---

# Evaluation suite and qualitative deliverables

Every evaluation, standard or ad hoc, lives in its own folder
`<results_dir>/qualitative/<YYYY-MM-DD>-<topic>/` with a `README.md` whose YAML front matter has
`title, date, data_version, question, models, outputs, finding` (see `scripts/qualitative_index.py`
docstring) and the script that produced the outputs. `python scripts/qualitative_index.py` rebuilds
`qualitative/INDEX.md` from those READMEs — run it after every delivery. Nothing is written into a
run's folder; runs are read-only inputs. Videos and figures follow the `pose-video` skill
(colors, legend in the banner, frame identifier per panel, re-encode with ffmpeg, no separate
legend PNG); its project adapter now names this convention.

## The standard battery (run once per corpus version, after `train-plan` reports `done`)

Ask the user which parts to run; suggest all four, in order, and never launch training.

1. **Tables:** `python scripts/eval_suite.py` — pooled and per-keypoint pixel error of every
   finished trunk and dedicated model on every dataset's test set (`summary.csv`,
   `per_keypoint.csv`, README table). Fill in `finding:` in the README with one sentence.
2. **Previous-version comparison:** for datasets whose test set did not change (same
   `dataset_version` in both manifests), add the previous corpus version's runs with
   `--run prev-all=<path>` so the table shows old vs new side by side. Say clearly when a
   comparison is not valid (changed test labels, changed vocabulary).
3. **Transfer videos:** with `pose-video`, the 12-view panel of the all-data model on each
   dataset's test frames, one version showing every keypoint and one showing only keypoints the
   dataset does not label (confidence >= 0.6). These are the videos the user looks at first.
4. **Suspect-channel check:** raw heatmaps for any keypoint the user or the tables flag
   (the `whisker-heatmaps` scripts from v1 are the template: heatmap stats, bump count,
   flat-map rate, raw peak), because confidence alone does not reveal a channel firing on the
   wrong feature.

## Ad-hoc investigations

Same folder shape and README; state the question first, the models and data version used, and
end with the finding. Prefer reading existing `eval/<dataset>/predictions.csv` over new
inference; when new inference is needed (videos, heatmaps, crops), say which checkpoint and
verify the reported keypoints reproduce the saved predictions before drawing conclusions.
