# Transferring pseudo-labels from one dataset's model onto another dataset

Once a standalone Lightning Pose model exists for one `_raw/<dataset>/` (see
`skills/train-lightning-pose-model/`), it's often useful to run that model's inference
against a *different* dataset's already-labeled images and fill in gaps: cells that
are currently empty, where the model is confident. This is separate from — and
upstream of — the combined-corpus pipeline (`preprocess-new-dataset`, `convert_dataset.py`):
it edits a dataset's own `_raw/<dataset>/CollectedData.csv`/`CollectedData_test.csv`
directly, in place.

`scripts/transfer_pseudo_labels.py` does the actual work. This README covers the
decisions to confirm with the user *before* running it — none of them are recoverable
from the data alone — plus the gotchas the script's design works around.

## Decision checklist (ask before running)

1. **Which keypoints actually transfer?** The source model's keypoint names and the
   target dataset's CSV columns are not necessarily the same set, even if both datasets
   look related. Check both `project.yaml`s (or the model's `config.yaml` `data.keypoint_names`
   vs. the target CSV's `bodyparts` header row) — don't assume a shared naming convention
   means a shared keypoint set. A keypoint the source model predicts but the target
   dataset doesn't have a column for is a **new-keypoint decision** (see
   `preprocess-new-dataset`'s equivalent question) — ask whether to add the column or
   just skip that keypoint; don't add it silently.
2. **Confidence threshold.** 0.7 has been the starting point so far, but ask — don't
   default silently.
3. **Scope: existing labeled rows only, or also new frames?** Filling gaps in images
   already present in the target CSVs is very different from also running inference on
   additional (currently unlabeled) frames and adding new rows. The script only supports
   the former (existing rows) — if the user wants new frames added too, that's a
   different, bigger task (needs a frame-selection strategy, not just a merge).
4. **Per-view/structural masking beyond the confidence threshold?** Some target cells
   are empty for structural reasons (the keypoint genuinely isn't visible in that
   camera view), not because of an annotation gap. If the source model was trained
   treating those as "occluded" (uniform heatmap target, see
   `[[project_combined_dataset]]`'s visible-column convention), the confidence threshold
   alone is usually the right filter — a model trained that way should predict low
   confidence there. Ask if the user wants additional hard masking on top of that.

## Running it

```
conda run -n pose python scripts/transfer_pseudo_labels.py \
    --model_dir <path to a trained model dir, e.g. results/cheese-3d/2026-09-26_15-45-24> \
    --target_dataset <name matching a directory under raw_dir> \
    --keypoints "kp1" "kp2" ... \
    --confidence_threshold 0.7 \
    --dry_run
```

Run with `--dry_run` first and show the user the per-keypoint / per-group inventory it
prints. On confirmation, re-run without `--dry_run` to actually write the target CSVs.

`--csvs` defaults to both `CollectedData.csv` and `CollectedData_test.csv`; pass it
explicitly to restrict to one.

## What the script guarantees

- Only fills `(x, y)` for the requested `--keypoints`, only where the target cell was
  already empty, only where prediction likelihood >= `--confidence_threshold`.
- Never adds rows, never adds keypoint columns, never touches an already-labeled cell —
  it asserts this internally (`verify_untouched`) and raises rather than writing if it
  finds a violation.
- Never calls `bump_version.py`. See `skills/bump-dataset-version/` and
  `[[feedback_version_bump_confirmation]]` — versioning the result is a separate,
  user-requested step, after they've verified the fill by hand (e.g. in the LP labeling
  app). Report the fill counts and stop there.

## Gotchas this script works around

- **`litpose predict` (the CLI) can't do this** — its CSV-input path always uses the
  model's own training `data_dir`, with no override. The script uses the
  `lightning_pose.api.Model.predict_on_label_csv(..., data_dir=...)` Python API directly
  instead, which does support an explicit `data_dir` for the target dataset's images.
- **Output-path collision:** `predict_on_label_csv` keys its output directory purely off
  the input CSV's basename (`<model_dir>/image_preds/<csv_basename>/predictions.csv`).
  Feeding it a file literally named `CollectedData.csv` would silently overwrite the
  model's own self-eval predictions from training (which live at that exact same path).
  The script works around this by copying the target CSV to a temp file named
  `<target_dataset>__<original_name>` before calling predict — always distinct from
  anything the model has predicted on before.
- **Verify after every revert/redo, not just the first attempt.** If a run needs to be
  re-scoped (different keypoints, different threshold) after already writing once, always
  diff the freshly-produced result cell-by-cell against a known-clean original copy of the
  target CSVs before trusting the counts — see `[[feedback_pseudo_label_verify_before_iterating]]`
  for why (a real instance of this going wrong, silently, mid-session).
