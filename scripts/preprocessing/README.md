# Preprocessing a new dataset

This directory holds one subfolder per dataset that needed custom work *before*
`scripts/convert_dataset.py` could run on it — anything not already in the standard
DLC layout (`labeled-data/<session>/<frame>.png` + `CollectedData.csv` +
`CollectedData_test.csv`, all keypoints as plain `x`/`y` columns, NaN = unlabeled).
See `ibl-face/` (DLC source, but needs pseudo-label generation first) and
`hantman-sleap/` (raw SLEAP `.slp` source) for two different examples.

## Three separable stages

Turning a raw contributed dataset into something usable here is three distinct stages,
and **doing stage 1 does not commit you to stages 2 or 3**:

1. **Convert to LP format** (this directory's job) — raw source → standard DLC-layout
   `_raw/<name>/`. Output is a standalone, inspectable LP project. This is a complete,
   valid stopping point: sometimes a dataset just needs to be converted and looked at,
   with no decision yet about whether it belongs in the combined corpus.
2. **Add to the corpus** — wire `<name>` into the canonical keypoint vocabulary and
   registration lists (`configs/datasets/<name>.yaml`, `configs/keypoints.yaml` /
   `configs/model.yaml` if it has new keypoints, `ALL_DATASETS` / `EVAL_DATASETS`), then
   run `convert_dataset.py`. This is the step that actually merges the dataset's
   semantics into the shared vocabulary — it should not happen automatically just
   because stage 1 happened. **Ask before starting stage 2**, even if stage 1 just
   finished in the same conversation. `hantman` is a live example of stage 1 done,
   stage 2 deliberately not started.
3. **Rebuild the combined dataset** — re-run `scripts/build_dataset.py` for any merged
   tags that should include the new dataset.

This README covers stage 1. See the main `README.md`'s "Adding a new dataset" section
for stages 2 and 3.

## Questions to ask before converting (stage 1)

Most of stage 1 is mechanical once the shape of the source data is understood. The part
that repeatedly needed a human call — not something inferable from the data alone — is
the list below. **Ask about these up front, before writing any conversion code**, rather
than picking a default and mentioning it after the fact.

1. **New keypoints.** Does the source track anything not already in
   `configs/keypoints.yaml`? If so: what canonical name/section should it get, and —
   separately — is it even a keypoint that belongs in this vocabulary? (The hantman
   conversion tracked a `pellet`, the target object being reached for, not a mouse
   body part — excluded rather than added.) Note this is really a stage-2 question
   (the canonical vocabulary) — it only needs an answer in stage 1 if you're deciding
   what to *name* the source columns for later use; it doesn't have to be resolved to
   finish stage 1.

2. **Laterality.** If a tracked point is one-sided in the source (e.g. a single
   `wrist` rather than `wrist_l`/`wrist_r`) but the canonical name is `_left`/`_right`,
   which side does it map to? This is usually *not* recoverable from a single frame's
   pixels — mirroring/camera-orientation conventions and per-subject handedness are
   both invisible without lab context. Don't guess from an image; ask. (Also a
   stage-2 question in the strictest sense — it's about the mapping to canonical
   names, in `configs/datasets/<name>.yaml` — but worth settling early since it
   affects what stage 1's output is useful for.)

3. **Multi-view sources.** If the raw data has more than one camera view (or more
   generally, more than one natural sub-grouping), should each view become its own
   dataset entry (like `petersen-side` / `petersen-top`), or should they be merged
   into a single project (like `cheese-2d`, or `hantman`)? This changes stage 1's
   raw directory layout directly (one `_raw/<name>/` or several) — decide before
   writing the conversion script, not after.

4. **Train/test split.** If the source doesn't already provide a split (most
   contributed datasets don't), you need to invent one. Ask, don't default silently:
   - **Fraction** — what proportion of frames/sessions/subjects should be held out?
   - **Grouping unit** — every dataset in this repo splits so that no group leaks
     across train/test, but *what the group is* varies: session-level is the default
     (one video's frames stay together), but a multi-subject, multi-view dataset may
     need subject-level grouping instead, pooled across views, so the same animal
     never appears in both splits from a different angle. Always check for naming
     inconsistencies (casing, typos) in whatever field the grouping key is derived
     from, since those silently create leaks otherwise.

5. **Whether to go past stage 1 at all, right now.** Once the LP-format conversion
   works, don't assume the next move is corpus integration. Ask.

## Stage 1 mechanics

1. Write a `scripts/preprocessing/<name>/` script that reads the raw source and
   produces `_raw/<name>/labeled-data/...` + `CollectedData.csv` +
   `CollectedData_test.csv` in standard DLC format.
2. Spot-check: overlay a converted frame's keypoints on its image and confirm they
   land in the right place — this catches skeleton-ordering and coordinate-system bugs
   that nothing downstream will validate for you (`convert_dataset.py`'s own validation
   only checks names, not values, and only runs in stage 2 anyway).
3. Write `scripts/preprocessing/<name>/README.md` documenting the source format, why a
   custom script was needed, and any design decisions from the questions above.
4. Add the dataset to the main `README.md`'s `## Preprocessing` index, noting whether
   it's stage-1-only or already in the corpus.

Stages 2 and 3 (corpus integration, rebuilding merged tags) are in the main
`README.md`'s "Adding a new dataset" and "Renaming or deprecating a dataset" sections —
only do those once asked.
