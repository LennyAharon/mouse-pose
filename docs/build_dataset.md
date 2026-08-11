# Building the head-fixed combined dataset

This covers how the training tags are built and — more importantly — *why* this particular set of
tags and not others. The question the head-fixed experiment exists to answer is: **does adding more
datasets to the training mix improve or degrade pose estimation, per dataset and overall?**

---

## Every tag uses every frame

There is one rule: a tag contains **all available train frames** from each of its datasets, built
with `build_dataset.py --n_frames -1`. No subsampling, no per-dataset frame cap, no `-600`-style
balanced variants.

An earlier generation of this dataset (`data/head-fixed_v1`, `_v2`) also built balanced tags that
capped every dataset at the same frame count, so that a merged model's advantage could be attributed
to cross-dataset *variety* rather than to simply having seen more frames. That comparison was
dropped: the practically relevant question is what the best model you can train from everything
currently labeled looks like, and the balanced variants doubled the tag count to answer a
methodological side question. If you find yourself needing them again, the mechanism still exists
(`--n_frames 600`) — it is just not part of the current design.

Because there is no pool to subsample from, **every training run uses `--train_frames 1`**, which is
Lightning Pose's convention for "use every frame in the CSV." The 200/400/600 learning-curve sweeps
described in older revisions of `train_sweep.md` no longer apply.

## The tag set: n=1, n−1, n=all

With `n` datasets, three groups of tags matter:

| Group | Tags | Answers |
|---|---|---|
| **n = 1** | each dataset alone | baseline — how good is this dataset by itself? |
| **n = all** | every dataset merged | does adding other datasets help? |
| **n − 1** | leave-one-out, one tag per omitted dataset | what does *removing* dataset X cost? |

Start with **n=1 and n=all** — together they answer the headline question directly. The **n−1**
ablations are the follow-up that attributes the result to individual datasets.

**Why not every combination.** The full power set is `2^n − 1` tags: 15 at n=4, 255 at n=8. The
n=1/n−1/n=all design is `2n + 1` — 9 at n=4, 17 at n=8 — and answers the questions actually being
asked. Adding a fifth dataset costs 2 more tags here versus 16 more under the full-combination
design, which is the property that keeps this tractable as datasets accumulate.

**n=1 tags require no build step.** `convert_dataset.py` already writes
`CollectedData_<dataset>_{train,test}.csv` containing every frame, which is byte-identical to what
`build_dataset.py --tag <dataset> --datasets <dataset> --n_frames -1` would produce. Pass the
convert output directly to `train_sweep.py --csv_files`.

## Test sets are never subsampled

`build_dataset.py` always writes the *full*, unsampled test split for every dataset in a tag — only
the train split is ever capped. Evaluation should use every available labeled test frame for
statistical power, and evaluation happens per-dataset (`eval/<dataset>/pixel_error.csv`) rather than
on the merged test set as a whole, so there is no balance argument for discarding test frames.

---

## Commands

`build_dataset.py` has no `--data_dir` override (unlike `convert_dataset.py`) — it always writes to
whatever `data_dir` in `paths.yaml` points at. The convention is to leave `paths.yaml` on the
unversioned working path (`data/head-fixed`), run the full pipeline there, then rename the directory
once everything succeeds (see the main README's dataset-versioning section).

```bash
# 1. Convert each raw dataset (slow — copies images). Run once each, or whenever a
#    dataset's raw labels change. This also produces the n=1 tags as a side effect.
python scripts/convert_dataset.py --dataset facemap
python scripts/convert_dataset.py --dataset ibl
python scripts/convert_dataset.py --dataset cheese-2d
python scripts/convert_dataset.py --dataset cazettes-side

# 2. n = all
python scripts/build_dataset.py --tag face+ibl+cheese+caz \
    --datasets facemap ibl cheese-2d cazettes-side --n_frames -1

# 3. n - 1, one tag per omitted dataset
python scripts/build_dataset.py --tag face+ibl+cheese --datasets facemap ibl cheese-2d      --n_frames -1
python scripts/build_dataset.py --tag face+ibl+caz    --datasets facemap ibl cazettes-side  --n_frames -1
python scripts/build_dataset.py --tag face+cheese+caz --datasets facemap cheese-2d cazettes-side --n_frames -1
python scripts/build_dataset.py --tag ibl+cheese+caz  --datasets ibl cheese-2d cazettes-side --n_frames -1
```

`--seed` is irrelevant at `--n_frames -1` — taking every frame involves no random choice. It only
matters if you reintroduce capped tags, where the same seed plus dataset name always selects the
same frames regardless of which other datasets share the tag.

### Result: 9 tags, 18 CSVs

| Tag | Datasets | Left out | Train frames | Test frames |
|---|---|---|---|---|
| `facemap` | single | — | 1800 | 100 |
| `ibl` | single | — | 5962 | 1446 |
| `cheese-2d` | single | — | 665 | 291 |
| `cazettes-side` | single | — | 830 | 217 |
| `face+ibl+cheese` | n−1 | cazettes-side | 8427 | 1837 |
| `face+ibl+caz` | n−1 | cheese-2d | 8592 | 1763 |
| `face+cheese+caz` | n−1 | ibl | 3295 | 608 |
| `ibl+cheese+caz` | n−1 | facemap | 7457 | 1954 |
| `face+ibl+cheese+caz` | n=all | — | 9257 | 2054 |

Train frames are post-`exclude.sessions` counts; see the main README for what exclusion means.

## Adding a dataset to the design

Adding dataset `X` to an existing `n`-dataset design means rebuilding the `n=all` tag (now `n+1`
datasets) and adding one new n−1 tag for each dataset — the old n−1 tags are no longer
leave-one-out, since they now omit two datasets rather than one. In practice: rerun every
`build_dataset.py` command above with `X` added, plus one new command omitting `X`. The n=1 tag for
`X` comes free with its `convert_dataset.py` run.

## Why `ibl` and not `ibl-paw`

`ibl-paw` (wrist-only) is deprecated — `ibl` is a strict superset (wrist + pupil_center + nose_tip +
tongue_end, and as of July 2026 fully human-reviewed rather than pseudo-labeled) and should always be
used instead. `_raw/ibl-paw` still exists on disk only because it's the input for regenerating
`ibl`'s face-keypoint pseudo-labels (see `scripts/preprocessing/ibl-face/README.md`) — don't
convert/build with it directly.
