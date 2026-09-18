# Fine-tuning the super-mouse model on a new dataset (few-shot)

How to take a trained super-mouse checkpoint and adapt it to a dataset with N labeled
frames, exactly the way the `fewshot-exp*` results were produced — so the numbers you get
are comparable to ours. Written for someone new to the project; every override is explained.

## 0. What you need

- The two repos on their experiment branches: `mouse-pose` on `zoom_aug`, `lightning-pose`
  on `super_mouse_paper` (the latter provides `training.epoch_repeat` and the per-dataset zoom
  augmentation; `lightning_pose` must be the editable install of that clone — see the
  README for setup and `paths.yaml`).
- A super-mouse checkpoint. Two kinds exist:
  - **All-data trunk** (use this for a genuinely new lab):
    `<results_dir>/zoom-aug-exp/face+ibl+cheese+caz+kondo-T2-zoomaug/seed0`
  - **Leave-one-out trunks** (use these to benchmark on one of our five datasets, so the
    target was never seen): `<results_dir>/zoom-aug-exp/<n-1 tag>-T2-zoomaug/seed0`, e.g.
    `face+cheese+caz+kondo-T2-zoomaug` when the target is ibl.
  The checkpoint file is `<run>/tb_logs/test/version_0/checkpoints/*.ckpt`. It holds the
  DINOv3 ViT-S backbone (21.6 M params) and the 36-channel heatmap head (one 3×3 transposed
  conv, 31 k params) — see `docs/fewshot_lora_plan.md` §0 for what is inside.
- The target dataset in the project's canonical format: `CollectedData_<dataset>_train.csv`
  under `<data_dir>` with its frames under `labeled-data/<dataset>/…`, keypoints mapped onto
  the 36-keypoint vocabulary. For a new dataset this is the convert + build step in the
  README; for our five it already exists.

The README and docs prefix commands with `conda run -n pose`; on the studio machine run
`python` / `litpose` directly.

## 1. The command

One fine-tune = one `litpose train` call. This is what `scripts/fewshot_cell.sh` runs
(arm `trunk5`, our primary line):

```bash
cd mouse-pose
litpose train configs/model.yaml --output_dir <out_dir> --overrides \
    data.data_dir=<data_dir> \
    data.csv_file=CollectedData_<dataset>_train.csv \
    model.backbone=vits_dinov3 "model.losses_to_use=[]" \
    "+model.checkpoint='<path to the trunk .ckpt>'" \
    training.train_frames=<N> training.rng_seed_data_pt=<draw> \
    training.optimizer_params.learning_rate=5e-05 \
    training.min_steps=2000 training.max_steps=2000 training.unfreezing_step=1 \
    training.val_check_interval=1000 \
    "training.lr_scheduler_params.multisteplr.milestone_steps=[1000]" \
    +training.epoch_repeat=100 +training.num_workers=2
```

What each line does and why it is set this way:

| override | meaning | why |
|---|---|---|
| `configs/model.yaml` | the stock config: heatmap model, `dlc` augmentation, no per-dataset zoom, no sampling temperature | fine-tuning is on ONE dataset, so the multi-dataset machinery (registry, temperature sampler, per-dataset zoom) is not used. Do not pass `data.dataset_names` or `training.sampling_temperature`. |
| `data.csv_file` | the target dataset's training CSV | the split into train/val/test is made from it with `train_prob 0.95 / val_prob 0.05` (config defaults); the N frames come from the train part |
| `model.backbone=vits_dinov3`, `losses_to_use=[]` | same backbone as the trunk, supervised loss only | must match the checkpoint's architecture |
| `+model.checkpoint='…'` | load backbone AND head weights from the trunk before training | this is the whole point; the log must contain `loading weights from` — our launcher aborts if it does not. The `+` is required because the key is not in the base config. |
| `training.train_frames=N` | keep only N of the training frames | N ∈ {10, 25, 50} in our grid. `train_frames=1` means *all* frames (Lightning Pose convention) — never use 1 to mean one frame. |
| `training.rng_seed_data_pt=<draw>` | seed for the split and the N-frame draw | the same seed picks the same frames for any model, so arms are compared on identical frames. We use draws 0, 1, 2 — this is the *frame* seed; the model seed stays 0. |
| `learning_rate=5e-05` | one Adam lr for everything | the project's lr for `vits_dinov3` (config default is 1e-3 — always override). 5e-5 beats 1e-5 on every dataset with keypoints the trunk never trained; 1e-5 keeps already-known keypoints slightly better but never learns new ones. See §4. |
| `min_steps=max_steps=2000` | fixed budget | short, honest protocol: no early stopping on the test set |
| `unfreezing_step=1` | backbone trainable from the first step | a fine-tune starts from a converged model; the default 1000-step freeze is for training from scratch |
| `val_check_interval=1000` | validate at steps 1000 and 2000 only | at most two model-selection points; the best of the two is kept (`epoch=…-step=1000-best.ckpt` or `…-step=2000-best.ckpt`) |
| `milestone_steps=[1000]` | lr halves at step 1000 | mirrors the full recipe's step-based schedule, scaled |
| `+training.epoch_repeat=100` | 100 shuffled passes per loader epoch | with 10–50 frames one pass is a single batch and Lightning's per-epoch overhead dominated (45–65 min per cell); this packs passes with *identical* batch boundaries, ~10 min per cell. Purely a speed setting. |
| `+training.num_workers=2` | dataloader workers | 8-core machine shared by several jobs; no effect on results |

Not set, and deliberately so: `imgaug` stays `dlc` (the trunk was trained with per-dataset
zoom, but a single-dataset fine-tune uses the stock augmentation), `train_batch_size`
stays 32 (a batch is the whole N-frame set when N ≤ 32).

## 2. Evaluate

```bash
python -m mouse_pose.train --output_dir <out_dir> --csv_file CollectedData_<dataset>_train.csv
```

This predicts on every dataset's test CSV and writes `<out_dir>/eval/<dataset>/pixel_error.csv`
and `predictions.csv`. It deletes the checkpoint afterwards unless you add
`--keep_checkpoints` (keep it if you want to fine-tune again or run video inference).

## 3. Score

```bash
python scripts/fewshot_score.py     # prints per-dataset tables + paired dino/trunk/trunk5 table
python scripts/fewshot_plot.py      # frames→error figure under <results_dir>/qualitative/fewshot-curves/
```

Metric (project convention): pooled mean pixel error over all labeled (test frame, keypoint)
cells of `eval/<target>/pixel_error.csv`, `pupil_center_right` excluded. Report **both**:

- **supported keypoints** — those at least one of the trunk's training datasets labels (the
  trunk has trained those head filters); computed from `dataset_inventory.json` as
  `eval(target) ∩ ⋃ trainable(training datasets)`;
- **all keypoints the target labels** — includes the ones the trunk never trained.

Anchors to put next to any number: **zero-shot** (the trunk's own `eval/<target>/` — no
labels) and **dedicated** (`<target>_train/supervised/tf1/vits_dinov3/seed0`, all labels).
For a completely new dataset there is no dedicated anchor until you train one.

## 4. The two learning rates we tested

| | lr 1e-5 (`fewshot-exp/`) | lr 5e-5 (`fewshot-exp-lr5/`) |
|---|---|---|
| keypoints the trunk already knew | kept slightly better (ibl 6.4 vs 6.6; cheese 19.1 vs 25.3 at N=50) | drifts on cheese-2d (28 → 43 px at N=10) |
| keypoints the trunk never trained | never learned (cheese all-keypoint error stuck at 55 px) | learned (cheese 39.5 → 24.7 px from N=10 to 50) |
| verdict | only for a dataset with no new keypoints (kondo) | **use this** |

Reason: Adam's update per step is ≈ lr in size, so 2000 steps give each weight a travel
budget of ≈ 0.015 at 1e-5 vs 0.075 at 5e-5. Filters that were never trained need the large
budget; already-trained weights would rather have the small one. `docs/fewshot_lora_plan.md`
describes the next protocol that gives each kind its own budget.

## 5. Baseline: the same fine-tune from the plain DINOv3 backbone

To measure what the super-mouse trunk is worth, run the identical command **without**
`+model.checkpoint`: the backbone is then Meta's pretrained DINOv3 ViT-S/16 (downloaded from
Hugging Face — the same init the trunk itself started from) and the head is random. That is
arm `dino` (`fewshot-exp-dino/`). Same frames (same `rng_seed_data_pt`), same 2000 steps,
same lr, so the only difference is the initialization.

## 6. Running many cells: the pool

`scripts/fewshot_cell.sh <arm> <dataset> <N> <draw>` wraps §1–2 for one cell with the
guards we rely on (lr present in the saved config, trunk weights loaded, training COMPLETED,
5 eval folders), writes a `.done` marker, and is idempotent (re-running a finished cell is a
no-op; a cell whose training finished but eval died runs eval only). Arms:

| arm | init | lr | root |
|---|---|---|---|
| `trunk5` | leave-<dataset>-out zoomaug trunk | 5e-5 | `fewshot-exp-lr5/` (primary) |
| `trunk` | same trunk | 1e-5 | `fewshot-exp/` (lr-sensitivity line) |
| `dino` | DINOv3 backbone, random head | 5e-5 | `fewshot-exp-dino/` (baseline) |

The trunk for each dataset is hard-coded in the script's `LOO` map; for a new dataset add
an entry pointing at the all-data trunk.

`scripts/fewshot_pool.sh <worker_id>` runs cells listed in `scripts/fewshot_queue/queue.txt`
(one `arm dataset N draw` per line), claiming each with an atomic `mkdir`, never more than
three cells on the GPU at once. Launch workers **detached** so they survive the session:

```bash
setsid nohup bash scripts/fewshot_pool.sh 1 > pool_w1.log 2>&1 < /dev/null & disown
```

(A `/compact` in Claude Code killed session-tracked background jobs once; detached workers
plus idempotent cells make a restart free.)

## 7. Output layout

```
<results_dir>/fewshot-exp-lr5/<dataset>/tf<N>-draw<draw>/
    config.yaml                       # the resolved config — check learning_rate, train_frames, epoch_repeat here
    train_status.json                 # {"status": "COMPLETED", ...}
    tb_logs/test/version_0/           # TensorBoard scalars (val at steps 999/1999) [+ checkpoint if kept]
    predictions.csv                   # Lightning Pose's own prediction on the train CSV (all frames)
    eval/<dataset>/pixel_error.csv    # what the scorer reads (test frames of each dataset)
    .done                             # written by fewshot_cell.sh after a complete eval
<results_dir>/fewshot-exp-lr5/<dataset>-tf<N>-draw<draw>.log       # training log
<results_dir>/fewshot-exp-lr5/<dataset>-tf<N>-draw<draw>-eval.log  # eval log
```

## 8. Checklist before trusting a number

1. `grep learning_rate <out_dir>/config.yaml` → `5.0e-05` (the config default 1e-3 has cost us two reruns).
2. `grep "loading weights from" <log>` → present for a trunk arm, absent for `dino`.
3. `grep train_frames <out_dir>/config.yaml` → your N (not 1).
4. `train_status.json` says COMPLETED; `eval/` has all five dataset folders.
5. Compare arms only on the same `rng_seed_data_pt`, and always report both keypoint sets.
