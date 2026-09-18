# Few-shot adaptation with LoRA — design note (2026-08-26)

What fine-tuning does today, weight by weight, and what changes if the same n−1
super-mouse checkpoint is adapted with LoRA instead. Companion to plan §15 and to the
results in `fewshot-exp*/` (scored by the scratchpad `fewshot_score.py`, figure at
`qualitative/fewshot-curves/fewshot_curves.png`). Sections 0–8 are the design as planned on 2026-08-26; the probe outcome is at the end.

## 0. Vocabulary, pinned to the actual tensors

The model we fine-tune is a leave-one-dataset-out super-mouse checkpoint, e.g.
`zoom-aug-exp/face+cheese+caz+kondo-T2-zoomaug/seed0` for ibl. The `.ckpt` holds two
groups of weights:

| term | what it is |
|---|---|
| **backbone** | DINOv3 ViT-S, 21.6 M parameters in 211 tensors: patch embedding, 12 transformer blocks (attention W_q W_k W_v W_o + MLP fc1/fc2 each), final norm. Frame → 16×16 grid of 384-dim features. Trained on the n−1 mouse datasets in the super-mouse checkpoint; Meta's self-supervised weights in the DINO arm. |
| **head** | One layer, 31 k parameters: parameter-free PixelShuffle (384 → 96 channels, 2× upsample), then a single 3×3 transposed conv `96 → 36` (weight shape (96, 36, 3, 3)), then spatial softmax. Its 36 output filters are the 36 canonical keypoint channels — **filter *i* is keypoint *i***. |
| **supported filter** | A head filter whose keypoint at least one of the n−1 training datasets labels — it has been trained. ibl: nose_tip, pupil_center_left, wrist_left, wrist_right. |
| **untrained filter** | A filter whose keypoint no training dataset labels. It never received a gradient and sits at initialization. Leave-cheese: 11 (pads, ear bases/tops, upper lips); leave-ibl: 2 (tongue_end pair); leave-kondo: 0. |

Important: the "novel" keypoints of our datasets are **not new channels** — every keypoint
all five datasets label is already one of the 36. They are existing filters that have not
been taught yet. ("Trunk" in earlier notes meant backbone + head; this note says backbone
and head.)

## 1. What fine-tuning does today (`fewshot-exp-lr5/`, `fewshot-exp/`)

Each cell is one `litpose train` call:

1. Build the model from `configs/model.yaml`: fresh DINOv3 backbone + fresh 36-filter head.
2. Overwrite both with the n−1 checkpoint (`+model.checkpoint=…`, logged as
   *loading weights from*). Backbone: 21.6 M trained weights. Head: supported filters
   trained, untrained filters at init.
3. Pick the N frames of the target dataset (`train_frames=N`, `rng_seed_data_pt=draw`);
   the same seed gives the same frames to every arm.
4. Train 2000 steps, one Adam group at lr 5e-5 (or 1e-5), backbone unfrozen from step 1,
   lr halving at step 1000. Loss = heatmap MSE on labeled keypoints only; unlabeled
   channels get no gradient.
5. **Every weight moves** — all 211 backbone tensors and all 36 head filters, same lr.
6. Validate at steps 1000 and 2000, keep the better; evaluate on the dataset's test
   frames. Result: a private 90 MB checkpoint per (dataset, N, draw).

The DINO arm (`fewshot-exp-dino/`) runs the identical loop with Meta's backbone weights
and a fully random head — no supported/untrained split, all 36 filters from scratch.

Why one learning rate cannot serve both jobs (draw 0 numbers):

| lr | supported filters | untrained filters |
|---|---|---|
| 1e-5 (`fewshot-exp`) | kept — cheese 19.1 px at N=50, better than the dedicated model (21.5) | never learned — cheese full set 55 px |
| 5e-5 (`fewshot-exp-lr5`) | forgotten — cheese 28 → 43 px at N=10 | learned — cheese full set 25 px |

## 2. The same steps with LoRA

Same checkpoint, same frames, same 2000 steps, same loss. Steps 1–3 and 6 unchanged.
Steps 4–5 become:

- **Backbone base weights frozen.** Beside each of the 6 linear layers in each of the 12
  blocks (72 matrices) add a low-rank pair `B·A` (B initialised to zero, so step 0 is
  *exactly* the n−1 model). Only A and B train: ≈ 0.6 M parameters at rank 16 instead of
  21.6 M. Whatever the backbone learned from the n−1 datasets is preserved by construction.
- **Head handled per filter.** The head is one tensor and filter = keypoint, so a
  per-channel rule is a one-line gradient mask:
  - supported filters: frozen, or lr 1e-5 (decision A);
  - untrained filters: full lr 5e-5 or higher — learned from scratch, which is what they need.
  Which filters are which is read from `dataset_inventory.json`, exactly as the
  zero-shot scoring rule does.
- **What is saved**: the 72 (A, B) pairs + the head filters that moved ≈ 1–3 MB.
  Inference = frozen base ⊕ adapter (peft merges, or applies on the fly — identical
  outputs). Drop the adapter → the n−1 model, bit-exact. Second rig → second adapter on
  the same checkpoint, no interference. That is the continual-learning property.

Expected: supported keypoints ≈ the 1e-5 line (nothing can drift far), untrained
keypoints ≈ the 5e-5 line (their filters train at full rate).
Not expected: faster training — the backward pass still traverses the whole ViT-S;
wall-clock stays ≈ 10 min per cell. The gain is retention, adapter size, and swapping.

## 3. With the DINO backbone

Mechanically identical: the same LoRA pairs on the same 72 matrices. What differs:

- The frozen base is Meta's DINOv3, which has never seen a labeled mouse. Freezing it
  preserves generic visual features, not pose knowledge; the low-rank pairs must build
  pose-specific features from N frames alone.
- The head is fully random, so all 36 filters train at full lr (31 k params, cheap).
- Expected: DINO + LoRA clearly worse than DINO + full fine-tuning at N=10. That is the
  point — with adaptation capacity constrained, the value of the super-mouse backbone
  becomes explicit. It is the identical-protocol baseline for the LoRA arm, as `dino` is
  for `trunk5`.

## 4. A lab whose keypoints fall outside the 36

Not our datasets. For a future lab: append k new filters to the head's single conv
(36 → 36 + k), train them at full lr, treat the 36 as above. The registry already carries
the mapping; the change is confined to the head's output dimension and the checkpoint
loader. Build only when such a lab exists.

## 5. A lab's workflow

```
super-mouse checkpoint (backbone + head, one download, never edited)
    │
    ├─ alone ──────────────────────► zero-shot on any rig (supported keypoints, 0 labels)
    │
    ├─ rig 1: label N frames ─► train adapter A (2000 steps, ~10 min L4) ─► infer: checkpoint ⊕ A
    │
    └─ rig 2: label N frames ─► train adapter B ────────────────────────► infer: checkpoint ⊕ B
                                   swap A ↔ B: no retraining, no forgetting
```

## 6. Build steps, in the order they unblock each other

1. **Per-filter head learning rates** — gradient mask on the head's (96, 36, 3, 3) tensor
   by output channel: supported filters at lr_keep (0 or 1e-5), untrained filters at
   lr_new (5e-5+); membership from `dataset_inventory.json`. ~30 lines in
   `models/base.py` (optimizer groups). Needed with or without LoRA, and probe-able today
   on the existing full-fine-tune path — if it alone fixes cheese retention, LoRA becomes
   the deployment story rather than the accuracy story.
2. **LoRA on the backbone** — `peft.LoraConfig` on the HF DINOv3 module in
   `backbones/vit_dino.py`: targets query/key/value/o-proj + fc1/fc2 (72 matrices), rank
   r, α = 2r, dropout 0, base frozen. LoRA params form the "backbone" optimizer group so
   the existing unfreeze callback keeps working. Config: `model.lora.rank`,
   `model.lora.targets`.
3. **Save / load / predict** — checkpoint = reference to the n−1 checkpoint + adapter
   state (A, B pairs + moved head filters). `Model.from_dir` rebuilds base → applies
   adapter → loads. Predict path unchanged. Round-trip must be bit-exact.
4. **Verification gate before any run** — (a) B = 0 at step 0 reproduces the n−1
   predictions to 1e-6; (b) after 50 steps frozen tensors are bit-identical and grad norms
   are non-zero only on A, B and untrained filters; (c) the N-frame split matches the
   existing cells (same `rng_seed_data_pt`).
5. **Probe — 6 cells, under an hour** — cheese-2d and ibl at N=10 and 50, draw 0,
   rank 16; plus rank 4 and 64 on cheese N=50. Pass: cheese supported ≤ 26 px (no-regret
   vs zero-shot 25.9) *and* full ≤ 25 (match DINO); ibl supported within 0.5 px of full
   fine-tuning.
6. **Grid** — same 5 datasets × 3 N × 3 draws, one rank, through the existing
   `fewshot_cell.sh` pool as new arms `lora` (root `fewshot-exp-lora/`) and `dino-lora`.
   Scored by `fewshot_score.py`; the figure gains two series.
7. **Continual demo (paper figure)** — two adapters on one checkpoint (e.g. cheese and
   cazettes): each rig scored under its own adapter, the other's, their average, and the
   bare checkpoint. Shows no interference; tests whether adapters compose.

## 7. Experiment arms

| arm | backbone base | what trains | role |
|---|---|---|---|
| `dino` | DINOv3 (Meta) | everything, lr 5e-5; head random | baseline — what a lab does today |
| `trunk5` | super-mouse n−1 | everything, lr 5e-5 | plasticity upper bound |
| `trunk` | super-mouse n−1 | everything, lr 1e-5 | retention reference |
| `dino-lora` | DINOv3, frozen | LoRA pairs + all 36 head filters | identical-protocol baseline for the method |
| `lora` | super-mouse n−1, frozen | LoRA pairs + untrained head filters | **the method** |

Evaluation unchanged: same frames (draw seeds 0–2), 2000 steps, validation at 1000/2000
only, both keypoint sets (supported / all labeled), zero-shot and dedicated anchors,
pooled mean pixel error over test (frame, keypoint) cells, `pupil_center_right` excluded.

## 8. Decisions

- **A.** Supported head filters: fully frozen, or lr 1e-5? Default: frozen.
- **B.** Run step 1 alone first as a same-day probe on the full-fine-tune path? Default: yes.
- **C.** Rank probe {4, 16, 64} on cheese N=50, then one rank for the grid.
- **D.** `dino-lora` on the full grid or draw 0 only? Default: full grid.

## Honest expectations

Likely wins: cheese-2d retention restored while untrained keypoints train; ibl/kondo
supported at or near the 1e-5 line with the 5e-5 line's full-set numbers; DINO + LoRA
clearly weak at N=10, making the backbone's value explicit; 1–3 MB adapters on one
shared checkpoint.

Risks: under-adaptation on large-shift rigs (facemap, cazettes at N=50) where full
fine-tuning's plasticity paid off — visible as `lora` > `trunk5` there; mitigate with
rank or by LoRA-ing the head too. Not faster to train. One new hyperparameter (rank),
kept to a 3-point probe. Engineering surface: optimizer groups, checkpoint format,
predict path — about a day plus the verification gate.

## Outcome of the probe (2026-08-27)

Implemented and verified (branch `fewshot_head` of the lightning-pose dev worktree; LoRA via
`LoRALinear`, no peft; reload bug fixed and re-verified before scoring). Draw 0:

| cheese-2d N=10 | supported | not supported | all |
|---|---|---|---|
| zero-shot | 25.9 | 147 | 68.1 |
| full FT 5e-5 | 38.8 | 40.9 | 39.5 |
| LoRA ad.5e-4 | 40.9 | 42.2 | 41.4 |
| LoRA ad.5e-5 | 36.0 | 61.8 | 45.0 |
| LoRA ad.5e-5 + head 5e-4 | **35.8** | **40.5** | **37.4** |
| backbone + supported filters frozen | **25.9** | 90.7 | 48.4 |

- Head-freeze alone: no effect (forgetting is in the backbone). Closed.
- LoRA r16 @ adapter lr 5e-4 matches full fine-tuning on ibl and kondo at every N — the
  method form for the paper: same accuracy, frozen trunk, 1–3 MB adapters.
- Retention on cheese at N≤25 cannot be had from a single model: adapting features enough to
  learn the new keypoints costs ≥10 px on the supported ones; freezing features keeps them
  exactly but the head alone learns the new ones poorly. Best single model: adapters 5e-5 +
  head 5e-4 — and at N=50 that split solves cheese outright (22.9 supported < zero-shot 25.9,
  23.9 new, 23.3 all; DINO 23.7, full FT 24.7, dedicated 21.3). Recipe for this regime: **per-keypoint routing** (frozen trunk for supported,
  adapted model for new; selected on the validation split) — ≈31 px all-keypoint at cheese
  N=10 vs 37–41 for any single arm. Prediction-time option; not yet implemented.

