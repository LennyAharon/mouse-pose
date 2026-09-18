# Results catalog — what every folder under `results/head-fixed` means

## TL;DR — the trained models, in one glance

Three kinds of models exist:

1. **Trunks** (36-kp supermodels, 12k steps, leave-one-out or all-data). Three aug
   generations: `zoom-aug-exp/*-T2-zoominout` = **final**; `*-T2-zoomaug` = older
   (zoom-out only); `face+…_train` in the root = oldest (no zoom aug, kept as baseline).
2. **Dedicated single-dataset models** — `<ds>_train` (facemap, ibl, cheese-2d,
   cazettes-side, kondo). Upper-bound reference.
3. **Few-shot fine-tuned models** — trunk + N=10/25/50 frames × 3 draws. Four current
   families: `fewshot-exp-lr5-zio` (full FT), `…-lora-r16-lr5e-5-head5e-4-zio` (LoRA),
   `…-anchor-lora-conf1-zio` (**anchored LoRA, our method**), `…-dino` (from scratch).
   Everything else `fewshot-exp-*` is a variant: `-mask-…` = labels hidden for the
   transfer measurement; the rest are ablations or superseded probes.

Rule of thumb: a few-shot root **without `-zio` is not current**. Details below.

Purpose: when making videos/figures, pick the **correct** run for each role. The roles are at
the top; the full name grammar and per-folder table follow. `<results>` is `results_dir` from
`paths.yaml` (here `poseinterface/results/head-fixed`).

Updated 2026-09-18 after the results cleanup (`scripts/cleanup_results_v1.py`, plan in
`docs/results_cleanup_plan.md`, manifest at `<results>/_cleanup_manifest.json`). This tree is
**data version 1, frozen**: superseded lines and closed probes now live under `<results>/_archive/`
(evals only), checkpoints were deleted wherever a model never needs to run again, and the masked
and plain-LoRA roots keep checkpoints for **draw 0 only**. Rows marked *archived* below are in
`_archive/`. Results for the next data version go to a new tree.

## The five model roles for videos and figures (USE THESE)

| Role | Path | Checkpoint? |
|---|---|---|
| **Zero-shot / teacher** (leave-one-out trunk, recipe of record) | `zoom-aug-exp/<combo>-T2-zoominout/seed0` where `<combo>` omits the target dataset (e.g. facemap videos → `ibl+cheese+caz+kondo-T2-zoominout`) | yes |
| **Anchored LoRA (our method)** | `fewshot-exp-anchor-lora-conf1-zio/<ds>/tf<N>-draw<d>` | yes |
| **Plain LoRA baseline** | `fewshot-exp-lora-r16-lr5e-5-head5e-4-zio/<ds>/tf<N>-draw<d>` | yes |
| **Full fine-tuning baseline** | `fewshot-exp-lr5-zio/<ds>/tf<N>-draw<d>` | no (eval CSVs only) |
| **Dedicated single-dataset model** (upper bound) | `<ds>_train/` (e.g. `facemap_train`) | check per run |

Also: `fewshot-exp-dino/<ds>/tf<N>-draw<d>` = DINO-from-scratch baseline (no trunk).

Every run dir has `eval/<dataset>/predictions.csv` on the canonical test frames — videos can
be rendered from those with no GPU. Fresh predictions (new clips, video inference) need a
checkpoint under `tb_logs/test/version_*/checkpoints/`; full-FT cells deleted theirs.

Conventions that always apply: `pupil_center_right` is excluded (hflip-only channel, zero
labels anywhere, conf ≈ 0 is expected). facemap `videos_test` sessions are eye-OOD for **all**
models (eye cluster conf ≈ 0) — use labeled test frames for facemap eye comparisons.

## Name grammar

**Few-shot roots** — `fewshot-exp[-<arm>][-<trunk>][-mask-<kps>]`, cells `<ds>/tf<N>-draw<d>`
(fine-tune on N frames, label draw d ∈ {0,1,2}; all arms 2000 steps unless noted).

Arm tokens:

- *(none)* / `-lr5` — full fine-tuning of the trunk: bare `fewshot-exp` = lr 1e-5 (early line,
  paused), `-lr5` = lr 5e-5 (the real full-FT baseline). `-lr5-zio` is THE full-FT baseline.
- `-dino` — ImageNet/DINO backbone trained from scratch on the N frames (no supermouse trunk).
- `-lora-r16-lr5e-5-head5e-4` — LoRA rank 16, adapters 5e-5, head 5e-4.
- `-anchor-lora-conf1` — the method: LoRA + anchoring loss (frozen zero-shot trunk = teacher on
  keypoint channels the target does not label), teacher-confidence weighting power 1.
- `-anchor` / `-anchor-w10` / `-anchor-lora-w10` / `-anchor-lora-w0.2-conf1` — ablations:
  anchor on full-FT form; anchor weight 10 / 0.2 (recipe uses weight 1).
- `-anchor-lora-video-conf1[-s4000]` — video-anchor ablation (teacher loss on unlabeled video
  frames; `s4000` = 4000 steps).
- `-headfreeze` / `-backfreeze` — freeze probes (both closed negatives).
- `-replay` / `-replay-lora` — replay baseline (mixes source-dataset frames into fine-tuning).
- `-xfer-cheese2d` — init-from-cheese-single probe (closed negative).

Trunk tokens (which trunk the arm fine-tunes; also selects the teacher for anchor arms):

- `-zio` — T2-**zoominout** trunks, the recipe of record. **Current results end in `-zio`.**
- `-zoomaug` — explicit old-trunk (T2-zoomaug) rerun kept for the trunk-comparison table.
- *(no token)* — implicitly the old T2-zoomaug trunk (roots predating the suffix). Superseded.

`-mask-<kps>` — masked-label protocol: `<kps>` was hidden from the training CSV (coords NaN,
visible 0) although the dataset really labels it, then scored against the held-back labels.
This is the "does fine-tuning destroy unlabeled keypoints?" measurement. Wrists are always
masked as a pair. Score with `scripts/fewshot_masked_score.py`; regular roots score with
`scripts/fewshot_score.py`.

**Trunk roots** — `zoom-aug-exp/<combo>-<temp>-<aug>[-20k]/seed0`, where `<combo>` is `+`-joined
dataset short names (`face`, `ibl`, `cheese`, `caz`, `kondo`); a missing name = leave-one-out
trunk for that dataset; all five = the all-data trunk. `T1`/`T2` = sampling temperature,
`-zoomaug` = zoom-out-only aug, `-zoominout` = per-dataset zoom-in+out pairs (recipe of
record), `-20k` = 20k-step probe (closed), `-perds` = per-dataset zoom probe (closed).
Warning: the **all-data** trunks (both augs) place the facemap pupil ~36 px off with high
confidence — use the leave-one-out zio trunk for facemap zero-shot demos.

## Full folder table

| Folder | What it is | Status |
|---|---|---|
| `zoom-aug-exp/*-T2-zoominout` | 6 recipe-of-record trunks (5 LOO + all-data), 12k steps | **CURRENT** |
| `zoom-aug-exp/*-T2-zoomaug` | zoom-out-only trunks | kept for trunk comparison |
| `zoom-aug-exp/*` (T1, `-20k`, `-perds`) | trunk-recipe probes | archived (`-20k`, `-perds`, T1-zoomaug); T1/Tinf/headperds-zoominout kept, evals only |
| `fewshot-exp-anchor-lora-conf1-zio` | **anchored LoRA, primary grid** (45 cells, 3 draws) | **CURRENT** |
| `fewshot-exp-lora-r16-lr5e-5-head5e-4-zio` | plain-LoRA baseline grid (45) | **CURRENT**; ckpts draw 0 only |
| `fewshot-exp-lr5-zio` | full-FT baseline grid (45) | **CURRENT** |
| `fewshot-exp-dino` | DINO-from-scratch grid (45) | **CURRENT** |
| `fewshot-exp-*-zio-mask-pupil_center_left` (3 arms) | masked protocol on zio trunks (6 settings, 3 draws) | **CURRENT**; ckpts draw 0 only |
| `fewshot-exp-replay-lora-zio` | replay baseline on zio | done, evals only |
| `<ds>_train` (5 single-dataset) | dedicated per-dataset models | **CURRENT** (upper-bound baseline) |
| `face+…_train`, `ibl+…_train` (6 multi-dataset) | original plain-DLC-aug LOO/all trunks | baseline for the augmentation finding |
| `face+ibl+cheese+caz+kondo_train/supervised/{sampling-T*,head-per_dataset/*}` | **the trunk recipe-search grid**: shared/per-dataset head × T=1/2/∞ (DLC aug, seed0; per-ds heads have `eval` = oracle routing and `eval_blind` = blind); shared×T=1 also has seeds 1–2 | recipe-search appendix (shared T=2 adopted) |
| `fewshot-exp`, `fewshot-exp-lr5` | full FT on old trunk (lr 1e-5 / 5e-5) | archived (superseded by `-lr5-zio`) |
| `fewshot-exp-lora-r16-lr5e-5-head5e-4` | plain LoRA, old trunk (25 cells) | archived |
| `fewshot-exp-anchor-lora-conf1` | anchored LoRA, old trunk | archived |
| `fewshot-exp-{anchor-lora-conf1,lora-…,lr5}-zoomaug` | explicit old-trunk reruns | archived (trunk-comparison table) |
| `fewshot-exp-*-mask-pupil_center_left` (non-zio) | original masked numbers, old trunk | archived |
| `fewshot-exp-anchor`, `-anchor-w10`, `-anchor-lora`, `-anchor-lora-w10`, `-anchor-lora-w0.2-conf1` | anchor form/weight ablations | archived |
| `fewshot-exp-anchor-lora-video-conf1[-zio,-s4000]` | video-anchor ablation (+ `-temporal`) | archived, neutral |
| `fewshot-exp-headfreeze`, `-backfreeze` | freeze probes | archived |
| `fewshot-exp-replay`, `-replay-lora` | replay on old trunk | archived |
| `fewshot-exp-xfer-cheese2d` | cheese→facemap init probe | archived |
| `token-exp`, `scale-exp` | recipe-search-era experiments | archived |
| `aug_check` | augmentation visualizations | reference |
| `qualitative` | every delivered video/figure/table — see `docs/qualitative_catalog.md` | deliverables |

**Never use for videos/figures:** any non-`-zio` few-shot root (unless the point *is* the
trunk comparison), the all-data trunk for facemap eye keypoints, `pupil_center_right`, or
facemap `videos_test` clips for eye keypoints.
