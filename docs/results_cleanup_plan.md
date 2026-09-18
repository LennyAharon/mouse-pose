# Results cleanup plan (2026-09-18) — PROPOSED, nothing executed yet

`results/head-fixed` = **46 GB in 88 top-level roots; 42.6 GB of that is checkpoints (`*.ckpt`)**.
Every root is fully evaluated (eval CSVs present for every cell, except 2 cells of the paused
`fewshot-exp` and 1 of `headfreeze`), so checkpoints are only needed where a model must run again
(videos, teachers for adaptation, ensembles). Evaluation CSVs are small (whole tree without
checkpoints ≈ 3.5 GB) and are what every figure, table and score script reads.

Nothing here is a live input to the other session's `experiments/` (its references to old roots
are inside copied docs/patches only), and no training is running.

## Principles

1. **Never delete an evaluation.** Only checkpoints and regenerable intermediates are deleted.
2. **Do not rename active roots.** `paper_figures/`, `fewshot_cell.sh`, both catalogs and the
   analysis scripts hard-code the current names; renaming buys nothing before the data changes.
3. **Superseded and closed work moves to `_archive/`** (evals only), so the top level shows only
   what is current. The catalog gets a matching `_archive` section.
4. **This tree is data version 1.** When the datasets change, results go to a NEW tree
   (`results/head-fixed-v2/`, layout below) via `paths.yaml`; this one is frozen with a README.

## Tier A — keep in place, checkpoints kept (the models of record)  ≈ 10 GB after cleanup

| root | why the checkpoint stays |
|---|---|
| `zoom-aug-exp/*-T2-zoominout` (5 LOO + all-data seeds 0/1/2) | teachers, zero-shot demos, 3-seed ensemble |
| `fewshot-exp-anchor-lora-conf1-zio` (45 cells) | the method; videos, any re-prediction |
| `fewshot-exp-anchor-lora-conf1-zio-mask-*` **draw 0 only** (6 settings) | masked-protocol videos (Fig. 5 video) |
| `fewshot-exp-lora-r16-lr5e-5-head5e-4-zio` and its `-mask-*` roots, **draw 0 only** | baseline videos |
| `<ds>_train` (5 dedicated models) | upper-bound reference (small) |

Draws 1 and 2 of the anchored-masked and all LoRA roots keep their evals and lose their
checkpoints: no figure or video uses those checkpoints, only their CSVs. (-7.5 GB masked, -3.1 GB LoRA grid)

## Tier B — keep in place, checkpoints deleted (results still used, model never re-run)  ≈ 1 GB

- `zoom-aug-exp/*-T2-zoomaug` (6): the zoom-out-only series in `fig_aug` — evals only. (-1.5 GB)
- `zoom-aug-exp/*-{T1,Tinf}-headperds-zoominout`, `*-T1/Tinf-zoominout`: `fig_arch` zio grid. (-0.8 GB)
- `face+ibl+cheese+caz+kondo_train/supervised/*` (DLC recipe grid) and the 5 LOO `*_train`
  DLC-aug trunks: `fig_arch` DLC grid and the DLC series in `fig_aug`. (-2.3 GB)
  *Judgment call:* the 3-seed DLC-aug all-data ensemble checkpoints go too; the paper's ensemble is the zio one.
- `fewshot-exp-lr5-zio`, `fewshot-exp-dino`, all `-lr5-zio-mask-*`, `-dino-mask-*`,
  `-psl-replay-*`, `-replay-lora-zio[-mask]`: evals only (full FT and DINO never kept ckpts; replay ckpts go, -0.5 GB).

## Tier C — move to `_archive/` with checkpoints deleted (superseded / closed)  ≈ 1.5 GB after

Old-trunk (pre-`-zio`) lines: `fewshot-exp`, `fewshot-exp-lr5`, `fewshot-exp-lora-r16-lr5e-5-head5e-4`,
`fewshot-exp-anchor-lora-conf1`, the three non-zio `-mask-pupil_center_left` roots.
Trunk-comparison reruns: `fewshot-exp-{anchor-lora-conf1,lora-r16-lr5e-5-head5e-4}-zoomaug`.
Ablations/probes (all reported or closed): `fewshot-exp-anchor`, `-anchor-w10`, `-anchor-lora`,
`-anchor-lora-w10`, `-anchor-lora-w0.2-conf1[-zio-mask-pupil]`, `-anchor-lora-video-conf1*` (incl.
`-temporal`, `-s4000`), `-anchor-conf1-zio-mask-pupil`, `-*-ema0.999*`, `-*-wf0.05*`, all `-s4000`,
`-headfreeze`, `-backfreeze`, `-replay`, `-replay-lora`, `-xfer-cheese2d`.
Trunk probes: `zoom-aug-exp/*-20k`, `*-perds-zoomaug`, `*-T1-zoomaug`; `token-exp`; `scale-exp`.
(≈ -12 GB of checkpoints)

## Tier D — regenerable intermediates in `qualitative/` (delete)  ≈ -0.5 GB

`ibl-facecrop-transfer/_work` (cropped PNGs), `transfer-traces-200f/_clips` (clips + per-seed CSVs; keep
the CSVs, drop the mp4 clips), `whisker-heatmaps/ear_why/_occl` + `_ibl__occl`, `whisker-heatmaps/*/heatmaps.npz`
larger than 20 MB, `ear_finetune/data` (frame copies). All are rebuilt by their scripts' `predict` step.
Delivered videos, figures, CSVs and scripts stay.

## Expected outcome

46 GB → **≈ 19 GB**; 88 top-level roots → **≈ 45** (rest under `_archive/`). A `README.md` at the
tree root states: data version 1, frozen 2026-09-18, which roots are current, where archives went.

## Layout for the next data version (`results/head-fixed-v2/`)

```
head-fixed-v2/
  README.md                  data version, corpus hash, recipe of record
  trunks/<combo>-<recipe>/seed<k>/       (today's zoom-aug-exp)
  dedicated/<dataset>/seed<k>/           (today's <ds>_train)
  fewshot/<arm>/<variant>/<dataset>/tf<N>-draw<d>/   arm ∈ {anchor-lora, lora, full-ft, dino}, variant ∈ {base, mask-<kps>, ...}
  baselines/<name>/                      (replay, psl-replay, ...)
  probes/<name>/                         (anything exploratory; checkpoints deleted on close)
  qualitative/<delivery>/                (as now, with the catalog)
  _archive/                              (never at the top level)
```
`fewshot_cell.sh` and `train_sweep.py` get the new root layout when the switch happens; the
catalogs start fresh for v2 and this tree's catalog stays with it.

## Execution order (once approved)

1. `find` + delete checkpoints in Tiers B, C (and draws 1–2 of the Tier A roots listed).
2. `mv` Tier C roots into `_archive/`; write `_archive/README.md` listing what each was.
3. Delete Tier D intermediates.
4. Write `results/head-fixed/README.md`; update `docs/results_catalog.md` (archive section, "draw 0 only" notes).
5. Re-run one figure script (`make_fig_aug.py` to the scratchpad) and one score script as a smoke test.
