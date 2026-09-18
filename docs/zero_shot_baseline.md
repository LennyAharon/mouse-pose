# Consolidated research baseline — 2026-09-17

Active branches are `post_sub_mm` in mouse-pose and `post_sub_lp` in Lightning Pose, renamed
from `research/zero-shot-baseline` on 2026-09-18 without changing the baseline code. Original `zoom_aug`
(mouse-pose) and `super_mouse_paper` (Lightning Pose) branches are preserved. Existing
mouse-pose local edits were copied and hashed before integration, and remain uncommitted.
This branch changes the installed code intentionally; previous experiment worktrees remain
frozen at their recorded commits. Historical experiment guards expecting original branch/HEAD
will reject this new checkout state; do not edit old manifests to bypass those guards.

## Keep and defer

Keep shared36, T2, zoom-in/out and original visibility-aware heatmap supervision, corrected
optimizer-step LR milestones4000/6000/8000 over12000 updates, and the existing anchored-LoRA
method. No longer-training experiment. Keep all keypoints as training targets.

The schedule candidate improved lip-excluded normalized all-data error6.1% versus a matched
control and7.7% versus the original. It remains unverified for zero-shot, and Cheese worsened.
Neither zero-shot claims nor automatic replacement of existing teacher checkpoints follow.

Do not integrate tongue occlusion downweighting or clean-teacher anchor changes: they improved
selected points but failed preservation/confidence checks. Their isolated branches, data,
checkpoints and reports remain available. No temporal objective is enabled yet.

New evaluations write original and `no_lips` comparison summaries. Exclude upperlip_left,
upperlip_right, lowerlip only from comparisons; retain mouth. Always exclude pupil_center_right.
Use visible==2 without confidence filtering, align frames/channels and reject invalid visible
predictions. Raw error CSVs keep lip channels. Record dataset and keypoint breakdowns plus
normalized macro; do not mix exclusions between compared methods.

## Ready recipes

`configs/zero_shot/vits_dinov3.yaml` and `vitb_dinov3.yaml` fix the shared training recipe,
including explicit imgaug_seed0. The installed LP entry point honors this optional seed;
existing configurations without it retain their prior augmentation-seeding behavior.

Use matching source-only data splits and independently pretrained backbones. ViT-S pose weights
cannot initialize ViT-B. The canonical dataset registry lists all five datasets even when the
training CSV omits the target; verify actual train/validation manifests, not registry names.
Select checkpoints on source validation only. Do not use target videos or labels in training.

Example preview for Cazettes held out (run from the mouse-pose directory):

```bash
python scripts/train_sweep.py --dry_run \
  --config_file configs/zero_shot/vitb_dinov3.yaml \
  --csv_files CollectedData_face+ibl+cheese+kondo_train.csv \
  --backbones 'vits_dinov3;vitb_dinov3' --sampling_temperatures 2 --head_modes shared \
  --seeds 0 --keep_checkpoints --stop_on_failure \
  --output_root /teamspace/studios/this_studio/experiments/zero-shot-backbones/results
```

The backbone CLI argument overrides the config backbone for each cell. Configs otherwise
match. Other held-out tags: Facemap=`ibl+cheese+caz+kondo`; IBL=`face+cheese+caz+kondo`;
Cheese=`face+ibl+caz+kondo`; Kondo=`face+ibl+cheese+caz`.

Keep output roots unique, run smoke attempts separately, and retain checkpoints. Hold the
existing `experiments/teacher-consistency/runs/.sequential.lock` using `flock -n` around each
actual local sequence. The general sweep itself is sequential but does not acquire this
cross-experiment lock. Exact data-stream resume remains unsupported: restart interrupted
attempts in new output directories. Do not reuse the smoke directory for full training.

ViT-B weights are cached under `experiments/consolidation-2026-09-17/hf-cache`, official revision
`5931719e67bbdb9737e363e781fb0c67687896bc`. Set HF_HUB_CACHE to that directory for offline ViT-B
runs. The cached ViT-S directory is linked into the same experiment cache, so joint offline sweeps
can set HF_HUB_CACHE to this directory with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.
No packages were installed. The ViT-B source-only GPU smoke passed32 updates at batch32,
with peak allocated memory7.80GiB; this is compatibility evidence, not pose accuracy.

## Masking research sequence (not enabled in baseline)

1. Compare the existing CoarseDropout baseline against modest spatial block masking on the
   same ViT-S recipe first. Keep visible GT targets even when pixels are artificially hidden;
   do not reinterpret artificial masks as real vis1/unknown labels. Use mixed clean/masked
   images and preserve useful context. Verify targets remain identical and val/inference
   are unmasked. Randomize mask locations and sizes independently of GT to avoid shortcuts.
2. If promising, test anatomy-region masking with jitter and non-keypoint control masks, or
   clean-view teacher / masked-view student consistency. Preserve supervised GT as the
   authority; teacher errors can otherwise be reinforced. These are separate ablations.
3. Annotation-group masking requires a held-back training-label prediction objective; merely
   removing labels supplies no new signal. Treat this separately from image masking.

The existing LP PatchMasking callback is gated to heatmap_multiview_transformer and expects
multiview tensors. It is NOT active for Mighty Mouse's single-view heatmap model. A suitable
single-view implementation and tests are required; adding patch_mask fields alone would not
run the experiment. Masking may improve contextual reasoning, but can also erase the only
useful pupil/tongue evidence. Require zero-shot gains on ordinary unmasked images, not just
synthetic-occlusion robustness. No novelty claim has been established.

## Recovery

The backup is `/teamspace/studios/this_studio/experiments/consolidation-2026-09-17/before.json`,
with copied local files and separate working/index patches. Returning to the old code can use
`git switch zoom_aug` in mouse-pose and `git switch super_mouse_paper` in lightning-pose;
first inspect status and preserve any subsequent edits. Do not reset/clean. Old checkpoints
and source labels were never rewritten. All uncommitted files from before integration are
verified against their recorded hashes at completion.
