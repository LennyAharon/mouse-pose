#!/bin/bash
# Dedicated single-dataset model on a subset of N train frames, drawn EXACTLY like an anchor_ft.sh draw
# (same csv, stock split: data.dataset_names null -> global random_split seeded by rng_seed_data_pt=<draw>,
# train_prob 0.95 / val_prob 0.05, first N shuffled train indices), so it trains on the same N frames and
# validates on the same frames as `scripts/anchor_ft.sh <trunk> <ds> N <steps> <out> <draw>`.
# Recipe = the v1 dedicated recipe (configs/model.yaml: pretrained DINOv3 backbone,
# random head, stock dlc aug, 12k steps, val every 250), model seed = draw. Checkpoint kept.
#
#   scripts/dedicated_subset.sh <dataset> <n_frames> <draw> <out_dir> [backbone=vits_dinov3]
set -u
DS="$1"; N="$2"; DRAW="$3"; OUT="$4"; BACKBONE="${5:-vits_dinov3}"
cd "$(dirname "$0")/.."
DATA=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['data_dir'])")
[ -f "$OUT/.done" ] && { echo "skip (done): $OUT"; exit 0; }
mkdir -p "$(dirname "$OUT")"; LOG="$OUT.log"
if [ -d "$OUT" ]; then mv "$OUT" "$OUT.partial-$(date -u +%m%d%H%M)"; fi
echo "=== [$(date -u +%H:%M)] dedicated $BACKBONE $DS N=$N draw=$DRAW"
litpose train configs/model.yaml --output_dir "$OUT" --overrides \
    data.data_dir="$DATA" "data.csv_file=CollectedData_${DS}_train.csv" \
    model.backbone="$BACKBONE" "model.losses_to_use=[]" \
    training.train_frames="$N" training.rng_seed_data_pt="$DRAW" training.rng_seed_model_pt="$DRAW" \
    > "$LOG" 2>&1
echo "train exit $?"
grep -q COMPLETED "$OUT/train_status.json" || { echo "ABORT: training not COMPLETED"; exit 1; }
grep -q "stratified split" "$LOG" && { echo "ABORT: stratified split used -- frames differ from the anchored draw"; exit 1; }
grep -q "dataset splits -- train: $N," "$LOG" || { echo "ABORT: train split is not $N frames"; exit 1; }
python -m mighty_mouse.train --output_dir "$OUT" --csv_file "CollectedData_${DS}_train.csv" --keep_checkpoints \
    > "${LOG%.log}-eval.log" 2>&1
E=$?
echo "eval exit $E - $OUT"
[ $E -eq 0 ] && [ -f "$OUT/eval/$DS/predictions.csv" ] && touch "$OUT/.done"
