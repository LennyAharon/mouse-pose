#!/bin/bash
# Anchored-LoRA fine-tuning of any trained trunk (ViT-S or ViT-B) on one dataset's train frames.
# Recipe of record from the few-shot grid (fewshot-exp-anchor-lora-conf1-zio): LoRA r16 on the
# backbone (adapters 5e-5, head 5e-4), frozen trunk as teacher distilled into every channel the
# trunk trained that the frame does not label (weight 1, teacher-confidence power 1), stock dlc
# aug, no temperature sampling, backbone base frozen, lr (head and adapters) halves at mid-run (as in
# every earlier anchored run), validation every VAL_EVERY steps (env, default 500; user 2026-09-29) --
# the best-validation checkpoint is the one evaluated.
#
#   scripts/anchor_ft.sh <trunk_run_dir> <dataset> <n_frames|all> <steps> <out_dir|auto> [draw=0]
#   out_dir=auto (use this): finetune/<trunks|trunks_24k>/<tag>_train/<backbone>-seed<k>/<method>/<target>/
#   tf<N>-s<steps>-draw<d>, plus <run>/run_info.json (trunk, checkpoint, method, settings, frames, draw, commits,
#   date, result; scripts/write_run_info.py)
#
# The trunk's backbone comes from its config.yaml; its datasets from the frames of the training csv
# it saved (config dataset_names lists every registry dataset, not only the trained ones); anchored
# channels = the union of `trainable` keypoints of those datasets (dataset_inventory.json). After training, the
# model is evaluated on every dataset's test csv (mighty_mouse.train, checkpoints kept) and the
# reloaded model is checked to carry the trained adapters. Writes <out_dir>/.done on success.
set -u
TRUNK="$1"; DS="$2"; N="$3"; STEPS="$4"; OUT="$5"; DRAW="${6:-0}"
TRUNK="${TRUNK%/}"
cd "$(dirname "$0")/.."
DATA=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['data_dir'])")
RESULTS=$(python -c "from mighty_mouse.paths import load_paths; print(load_paths()['results_dir'])")
# ablation knobs (env; defaults = recipe of record): LORA_RANK, LORA_LR (adapters), HEAD_LR (head), ANCHOR_W,
# FULL_FT=1 -> no LoRA, the whole backbone trains at FT_LR (default 5e-5, the few-shot `anchor` arm) with the anchor
LORA_RANK="${LORA_RANK:-16}"; LORA_LR="${LORA_LR:-5e-5}"; HEAD_LR="${HEAD_LR:-5e-4}"; ANCHOR_W="${ANCHOR_W:-1.0}"
FULL_FT="${FULL_FT:-0}"
FT_LR="${FT_LR:-5e-5}"
if [ "$FULL_FT" = 1 ]; then METHOD="anchor-fullft-lr$FT_LR-w$ANCHOR_W-conf1"
else METHOD="anchor-lora-r$LORA_RANK-lr$LORA_LR-head$HEAD_LR-w$ANCHOR_W-conf1"; fi
# OUT=auto (convention from 2026-09-30): the path starts with the exact trunk run, then method+settings, target, run:
#   finetune/<trunks|trunks_24k>/<tag>_train/<backbone>-seed<k>/<method>/<target>/tf<N>-s<steps>-draw<d>
# trunk run dir = <results>/<area>/<tag>_train/supervised/sampling-T2/tf1/<backbone>/seed<k>
if [ "$OUT" = auto ]; then
    SEEDD=$(basename "$TRUNK"); BBD=$(basename "$(dirname "$TRUNK")")
    TAGD=$(echo "$TRUNK" | awk -F/ '{print $(NF-5)}'); AREA=$(echo "$TRUNK" | awk -F/ '{print $(NF-6)}')
    OUT="$RESULTS/finetune/$AREA/$TAGD/$BBD-$SEEDD/$METHOD/$DS/tf$N-s$STEPS-draw$DRAW"
    echo "output: $OUT"
fi
[ -f "$OUT/.done" ] && { echo "skip (done): $OUT"; exit 0; }
CKPT=$(ls "$TRUNK"/tb_logs/test/version_0/checkpoints/*-best.ckpt 2>/dev/null | head -1)
[ -f "$CKPT" ] || CKPT=$(ls "$TRUNK"/tb_logs/test/version_0/checkpoints/*.ckpt | head -1)
[ -f "$CKPT" ] || { echo "ABORT: no checkpoint in $TRUNK"; exit 1; }
read -r BACKBONE AKP < <(python - "$TRUNK/config.yaml" "$DATA/dataset_inventory.json" <<'PY'
import json, sys, yaml
from pathlib import Path
import pandas as pd
cfg = yaml.safe_load(open(sys.argv[1])); inv = json.load(open(sys.argv[2]))["datasets"]
csv = Path(sys.argv[1]).parent / cfg["data"]["csv_file"]
trained = sorted(set(pd.read_csv(csv, header=[0, 1, 2], index_col=0).index.str.split("/").str[1]))
names = sorted(set().union(*[set(inv[d]["trainable"]) for d in trained]))
print(cfg["model"]["backbone"], "[" + ",".join(f"'{n}'" for n in names) + "]")
PY
)
TF="$N"; [ "$N" = all ] && TF=1
HALF=$(( STEPS / 2 )); VAL_EVERY="${VAL_EVERY:-500}"
if [ "$FULL_FT" = 1 ]; then
    LR="$FT_LR"; ADAPT=""
else
    LR="$HEAD_LR"; ADAPT="+model.lora.rank=$LORA_RANK +model.lora.alpha=$((2 * LORA_RANK)) +model.lora.lr=$LORA_LR"
fi
mkdir -p "$(dirname "$OUT")"; LOG="$OUT.log"
if [ -d "$OUT" ]; then mv "$OUT" "$OUT.partial-$(date -u +%m%d%H%M)"; fi
echo "=== [$(date -u +%H:%M)] anchored $([ "$FULL_FT" = 1 ] && echo "full FT lr $LR" || echo "LoRA r$LORA_RANK lr $LORA_LR head $HEAD_LR"): $BACKBONE $DS N=$N steps=$STEPS draw=$DRAW anchor w $ANCHOR_W"
echo "    trunk ckpt $CKPT"
litpose train configs/model.yaml --output_dir "$OUT" --overrides \
    data.data_dir="$DATA" "data.csv_file=CollectedData_${DS}_train.csv" \
    model.backbone="$BACKBONE" "model.losses_to_use=[]" "+model.checkpoint='$CKPT'" \
    training.train_frames="$TF" training.rng_seed_data_pt="$DRAW" \
    training.optimizer_params.learning_rate="$LR" \
    training.min_steps="$STEPS" training.max_steps="$STEPS" training.unfreezing_step=1 \
    training.val_check_interval="$VAL_EVERY" "training.lr_scheduler_params.multisteplr.milestone_steps=[$HALF]" \
    +training.epoch_repeat=100 +training.num_workers=2 \
    +model.anchor.weight="$ANCHOR_W" +model.anchor.mode=unlabeled +model.anchor.conf_power=1 "+model.anchor.keypoints=$AKP" \
    $ADAPT \
    > "$LOG" 2>&1
echo "train exit $?"
grep -q "loading weights from"             "$LOG" || { echo "ABORT: trunk weights not loaded"; exit 1; }
grep -q "anchor: frozen teacher attached"  "$LOG" || { echo "ABORT: anchor teacher not attached"; exit 1; }
if [ "$FULL_FT" = 1 ]; then
    grep -q "LoRA: wrapped" "$LOG" && { echo "ABORT: LoRA applied in a full-FT run"; exit 1; }
else
    grep -q "LoRA: wrapped 72 linear layers (rank $LORA_RANK," "$LOG" || { echo "ABORT: LoRA not applied (rank $LORA_RANK)"; exit 1; }
fi
grep -q COMPLETED "$OUT/train_status.json"        || { echo "ABORT: training not COMPLETED"; exit 1; }
python -m mighty_mouse.train --output_dir "$OUT" --csv_file "CollectedData_${DS}_train.csv" --keep_checkpoints \
    > "${LOG%.log}-eval.log" 2>&1
E=$?
[ "$FULL_FT" = 1 ] || python - "$OUT" <<'PY' || { echo "ABORT: reloaded model lacks the trained LoRA adapters"; exit 1; }
import os, sys, torch
from lightning_pose.api.model import Model
from lightning_pose.models.backbones.lora import LoRALinear
d = sys.argv[1]; mdl = Model.from_dir(d); mdl._load(); m = mdl.model
loras = [l for l in m.backbone.modules() if isinstance(l, LoRALinear)]
from lightning_pose.utils.io import ckpt_path_from_base_path
ck = ckpt_path_from_base_path(d, model_name="test")  # the file the loader uses (prefers *-best)
sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
ok = len(loras) == 72 and all(torch.equal(l.lora_B.detach().cpu(), sd["backbone." + n + ".lora_B"])
                             for n, l in m.backbone.named_modules() if isinstance(l, LoRALinear))
print("LoRA reload check:", "OK" if ok else "FAILED", f"({len(loras)} layers, {ck})"); sys.exit(0 if ok else 1)
PY
echo "eval exit $E - $OUT"
# run_info.json: everything needed to identify this run later (trunk, method, settings, frames, code, result)
python scripts/write_run_info.py --out "$OUT" --trunk "$TRUNK" --ckpt "$CKPT" --dataset "$DS" --n_frames "$N" \
    --steps "$STEPS" --draw "$DRAW" --method "$METHOD" --full_ft "$FULL_FT" --lora_rank "$LORA_RANK" \
    --lora_lr "$LORA_LR" --head_lr "$HEAD_LR" --ft_lr "$FT_LR" --anchor_w "$ANCHOR_W" --val_every "$VAL_EVERY" \
    --backbone "$BACKBONE"
[ $E -eq 0 ] && [ -f "$OUT/eval/$DS/predictions.csv" ] && touch "$OUT/.done"
