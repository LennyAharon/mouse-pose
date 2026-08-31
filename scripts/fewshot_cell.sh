#!/bin/bash
# Few-shot grid cell. Usage: fewshot_cell.sh <arm: trunk|dino> <dataset> <n_frames> <draw_seed>
#   trunk : init = zoomaug leave-<DS>-out super-mouse trunk, lr 1e-5  -> fewshot-exp/
#   dino  : init = DINOv3 backbone + random head (LP default), lr 5e-5 -> fewshot-exp-dino/
#   trunk5: init = same trunk as `trunk`, lr 5e-5 (identical protocol to dino) -> fewshot-exp-lr5/
#   trunk5-hf: trunk5 + head filters of the supported keypoints frozen (lightning-pose-dev,
#             branch fewshot_head, model.head_freeze_keypoints) -> fewshot-exp-headfreeze/
#   trunk5-bf: trunk5-hf + backbone frozen (unfreezing_step=1e6) -> fewshot-exp-backfreeze/
#   lora     : trunk + LoRA on the backbone (base frozen), head trainable -> fewshot-exp-lora-r<R>[-lr<L>]/
#   anchor-video: anchor-lora + the trunk distilled on unlabeled video (env STEPS, ANCHOR_VIDEO_DIR) -> fewshot-exp-anchor-lora-video[-conf<C>][-s<STEPS>]/
#   anchor / anchor-lora: trunk-distilled (anchored) fine-tuning, full FT / LoRA form (env ANCHOR_W, ANCHOR_CONF) -> fewshot-exp-anchor[-lora][-w<W>][-conf<C>]/
#   xfer     : full FT (trunk5 protocol) from any checkpoint (env SRC_CKPT, SRC_TAG) -> fewshot-exp-xfer-<tag>/
#   dino-lora: DINOv3 + LoRA, random head -> fewshot-exp-dino-lora-r<R>[-lr<L>]/   (env LORA_RANK, LORA_LR)
#   env AUG=zoom: train with the trunk's zoom-aug recipe (any arm) -> <root>-zoomaug/
#   env TRUNK_SUFFIX=T2-zoominout: start from the zoom-in/out trunks -> <root>-zio/
#   env MASK_KPS="kp1,kp2": hide those labels in training (any arm) -> <root>-mask-<kp1+kp2>/
# Shared honest protocol: N frames drawn by rng_seed_data_pt=<draw> (identical frames in both
# arms), 2000 steps, backbone unfrozen from step 1, lr halves at 1000, validation only at
# 1000/2000 (<=2 selection points), stock dlc aug, no temperature sampling, no registry.
# epoch_repeat=100 packs 100 shuffled passes per loader epoch (stock batches, only Lightning's
# per-epoch turnover amortized); num_workers=2 keeps 4 concurrent jobs off each other's CPUs.
set -u
ARM="$1"; DS="$2"; N="$3"; DRAW="$4"
declare -A LOO=(
  [ibl]="face+cheese+caz+kondo"       [kondo]="face+ibl+cheese+caz"
  [cheese-2d]="face+ibl+caz+kondo"    [cazettes-side]="face+ibl+cheese+kondo"
  [facemap]="ibl+cheese+caz+kondo"
)
DATA=$(python -c "from mouse_pose.paths import load_paths; print(load_paths()['data_dir'])")
RESULTS=$(python -c "from mouse_pose.paths import load_paths; print(load_paths()['results_dir'])")
case "$ARM" in
  trunk) ROOT="$RESULTS/fewshot-exp";      LR="1e-05"; LRPAT="learning_rate: 1.0e-05"
         CKPT=$(ls "$RESULTS/zoom-aug-exp/${LOO[$DS]}-${TRUNK_SUFFIX:-T2-zoomaug}/seed0/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
         CKPT_OVR="+model.checkpoint='$CKPT'" ;;
  dino)  ROOT="$RESULTS/fewshot-exp-dino"; LR="5e-05"; LRPAT="learning_rate: 5.0e-05"
         CKPT_OVR="" ;;
  trunk5) ROOT="$RESULTS/fewshot-exp-lr5"; LR="5e-05"; LRPAT="learning_rate: 5.0e-05"
         CKPT=$(ls "$RESULTS/zoom-aug-exp/${LOO[$DS]}-${TRUNK_SUFFIX:-T2-zoomaug}/seed0/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
         CKPT_OVR="+model.checkpoint='$CKPT'" ;;
  anchor|anchor-lora|anchor-video)
         # Anchored fine-tuning: the frozen trunk distills its heatmaps into every keypoint it
         # trained that the target frame does not label (model.anchor, lightning-pose-dev).
         # anchor: full FT at 5e-5; anchor-lora: LoRA r16 adapters 5e-5 + head 5e-4. env ANCHOR_W (1.0).
         AW="${ANCHOR_W:-1.0}"; AC="${ANCHOR_CONF:-0}"; SUF=""; [ "$AW" != "1.0" ] && SUF="-w$AW"; [ "$AC" != "0" ] && SUF="$SUF-conf$AC"
         if [ "$ARM" = anchor ]; then ROOT="$RESULTS/fewshot-exp-anchor$SUF"; LR="5e-05"; else ROOT="$RESULTS/fewshot-exp-anchor-lora$SUF"; LR="${HEAD_LR:-5e-4}"; fi
         if [ "$ARM" = anchor-video ]; then
             # + the trunk distilled on the lab's unlabeled video (semi-supervised path)
             declare -A VID=( [facemap]="/teamspace/studios/this_studio/poseinterface/_raw/facemap/videos_test" [cheese-2d]="/teamspace/studios/this_studio/poseinterface/_raw/cheese-2d/videos" )
             VDIR="${ANCHOR_VIDEO_DIR:-${VID[$DS]:-}}"; [ -d "$VDIR" ] || { echo "ABORT: no unlabeled video dir for $DS"; exit 1; }
             ROOT="$RESULTS/fewshot-exp-anchor-lora-video$SUF"; [ "${STEPS:-2000}" != 2000 ] && ROOT="$ROOT-s${STEPS}"
         fi
         LRPAT="learning_rate: $(python -c "print(f'{float(\"$LR\"):.1e}')")"
         CKPT=$(ls "$RESULTS/zoom-aug-exp/${LOO[$DS]}-${TRUNK_SUFFIX:-T2-zoomaug}/seed0/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
         CKPT_OVR="+model.checkpoint='$CKPT'"
         AKP=$(python - "$DS" <<'PY'
import json, sys
from mouse_pose.paths import load_paths
inv = json.load(open(load_paths()["data_dir"] + "/dataset_inventory.json"))["datasets"]
names = sorted(set().union(*[set(inv[o]["trainable"]) for o in inv if o != sys.argv[1]]))
print("[" + ",".join(f"'{n}'" for n in names) + "]")
PY
)
         EXTRA_OVR="+model.anchor.weight=$AW +model.anchor.mode=unlabeled +model.anchor.conf_power=$AC +model.anchor.keypoints=$AKP"
         [ "$ARM" != anchor ] && EXTRA_OVR="$EXTRA_OVR +model.lora.rank=${LORA_RANK:-16} +model.lora.alpha=$((2 * ${LORA_RANK:-16})) +model.lora.lr=${LORA_LR:-5e-5}"
         [ "$ARM" = anchor-video ] && EXTRA_OVR="$EXTRA_OVR data.video_dir=$VDIR model.losses_to_use=[anchor_video] callbacks.anneal_weight.init_val=1.0 callbacks.anneal_weight.freeze_until_epoch=0 dali.base.train.sequence_length=16"
         export PYTHONPATH=/teamspace/studios/this_studio/lightning-pose-dev ;;
  xfer)  # full FT (trunk5 protocol) from an arbitrary source checkpoint: env SRC_CKPT (path), SRC_TAG (name)
         ROOT="$RESULTS/fewshot-exp-xfer-${SRC_TAG:?SRC_TAG required}"; LR="5e-05"; LRPAT="learning_rate: 5.0e-05"
         CKPT="${SRC_CKPT:?SRC_CKPT required}"; [ -f "$CKPT" ] || { echo "ABORT: SRC_CKPT not found: $CKPT"; exit 1; }
         CKPT_OVR="+model.checkpoint='$CKPT'" ;;
  trunk5-hf) ROOT="$RESULTS/fewshot-exp-headfreeze"; LR="5e-05"; LRPAT="learning_rate: 5.0e-05"
         CKPT=$(ls "$RESULTS/zoom-aug-exp/${LOO[$DS]}-${TRUNK_SUFFIX:-T2-zoomaug}/seed0/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
         CKPT_OVR="+model.checkpoint='$CKPT'"
         # freeze the head filters the trunk already trained (= the supported set); the rest learn
         KEEP=$(python - "$DS" <<'PY'
import json, sys
from mouse_pose.paths import load_paths
inv = json.load(open(load_paths()["data_dir"] + "/dataset_inventory.json"))["datasets"]
ds = sys.argv[1]
others = set().union(*[set(inv[o]["trainable"]) for o in inv if o != ds])
print(",".join(k for k in inv[ds]["eval"] if k in others and k != "pupil_center_right"))
PY
)
         EXTRA_OVR="+model.head_freeze_keypoints=[$KEEP]"
         export PYTHONPATH=/teamspace/studios/this_studio/lightning-pose-dev ;;   # branch fewshot_head
  trunk5-bf) ROOT="$RESULTS/fewshot-exp-backfreeze"; LR="5e-05"; LRPAT="learning_rate: 5.0e-05"
         CKPT=$(ls "$RESULTS/zoom-aug-exp/${LOO[$DS]}-${TRUNK_SUFFIX:-T2-zoomaug}/seed0/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
         CKPT_OVR="+model.checkpoint='$CKPT'"
         # backbone never unfreezes (lr stays 0) + supported head filters frozen: supported
         # keypoints are exactly zero-shot by construction; untrained filters learn on frozen features
         KEEP=$(python - "$DS" <<'PY'
import json, sys
from mouse_pose.paths import load_paths
inv = json.load(open(load_paths()["data_dir"] + "/dataset_inventory.json"))["datasets"]
ds = sys.argv[1]
others = set().union(*[set(inv[o]["trainable"]) for o in inv if o != ds])
print(",".join(k for k in inv[ds]["eval"] if k in others and k != "pupil_center_right"))
PY
)
         EXTRA_OVR="+model.head_freeze_keypoints=[$KEEP] training.unfreezing_step=1000000"
         export PYTHONPATH=/teamspace/studios/this_studio/lightning-pose-dev ;;
  lora|dino-lora)
         # LoRA on the backbone (base frozen), head fully trainable. Env: LORA_RANK (16),
         # LORA_LR (global lr). Root name carries rank (+ lr when overridden).
         RANK="${LORA_RANK:-16}"; LLR="${LORA_LR:-}"; HLR="${HEAD_LR:-}"
         SUFFIX="r$RANK"; [ -n "$LLR" ] && SUFFIX="$SUFFIX-lr$LLR"; [ -n "$HLR" ] && SUFFIX="$SUFFIX-head$HLR"
         # HEAD_LR overrides the global lr (which the head group uses); adapters keep LORA_LR
         LR="${HLR:-5e-05}"; LRPAT="learning_rate: $(python -c "print(f'{float(\"$LR\"):.1e}')")"
         if [ "$ARM" = lora ]; then
             ROOT="$RESULTS/fewshot-exp-lora-$SUFFIX"
             CKPT=$(ls "$RESULTS/zoom-aug-exp/${LOO[$DS]}-${TRUNK_SUFFIX:-T2-zoomaug}/seed0/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
             CKPT_OVR="+model.checkpoint='$CKPT'"
         else
             ROOT="$RESULTS/fewshot-exp-dino-lora-$SUFFIX"; CKPT_OVR=""
         fi
         EXTRA_OVR="+model.lora.rank=$RANK +model.lora.alpha=$((2 * RANK))"
         [ -n "$LLR" ] && EXTRA_OVR="$EXTRA_OVR +model.lora.lr=$LLR"
         export PYTHONPATH=/teamspace/studios/this_studio/lightning-pose-dev ;;
  replay|replay-lora)
         # Replay fine-tuning: the N target frames (same split as the plain cell) mixed with the
         # n-1 corpus, equal supervision per dataset (T=inf), per-dataset zoom aug, 4000 steps.
         # replay-lora: backbone frozen + LoRA adapters (5e-4), head at 5e-4.
         REPLAY=1; RANK="${LORA_RANK:-16}"
         if [ "$ARM" = replay ]; then ROOT="$RESULTS/fewshot-exp-replay"; LR="5e-05"; else ROOT="$RESULTS/fewshot-exp-replay-lora"; LR="${HEAD_LR:-5e-4}"; fi
         LRPAT="learning_rate: $(python -c "print(f'{float(\"$LR\"):.1e}')")"
         CKPT=$(ls "$RESULTS/zoom-aug-exp/${LOO[$DS]}-${TRUNK_SUFFIX:-T2-zoomaug}/seed0/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
         CKPT_OVR="+model.checkpoint='$CKPT'"
         EXTRA_OVR="training.sampling_temperature='inf' data.dataset_names=['facemap','ibl','cheese-2d','cazettes-side','kondo']"
         [ "$ARM" = replay-lora ] && EXTRA_OVR="$EXTRA_OVR +model.lora.rank=$RANK +model.lora.alpha=$((2 * RANK)) +model.lora.lr=${LORA_LR:-5e-4}"
         export PYTHONPATH=/teamspace/studios/this_studio/lightning-pose-dev ;;
  *) echo "unknown arm $ARM"; exit 2 ;;
esac
REPLAY="${REPLAY:-0}"
if [ "$REPLAY" = 1 ]; then
    # build the merged CSV once per (dataset, N, draw); training then uses it with the zoomaug config
    REPLAY_CSV="CollectedData_replay-${DS}-tf${N}-draw${DRAW}_train.csv"
    [ -f "$DATA/$REPLAY_CSV" ] || python scripts/build_replay_csv.py --dataset "$DS" --n_frames "$N" --draw "$DRAW" > /dev/null 2>&1
    [ -f "$DATA/$REPLAY_CSV" ] || { echo "ABORT: replay CSV not built"; exit 1; }
fi
EXTRA_OVR="${EXTRA_OVR:-}"
# env TRUNK_SUFFIX (default T2-zoomaug): which zoom-aug-exp trunk variant to start from, e.g.
# T2-zoominout -> results root gets the suffix -zio.
case "${TRUNK_SUFFIX:-T2-zoomaug}" in T2-zoomaug) ;; T2-zoominout) ROOT="$ROOT-zio" ;; *) ROOT="$ROOT-$(echo ${TRUNK_SUFFIX} | tr '+' '_')" ;; esac
# Masked-label transfer protocol: env MASK_KPS="kp1,kp2" hides those labels in the TRAIN csv
# (scripts/build_masked_csv.py); test csvs are untouched so the hidden keypoints can be scored.
TRAIN_CSV="CollectedData_${DS}_train.csv"
if [ -n "${MASK_KPS:-}" ]; then
    MSLUG=$(echo "$MASK_KPS" | tr ',' '+')
    TRAIN_CSV="CollectedData_${DS}_train_mask-${MSLUG}.csv"
    [ -f "$DATA/$TRAIN_CSV" ] || python scripts/build_masked_csv.py --dataset "$DS" --mask "$MASK_KPS"
    ROOT="$ROOT-mask-$MSLUG"
fi
# env AUG=zoom: fine-tune with the trunk's own augmentation recipe (model_zoomaug.yaml: rotate/blur/
# dropout/elastic/CropAndPad + this dataset's per-dataset zoom-out bound) instead of stock dlc.
FT_CFG="configs/model.yaml"; AUG_OVR=""
if [ "${AUG:-}" = zoom ]; then
    FT_CFG="configs/model_zoomaug.yaml"
    # all five names so the config's per-dataset zoom map validates (only $DS frames exist in the csv;
    # sampling_temperature is null in this config, so no sampler is involved)
    AUG_OVR="data.dataset_names=['facemap','ibl','cheese-2d','cazettes-side','kondo']"
    ROOT="$ROOT-zoomaug"
fi
OUT="$ROOT/$DS/tf$N-draw$DRAW"
LOG="$ROOT/$DS-tf$N-draw$DRAW.log"
mkdir -p "$ROOT"
[ -f "$OUT/.done" ] && { echo "skip (done): $ARM $DS tf$N draw$DRAW"; exit 0; }

# never start a cell (or its eval) with < 6 GB free on the GPU — the GPU is shared with other queues
gpu_gate() { while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6000 ]; do sleep 60; done; }
# GPU slot semaphore: at most MAX_CELLS cells (train or eval) on the GPU at once. Slots are
# mkdir-claimed directories holding the owner pid; dead owners are reclaimed. Released on exit.
SLOTS=/teamspace/studios/this_studio/mouse-pose/scripts/fewshot_queue/gpu_slots; mkdir -p "$SLOTS"; MAX_CELLS="${MAX_CELLS:-$(cat "$SLOTS/max" 2>/dev/null || echo 3)}"
acquire_slot() {
    while true; do
        MAX_CELLS="$(cat "$SLOTS/max" 2>/dev/null || echo 3)"
        for i in $(seq 1 "$MAX_CELLS"); do
            d="$SLOTS/$i"
            if [ -d "$d" ] && [ -f "$d/pid" ] && ! kill -0 "$(cat "$d/pid")" 2>/dev/null; then rm -rf "$d"; fi
            if mkdir "$d" 2>/dev/null; then echo $$ > "$d/pid"; SLOT="$d"; return; fi
        done
        sleep 30
    done
}
acquire_slot; trap 'rm -rf "$SLOT"' EXIT
gpu_gate
echo "=== [$(date -u +%H:%M)] $ARM $DS N=$N draw=$DRAW ==="
cd "$(dirname "$0")/.."
if [ -d "$OUT" ] && grep -q COMPLETED "$OUT/train_status.json" 2>/dev/null \
   && ls "$OUT"/tb_logs/test/version_0/checkpoints/*.ckpt >/dev/null 2>&1; then
    echo "training already complete — eval only"
else
    if [ -d "$OUT" ]; then
        mkdir -p "$RESULTS/_killed-partials"
        mv "$OUT" "$RESULTS/_killed-partials/$ARM-$DS-tf$N-draw$DRAW-$(date -u +%m%d%H%M)"
    fi
    if [ "$REPLAY" = 1 ]; then
    litpose train "${REPLAY_CFG:-configs/model_zoomaug.yaml}" --output_dir "$OUT" --overrides \
        data.data_dir="$DATA" \
        "data.csv_file=$REPLAY_CSV" \
        model.backbone=vits_dinov3 "model.losses_to_use=[]" \
        $CKPT_OVR \
        training.train_frames=1 training.rng_seed_data_pt="$DRAW" \
        training.optimizer_params.learning_rate="$LR" \
        training.min_steps=4000 training.max_steps=4000 training.unfreezing_step=1 \
        training.val_check_interval=2000 \
        "training.lr_scheduler_params.multisteplr.milestone_steps=[2000]" \
        +training.num_workers=2 \
        $EXTRA_OVR \
        > "$LOG" 2>&1
    else
    litpose train $FT_CFG --output_dir "$OUT" --overrides \
        data.data_dir="$DATA" \
        "data.csv_file=$TRAIN_CSV" $AUG_OVR \
        model.backbone=vits_dinov3 "model.losses_to_use=[]" \
        $CKPT_OVR \
        training.train_frames="$N" training.rng_seed_data_pt="$DRAW" \
        training.optimizer_params.learning_rate="$LR" \
        training.min_steps=${STEPS:-2000} training.max_steps=${STEPS:-2000} training.unfreezing_step=1 \
        training.val_check_interval=$(( ${STEPS:-2000} / 2 )) \
        "training.lr_scheduler_params.multisteplr.milestone_steps=[$(( ${STEPS:-2000} / 2 ))]" \
        +training.epoch_repeat=100 +training.num_workers=2 \
        $EXTRA_OVR \
        > "$LOG" 2>&1
    fi
    echo "train exit $?"
    python - "$OUT/config.yaml" "$LR" <<'PY' || { echo "ABORT: lr override missing"; exit 1; }
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); got = float(cfg["training"]["optimizer_params"]["learning_rate"]); want = float(sys.argv[2])
print(f"lr check: config {got:g} vs expected {want:g}"); sys.exit(0 if abs(got - want) <= 1e-12 else 1)
PY
    if [ "$ARM" != dino ] && [ "$ARM" != dino-lora ]; then
        grep -q "loading weights from" "$LOG" || { echo "ABORT: trunk weights not loaded"; exit 1; }
        if [ "$ARM" = trunk5-hf ] || [ "$ARM" = trunk5-bf ]; then
            grep -q "head filter freeze:" "$LOG" || { echo "ABORT: head freeze not installed"; exit 1; }
        fi
        [ "$REPLAY" = 1 ] && { grep -q "TemperatureSampler" "$LOG" || { echo "ABORT: replay without sampler"; exit 1; }; }
        case "$ARM" in anchor|anchor-lora|anchor-video)
            grep -q "anchor: frozen teacher attached" "$LOG" || { echo "ABORT: anchor teacher not attached"; exit 1; } ;;
        esac
        case "$ARM" in lora|dino-lora|replay-lora|anchor-lora|anchor-video)
            grep -q "LoRA: wrapped 72 linear layers" "$LOG" || { echo "ABORT: LoRA not applied"; exit 1; } ;;
        esac
        if [ "$ARM" = trunk5-bf ]; then
            grep -q "unfreezing_step: 1000000" "$OUT/config.yaml" || { echo "ABORT: backbone not frozen (unfreezing_step override lost)"; exit 1; }
        fi
    else
        grep -Eq "^\s+checkpoint: " "$OUT/config.yaml" && { echo "ABORT: dino arm has a checkpoint"; exit 1; }
    fi
    grep -q COMPLETED "$OUT/train_status.json" || { echo "ABORT: training not COMPLETED"; exit 1; }
fi
gpu_gate
KEEP_CKPT=""; case "$ARM" in lora|dino-lora|replay|replay-lora|anchor|anchor-lora|anchor-video) KEEP_CKPT="--keep_checkpoints" ;; esac   # adapters: keep the weights
python -m mouse_pose.train --output_dir "$OUT" --csv_file "CollectedData_${DS}_train.csv" $KEEP_CKPT \
    > "${LOG%.log}-eval.log" 2>&1
E=$?
case "$ARM" in lora|dino-lora|replay-lora|anchor-lora|anchor-video)
    # the reload must carry the adapters: verify on the checkpoint + reloaded model directly
    # (Lightning's first-pass "keys not in the model state dict" warning is expected and harmless)
    python - "$OUT" <<'PY' || { echo "ABORT: reloaded model lacks the trained LoRA adapters"; exit 1; }
import sys, os, torch
from lightning_pose.api.model import Model
from lightning_pose.models.backbones.lora import LoRALinear
d = sys.argv[1]; mdl = Model.from_dir(d); mdl._load(); m = mdl.model
loras = [l for l in m.backbone.modules() if isinstance(l, LoRALinear)]
ck = os.path.join(d, "tb_logs/test/version_0/checkpoints", os.listdir(os.path.join(d, "tb_logs/test/version_0/checkpoints"))[0])
sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
ok = len(loras) == 72 and all(torch.equal(l.lora_B.detach().cpu(), sd["backbone." + n + ".lora_B"]) for n, l in m.backbone.named_modules() if isinstance(l, LoRALinear))
print("LoRA reload check:", "OK" if ok else "FAILED", f"({len(loras)} layers)"); sys.exit(0 if ok else 1)
PY
    ;;
esac
echo "eval exit $E — $ARM $DS/tf$N-draw$DRAW DONE"
[ $E -eq 0 ] && [ "$(ls "$OUT/eval" | wc -l)" -eq 5 ] && touch "$OUT/.done"
