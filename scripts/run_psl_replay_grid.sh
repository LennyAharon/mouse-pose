#!/bin/bash
# SuperAnimal-style pseudo-label replay, run over the nine masked settings at N=10 draw 0.
# Per cell: teacher zero-shot inference on the cell's frames -> pseudo-label CSV (threshold
# 0.6, scripts/build_psl_replay_csv.py) -> full fine-tuning under the trunk5 masked protocol
# -> eval. Idempotent: a cell with an eval pixel_error.csv is skipped, so the queue can be
# relaunched after any interruption.
#   setsid nohup bash scripts/run_psl_replay_grid.sh > /tmp/psl_grid.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
THR=0.6
DATA=$(python -c "from mouse_pose.paths import load_paths; print(load_paths()['data_dir'])")
RESULTS=$(python -c "from mouse_pose.paths import load_paths; print(load_paths()['results_dir'])")
declare -A LOO=(
  [ibl]="face+cheese+caz+kondo"       [kondo]="face+ibl+cheese+caz"
  [cheese-2d]="face+ibl+caz+kondo"    [cazettes-side]="face+ibl+cheese+kondo"
  [facemap]="ibl+cheese+caz+kondo"
)
SETTINGS=(
  "ibl:pupil_center_left"
  "ibl:wrist_left+wrist_right"
  "cazettes-side:pupil_center_left"
  "cazettes-side:wrist_left+wrist_right"
  "kondo:wrist_left+wrist_right"
  "cheese-2d:eye_back_left+eye_back_right"
  "facemap:nose_tip"
  "kondo:lowerlip"
  "cazettes-side:tongue_tip"
)
for S in "${SETTINGS[@]}"; do
  DS="${S%%:*}"; SLUG="${S#*:}"
  TRUNK="$RESULTS/zoom-aug-exp/${LOO[$DS]}-T2-zoominout/seed0"
  OUT="$RESULTS/fewshot-exp-psl-replay-thr$THR-zio-mask-$SLUG/$DS/tf10-draw0"
  if [ -f "$OUT/eval/$DS/pixel_error.csv" ]; then echo "skip (done): $DS $SLUG"; continue; fi
  echo "=== [$(date -u +%H:%M)] psl-replay $DS $SLUG ==="
  # wait for >= 6 GB free GPU memory, as fewshot_cell.sh does
  while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6000 ]; do sleep 60; done

  FRAMES_CSV="CollectedData_${DS}_psl10d0_${SLUG}_frames.csv"
  python - "$DS" "$SLUG" "$FRAMES_CSV" <<'PY' || { echo "ABORT: frame selection $DS"; exit 1; }
import sys
sys.path.insert(0, "scripts")
from pathlib import Path
import pandas as pd
from build_replay_csv import selected_frames
from mouse_pose.paths import load_paths
ds, slug, out = sys.argv[1], sys.argv[2], sys.argv[3]
data = Path(load_paths()["data_dir"])
frames = selected_frames(data, ds, 10, 0)
df = pd.read_csv(data / f"CollectedData_{ds}_train_mask-{slug}.csv", header=[0, 1, 2], index_col=0)
df.loc[frames].to_csv(data / out)
print(f"{out}: {len(frames)} frames")
PY

  litpose predict "$TRUNK" "$DATA/$FRAMES_CSV" > /dev/null 2>&1
  PREDS="$TRUNK/image_preds/$FRAMES_CSV/predictions.csv"
  [ -f "$PREDS" ] || { echo "ABORT: teacher predictions missing for $DS $SLUG"; exit 1; }

  python scripts/build_psl_replay_csv.py --dataset "$DS" \
      --train_csv "CollectedData_${DS}_train_mask-${SLUG}.csv" \
      --preds "$PREDS" --threshold "$THR" || { echo "ABORT: psl csv $DS $SLUG"; exit 1; }
  PSL_CSV="CollectedData_${DS}_train_psl_mask-${SLUG}-thr${THR}.csv"

  CKPT=$(ls "$TRUNK/tb_logs/test/version_0/checkpoints/"*.ckpt | head -1)
  mkdir -p "$(dirname "$OUT")"
  litpose train configs/model.yaml --output_dir "$OUT" --overrides \
      data.data_dir="$DATA" "data.csv_file=$PSL_CSV" \
      model.backbone=vits_dinov3 "model.losses_to_use=[]" \
      "+model.checkpoint='$CKPT'" \
      training.train_frames=10 training.rng_seed_data_pt=0 \
      training.optimizer_params.learning_rate=5e-05 \
      training.min_steps=2000 training.max_steps=2000 training.unfreezing_step=1 \
      training.val_check_interval=1000 \
      "training.lr_scheduler_params.multisteplr.milestone_steps=[1000]" \
      +training.epoch_repeat=100 +training.num_workers=2 \
      > "$OUT.log" 2>&1
  grep -q COMPLETED "$OUT/train_status.json" 2>/dev/null || { echo "ABORT: training failed $DS $SLUG"; exit 1; }
  grep -q "loading weights from" "$OUT.log" || { echo "ABORT: trunk weights not loaded $DS $SLUG"; exit 1; }

  python -m mouse_pose.train --output_dir "$OUT" --csv_file "CollectedData_${DS}_train.csv" \
      > "${OUT}-eval.log" 2>&1
  [ -f "$OUT/eval/$DS/pixel_error.csv" ] || { echo "ABORT: eval failed $DS $SLUG"; exit 1; }
  echo "done: $DS $SLUG"
done
echo "=== grid complete ==="
