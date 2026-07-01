#!/bin/bash
# Unattended training pipeline: wait for the base self-play run to finish, then train a
# robustness-focused LEAGUE champion (self-play vs a pool of past snapshots), rank base-vs-league
# head-to-head, and export the stronger one to the app assets (bot.onnx + bot.plb).
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
COMMON="--B 256 --T 64 --width 32 --blocks 2 --scalar_dim 64 --trunk_dim 128 \
  --n_min 4 --n_max 12 --max_steps 2200 --hold_weight 0.10 --kill_reward 2.0 \
  --death_penalty -1.0 --area_scale 100 --win_bonus 10 --place_weight 5 --eval_every 50"

echo "[pipeline] waiting for base to finish..."
while ! grep -q "^done\." logs/base.log 2>/dev/null; do sleep 30; done
echo "[pipeline] base done."

echo "[pipeline] training league champion (init from base/best)..."
$PY -u train.py --run league --init_from checkpoints/base/best.pt --updates 500 \
  --league_frac 0.5 --snapshot_every 100 --ckpt_every 100 --seed 1 $COMMON > logs/league.log 2>&1
echo "[pipeline] league done."

echo "[pipeline] tournament base vs league..."
$PY -u tournament.py checkpoints/base/best.pt checkpoints/league/best.pt \
  --B 64 --max_steps 1500 > logs/tournament.log 2>&1 || true

# Pick the champion: the tournament's last 'CHAMPION:' line; default league/best.
CH=$(grep "^CHAMPION:" logs/tournament.log | tail -1 | sed 's/CHAMPION: //')
CKPT="checkpoints/league/best.pt"
case "$CH" in
  base/*)   CKPT="checkpoints/base/best.pt";;
  league/*) CKPT="checkpoints/league/best.pt";;
esac
echo "[pipeline] champion = $CH  -> exporting $CKPT"
$PY export.py --ckpt "$CKPT" > logs/export.log 2>&1
echo "champion=$CH ckpt=$CKPT" > logs/pipeline_done.txt
echo "[pipeline] complete."
