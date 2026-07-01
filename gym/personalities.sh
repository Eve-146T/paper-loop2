#!/bin/bash
# Train three "personality" bots as reward-shaped fine-tunes from base/best (mirrors Paper Loop 1's
# retrain_new_rules.sh personalities), then rank everything head-to-head. GPU-bound -> sequential.
#   aggro:    low-territory killer   (cheap deaths, huge kill bonus, low hold)
#   greedy:   steady land-grabber    (big area weight, high hold, low kills)
#   survivor: cautious holder        (harsh death penalty, modest kills)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
COMMON="--B 256 --T 64 --width 32 --blocks 2 --scalar_dim 64 --trunk_dim 128 \
  --n_min 4 --n_max 12 --max_steps 2200 --eval_every 50 --ckpt_every 100 \
  --init_from checkpoints/base/best.pt --updates 300"

echo "[pers] aggro..."
$PY -u train.py --run aggro    --seed 1 --death_penalty -0.5 --kill_reward 6.0 --area_scale 100 --hold_weight 0.06 $COMMON > logs/aggro.log 2>&1
echo "[pers] greedy..."
$PY -u train.py --run greedy   --seed 2 --death_penalty -1.0 --kill_reward 1.0 --area_scale 140 --hold_weight 0.20 $COMMON > logs/greedy.log 2>&1
echo "[pers] survivor..."
$PY -u train.py --run survivor --seed 3 --death_penalty -2.5 --kill_reward 1.5 --area_scale 100 --hold_weight 0.12 $COMMON > logs/survivor.log 2>&1

echo "[pers] tournament over all five..."
$PY -u tournament.py checkpoints/base/best.pt checkpoints/league/best.pt \
  checkpoints/aggro/best.pt checkpoints/greedy/best.pt checkpoints/survivor/best.pt \
  --B 64 --max_steps 1500 > logs/tournament_all.log 2>&1 || true
echo "done" > logs/personalities_done.txt
echo "[pers] complete."
