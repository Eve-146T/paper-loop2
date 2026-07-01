# Paper Loop 2 RL gym

Train the strongest possible **Paper Loop 2** (free-movement) AI by **self-play PPO** (PyTorch + CUDA),
in a fast **headless, byte-faithful** clone of the game's rule kernel, then run it on-device via two
backends (onnxruntime + a pure-Kotlin fallback). Same recipe as Paper Loop 1's gym (`../../paper-io-clone/gym`),
adapted for **continuous movement**.

## What's different from Paper Loop 1's gym (grid-locked → free movement)

| | Paper Loop 1 gym | Paper Loop 2 gym |
|---|---|---|
| Movement | 4 cardinal grid steps | float pos+heading, steer toward target at capped `TURN_RATE` |
| Arena | square | **circle** (`ARENA_R`) — radial spawns, wall slide/die |
| Actions | 3 (straight/L/R, 90°) | **5** relative-turn buckets (hard/soft-L, straight, soft/hard-R) |
| Obs rotation | 90° increments | **arbitrary heading** (crop sampled + rotated forward=up) |
| Trail | single cells | thick **brush** (`stampDisc`); geometric **self-cross** w/ `SELF_GRACE` |
| Capture | grid BFS flood-fill | **same** grid BFS flood-fill (kept — it's what makes capture robust) |

One env step = one **decision** = `SUBSTEPS` (=4) fixed sub-steps holding the chosen target heading,
matching the on-device cadence (`World.DECIDE_SUBSTEPS`). The capture flood-fill, area accounting, and
collision priorities are mirrored 1:1 from `World.kt`.

## Setup
```bash
cd gym
uv sync                            # torch (cuda), numba, numpy, onnx, tensorboard
uv run python tests/test_env.py    # rule-kernel PARITY tests (must pass before trusting a run)
uv run python bench.py             # throughput + area-invariant + obs sanity
```
Needs an NVIDIA GPU.

## Pipeline
```bash
# 1. base self-play (resumes from checkpoints/base/latest.pt; TensorBoard in runs/)
uv run python train.py --run base --updates 1200 \
  --width 32 --blocks 2 --scalar_dim 64 --trunk_dim 128 --hold_weight 0.10

# 2. league champion (robust vs a pool of past selves) + ranking + export — all unattended:
bash run_pipeline.sh               # waits for base, trains league, tournament, exports champion

# or manually:
uv run python tournament.py 'checkpoints/*/best.pt'          # crown by head-to-head + vs-scripted
uv run python export.py --ckpt checkpoints/league/best.pt    # -> ../app/src/main/assets/{bot.onnx,bot.plb}
```
`tensorboard --logdir runs` to watch entropy / KL / value-loss / reward / win-vs-scripted.

## Layout
- `paper_loop/env.py` — batched Numba sim (free-movement rule kernel) + rotated egocentric obs.
- `paper_loop/rules.py` — constants mirrored 1:1 from `World.kt`/`Player.kt`/`Geom.kt`.
- `paper_loop/scripted.py` — Numba port of `Bot.kt` (benchmark opponent).
- `paper_loop/model.py` — actor-critic ResNet (0.53M default = the shipped on-device size).
- `paper_loop/match.py` — eval vs scripted, head-to-head, replay snapshots.
- `train.py` / `tournament.py` / `export.py` / `run_pipeline.sh`.
- `tests/test_env.py` — rule-kernel parity (mirrors `app/.../SimTest.kt`).

## The one lesson that matters
**Pure Δarea reward makes agents turtle** (park on the spawn disc, never venture). The fix is a dense
**`hold_weight` term ∝ territory held every step** — a living agent always nets positive, so there's no
suicide incentive, but sitting still is strictly dominated by expanding. Everything else is secondary.

## On-device (see `../app/.../BotNet.kt`)
`export.py` writes the champion as a single-file `bot.onnx` (onnxruntime-android, fast primary) **and**
`bot.plb` (PL21 named-float32 blob, pure-Kotlin forward — F-Droid-clean fallback + cross-check). The
two backends are verified identical on-device (argmax agreement + max-logit-diff ~1e-7). `BotNet.encode`
is a 1:1 port of `env.encode_all`, so the net sees exactly what it trained on — **keep them in lockstep.**
