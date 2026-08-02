# CLAUDE.md

Guidance for coding agents working in this repository.

## What this is

Paper Loop 2 (package `paper.loop2`) is a single-activity Android territory-capture game built with
libGDX and Kotlin. Movement is free and continuous, not grid-locked: the head has a float position
and heading and steers toward a target angle at a capped turn rate. The map is a circle, the trail is
a smooth ribbon, territories render as smooth rounded blobs.

Opponents are neural bots — a small self-play PPO net (`BotNet`) trained in the headless gym
(`gym/`) and run on-device through `onnxruntime-android`, with the scripted `Bot` as the automatic
fallback if the net fails to load. No networking, no multiplayer, `VIBRATE` is the only permission.
GPL-v3, F-Droid.

Everything is drawn inside libGDX — every screen, the HUD, the game-over card. There is no
native-view HUD and no cross-thread UI marshalling.

Mechanism lives in the code comments; this file is the map and the rules that are not local to any
one file.

## Build, run, test

JDK 17 (21 also builds), Android SDK platform 35, build-tools 35.0.0. SDK path comes from
`local.properties` (`sdk.dir=…`) or `$ANDROID_HOME`.

```bash
./gradlew :app:assembleDebug      # -> app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:testDebugUnitTest  # headless JVM unit tests
./gradlew :app:installDebug
./gradlew :app:lint               # ObsoleteSdkInt disabled (adaptive-icon false positive)
```

On a device or emulator:

```bash
adb shell am start -n paper.loop2/paper.loop2.GameActivity
adb exec-out screencap -p > shot.png   # then read the PNG to inspect
# drive it: `input tap 540 700` (tap to begin), `input swipe x1 y1 x2 y2 ms` (steer = joystick angle)
```

`app/libs/` is gitignored: `copyAndroidNatives` re-creates it each build from the Maven-pulled libGDX
natives.

## Tests

`app/src/test/kotlin/paper/loop2/game/SimTest.kt` drives `World` headless: capture grows territory,
the incremental `area` matches the grid, the wall and self-crossing kill, a 40s 8-player bot run
stays stable and productive, score % is of the arena, bot respawn telegraphs and never lands on turf
or near a head. Run `:app:testDebugUnitTest` after any change to `World`/`Player`/`Bot`/`Geom`.

## Architecture

Package root `paper.loop2` under `app/src/main/kotlin/`. `App` inits the Android-context singletons
(`Scores`, `Settings`, `Haptics`); `GameActivity` runs `PaperLoopGame`; `SoundFx.init()` runs after
`create()` because it needs `Gdx.audio`. Only the 1px white texture is built on the GL thread at
startup — fonts rasterise on one background thread and the sound bank synthesises on another, so the
world draws from frame 1.

`game/` is the game:

- `World` — the simulation. Two flat `GW*GH` arrays, `owner` (the source of truth for territory) and
  `trail`, plus a player list. `update(dt)` accumulates into fixed sub-steps; each sub-step decides,
  moves, lays trail, resolves collisions, then captures. Free of libGDX and Android imports.
- `Player` — one player's continuous head, trail polyline, stats, and the bboxes the renderer and
  capture use to work on a sub-rect instead of the whole grid. `Geom` — angle helpers, a primitive
  `FloatArrayList`, the point-segment distance used for self-cross.
- `Bot` — the scripted AI: out leg, side leg, beeline home, with a reactive heading filter. Used as
  the fallback brain and as the gym's benchmark opponent.
- `BotNet` — the neural brain and the libGDX/Android boundary. Inference runs off the sim thread from
  a snapshot; `encodeAt` is a 1:1 port of the gym's `encode_all`.
- `GameScreen` — the only screen: a READY/PLAYING/OVER phase machine, the joystick, the start menu,
  and the wiring of `World`'s seams to `SoundFx`/`Haptics`/`BotNet`.
- `WorldRenderer` — territory as per-player coverage textures through a smoothstep shader, trails as
  miter ribbons, heads as oriented squares. `Hud` — score, minimap, joystick, name labels.
  `Ui`/`Draw` — shared screen-space context and helpers.

`core/` is shared with the sibling games: `Palette` (light/dark themes, 12 team colours, index 0 is
the human), `Assets` (fonts rasterised from the two bundled `.ttf`s — the only bundled assets; new UI
text needs its glyphs in `CHARS_UPPER`/`CHARS_FULL` or it renders blank), `SoundFx` (PCM synthesised
at startup), `Settings`, `Scores`, `Haptics`.

## The gym (`gym/`)

Self-play PPO training against a headless re-implementation of `World.kt`'s rule kernel. Separate
Python project managed by `uv` (PyTorch, Numba, numpy).

- `env.py` — batched Numba sim mirroring `World.substep`. One env step is one decision, `SUBSTEPS`
  sub-steps holding the chosen target heading. `encode_all` is the observation encoder.
- `rules.py` — constants mirrored 1:1 from `World.kt`. `scripted.py` — Numba port of `Bot.kt`.
- `model.py` — actor-critic ResNet, ≈0.53M params as shipped.
- `train.py` — clipped PPO, true self-play, per-agent GAE masked across permadeath gaps, optional
  league. `hold_weight` in the reward is the anti-turtle lever.
- `tournament.py` crowns a champion, `export.py` writes it into the app assets as `bot.onnx`,
  `run_pipeline.sh` chains base → league → tournament → export.
- `tests/test_env.py` — rule-kernel parity tests mirroring `SimTest.kt`; they must pass before
  training. `cd gym && uv sync && uv run python tests/test_env.py`.

## Conventions

- `World`, `Player`, `Bot` and `Geom` stay free of libGDX and Android imports — that is what makes
  the headless tests work. Side effects leave `World` through the `onSound`/`onHaptic`/`onLog` seams
  and the bot brain enters through `neuralDecide`; `BotNet` is the boundary and uses nothing else.
- Sound and haptics fire for human events only, guarded by `p.isHuman`.
- All `owner` writes go through `setOwner()` so each player's `area` and owned-cell bbox stay correct
  incrementally.
- The trail brush stays ≥1 cell thick (no flood-fill gaps) and never paints on own territory.
- Neural bots decide every `DECIDE_SUBSTEPS` sub-steps; this must equal the gym's `SUBSTEPS`, and the
  action mapping in `BotNet` must match the gym's.
- Gym and game parity: changing `World.kt`'s rules, movement or the observation encoding means
  mirroring it in `gym/paper_loop/{env,rules}.py` and `BotNet.encodeAt`, re-running the parity tests,
  and retraining — an out-of-distribution net plays badly. The encoder exists in two places; keep
  them in lockstep.
- Verify gameplay changes with `:app:testDebugUnitTest` and a screenshot. Sim and gym changes also
  need `gym/tests/test_env.py`.
