# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Paper Loop 2** (`applicationId` / package `paper.loop2`) is a single-activity Android game: a
2D Paper.io-**2**-style territory-capture game built with **libGDX** and **Kotlin**. It is the
free-movement successor to its sibling **Paper Loop** (`../paper-io-clone`, package `paper.loop`).

The defining difference from Paper Loop: **movement is free/continuous, not grid-locked.** You no
longer step square-by-square in 4 directions — the head has a continuous float position + heading and
is steered toward a target angle at a capped turn rate (smooth curves, like Paper.io 2). The **map is
a circle**, the trail is a smooth ribbon, and territories render as smooth rounded blobs.

Opponents are **neural bots** — a small self-play PPO net (`BotNet`) trained in the headless gym
(see **`gym/`** + the RL section below), with the scripted **`Bot`** as the automatic fallback (and
the gym's benchmark opponent). Like Paper Loop 1, inference has **two interchangeable backends**:
`onnxruntime-android` (fast primary) and a **pure-Kotlin** forward (F-Droid-clean fallback +
cross-check). No networking, no multiplayer. Privacy-respecting, `VIBRATE`-only, intended for
GPL-v3 / F-Droid release like its siblings.

It is an **Eve Games** sibling of `../cube-run`, `../edge-roll`, and `../paper-io-clone` and follows
the same release/F-Droid pipeline — see **`../EVE_GAMES_PLAYBOOK.md`** (shared keystore, fastlane
metadata, reproducible builds, `Eve-146T` GitHub).

> **Architecture note:** like Paper Loop, this is **all-libGDX** — every screen, the HUD, and the
> game-over card are drawn inside libGDX. There is no native-view HUD and no two-thread marshalling.

## Build, run, test

Requires JDK 17 (system JDK 21 also builds fine), Android SDK platform 35, build-tools 35.0.0.
SDK location comes from `local.properties` (`sdk.dir=…`) or `$ANDROID_HOME`.

```bash
./gradlew :app:assembleDebug          # -> app/build/outputs/apk/debug/app-debug.apk
./gradlew :app:testDebugUnitTest      # headless JVM unit tests (see "Testing" below)
./gradlew :app:installDebug           # install to a connected device
./gradlew :app:lint                   # ObsoleteSdkInt disabled (adaptive-icon false positive)
```

**Emulator verification** (AVDs `eve-pool-1/2` exist; emulators usually already running as
`emulator-5554` / `emulator-5556`; KVM available):
```bash
adb -s emulator-5554 install -r app/build/outputs/apk/debug/app-debug.apk
adb -s emulator-5554 shell am start -n paper.loop2/paper.loop2.GameActivity
adb -s emulator-5554 exec-out screencap -p > /tmp/shot.png   # then Read the PNG to inspect
# drive it: `input tap 540 700` (tap to begin), `input swipe x1 y1 x2 y2 ms` (steer ~ joystick angle)
```
Note: a "**System UI isn't responding**" dialog on an emulator is host load (running 2 emulators +
a device at once), **not** an app ANR — check logcat for `ANR in paper.loop2` / `FATAL .* paper.loop2`
to confirm the app is fine.

**libGDX natives** (incl. **gdx-freetype**) are pulled from Maven and extracted into `app/libs/<abi>`
by `copyAndroidNatives` before the JNI merge; `app/libs/` is gitignored and re-created each build.
Toolchain: AGP 8.7.3, Kotlin 2.0.21, Gradle 8.13, libGDX 1.13.1, min/target/compile = 29/35/35.

## Testing (this is new vs Paper Loop, which had none)

The **core simulation is pure Kotlin with no libGDX/Android imports** (see the effect seam below), so
it runs headless on the JVM. `app/src/test/kotlin/paper/loop2/game/SimTest.kt` drives `World` with a
fixed sub-step and checks the fundamentals: capture grows territory, the incremental `area` count
matches the grid, the wall and self-crossing kill, and a 40s 8-player bot run stays stable + productive.
**Run `:app:testDebugUnitTest` after any change to `World`/`Player`/`Bot`/`Geom`.** Keep these four
files free of libGDX/Android imports so the tests keep compiling on plain JVM.

## Architecture

Package root `paper.loop2` (under `app/src/main/kotlin/`). `App` inits the Android-context singletons
(`Scores`, `Settings`, `Haptics`); `GameActivity` runs `PaperLoopGame`. `SoundFx.init()` runs after
`PaperLoopGame.create()` (it needs `Gdx.audio`). Startup is tuned identically to Paper Loop: only the
1px white texture is made on the GL thread up front, fonts rasterise on a background thread (gated by
`Assets.loaded`, faded in), the sound bank synthesises on another — the live world draws from frame 1.

### `game/` — the game

- **`Geom.kt`** — angle helpers (`wrapAngle`, `HALF_PI`/`PI_F`/`TWO_PI`), a primitive `FloatArrayList`
  (the trail polyline, GC-free), and `distSqPointSeg` (self-cross test). No libGDX.
- **`Player`** — one player. Continuous head: `x`,`y` (cell coords, y-up, float), `heading`,
  `targetHeading`, `prevX/prevY` (last sub-step, for brush laying). Trail: `hasTrail`, `exitX/exitY`
  (where it left territory — home anchor + dissolve seed), `trailX`/`trailY` (decimated polyline) and
  `trailCells` (grid cells stamped, in lay order). Plus `area`/`peakArea`/`kills`, scripted-bot
  fields, and fx (`deathFx`, `respawn`). Also an **expand-only owned-cell bbox** (`bbMinX/Y`/`bbMaxX/Y`,
  maintained in `World.setOwner`, reset at spawn) + a **trail bbox** (`trMinX/Y`/`trMaxX/Y`) — pure-data
  hints the renderer and capture use to work on a sub-rect instead of scanning the whole grid. No libGDX.
- **`World`** — the simulation (no libGDX/Android imports). The board is two flat arrays of size `GW*GH`:
  `owner` (the territory source of truth, `playerId+1`) and `trail`. **Movement is fixed-sub-step** (`SUB_DT`, 90 Hz) with continuous integration:
  `update(dt)` accumulates and runs `substep()`s. Each sub-step: bots decide (throttled), every alive
  head turns toward `targetHeading` (capped at `TURN_RATE`) and advances by `SPEED`, then:
  - **Trail brush** (`layTrail`/`stampSeg`/`stampDisc`): while outside its own land the head paints a
    thick disc (`TRAIL_R`) of trail cells along `prev→cur` and records a decimated polyline. The brush
    is wide enough (≥1 cell) that the band never has flood-fill gaps. Painting over an **enemy** trail
    cell **cuts** that enemy (kills them). It never paints over its own territory.
  - **Capture** (`capture`): when the head re-enters its own territory with a trail, the trail cells
    become territory, then a BFS floods "outside" from the border treating self-owned cells as walls;
    any non-self cell the flood can't reach is enclosed → claimed (enclosed enemy trail → that enemy
    dies). **This is the same proven grid flood-fill as Paper Loop** — the grid is what makes capture
    robust for arbitrary free-form loops (no fragile polygon-boolean math). The flood + final scan are
    **limited to the player's owned bbox (+1 ring)**, not the whole 200×200 grid: enclosed cells are
    surrounded by self so they're inside the bbox, and the region between the +1 ring and the grid edge
    is all non-self/connected — so the result is **identical** to a full-grid flood, just far cheaper
    (the arena being a circle well inside the grid guarantees the +1 ring is a valid non-self seed).
  - **Collisions** (`resolveCollisions`): head-to-head within `HEAD_KILL` (survivor by turf-priority
    then territory, tie kills both); a head with **no** trail running onto an enemy trail cuts them;
    crossing your **own** trail (geometry test on the polyline, skipping `SELF_GRACE` of path behind
    the head so the freshly-laid tail never false-triggers) kills you. The **circular wall**: grazing
    the ring at a shallow angle (≤60° from tangent, `v·n ≤ WALL_DIE_DOT = sin60°`) clamps you onto the
    ring and you **slide** along it (a touch slower); slamming in steeper kills. The **start-block** rule
    (Paper Loop's) is kept: a live trail whose exit cell is captured out dies.
  - **Death dissolve** (restored to the v1 feel): on death the trail **ribbon zips back** (the renderer
    retracts the kept polyline via `Player.dyingFrac`, paced by `TRAIL_RETRACT`), then the territory
    dissolves outward (the per-player texture melts as `owner` clears in `advanceDissolves`). `GameScreen`
    keeps simulating in the **OVER** phase so the animation finishes behind the game-over card.
  - **Circular arena & frame-independence:** the map is a **circle** (`ARENA_R` inside the `GW*GH`
    grid) — spawns ring inside it, wall death is radial, and the human starts heading **inward** so a
    near-edge spawn never drives straight into the wall. Movement is fixed-sub-step (`SUB_DT`) with the
    `GameScreen` dt clamped to 0.1 s, so it stays **real-time even at low frame rates** (no slow-motion
    on a slow device); `World.ownerVersion` bumps on every territory change to trigger renderer rebuilds.
  - **Death dissolve** (`killPlayer`→`startDissolve`/`advanceDissolves`): unchanged from Paper Loop —
    the trail retracts to its entry point, then the territory dissolves outward as a circular ripple,
    flashing in the dying colour. Bots respawn (`respawnBot`) if `respawnEnabled`; the human ends the run.
  - **Invariant: sound/haptics fire for HUMAN events only** — guarded by `p.isHuman`. Preserve this.
  - **Effect seam:** `World` has `onSound`/`onHaptic`/`onLog` function properties (default no-ops),
    wired by `GameScreen` to `SoundFx`/`Haptics`/`Gdx.app.log`. This is what keeps `World` free of
    Android deps and headless-testable — do not call libGDX/Android directly from `World`.
  - **Bot-brain seam:** `World` also has `neuralDecide: ((World) -> Unit)?` (default null → scripted
    `Bot`). When set (by `GameScreen` to `BotNet.actAll`), it drives all non-human bots. Kept a plain
    lambda so `World` stays libGDX/Android-free. `actAll` is called every sub-step but is **cheap** — the
    net forward runs **off-thread** (see `BotNet` below); `actAll` only snapshots once per
    `DECIDE_SUBSTEPS` window and applies the worker's latest result. Neural bots decide every
    `DECIDE_SUBSTEPS` sub-steps (≈`BOT_DECIDE_DT`) and hold target heading between — this cadence
    **must equal the gym's `SUBSTEPS`**.
- **`Bot`** — scripted free-movement AI (the **fallback** + the gym's benchmark opponent). A definite
  venture: **out** leg → 90° **side** leg → **beeline home** to the exit anchor (re-entering territory
  banks the loop). A reactive `bestHeading` filter probes candidate headings and steers away from the
  wall, from re-crossing its own trail, and out of pockets too small to escape (`reachFrom`). Bails
  home early if the trail is long or an enemy head is near. Only ever sets `targetHeading`. (~6% over a
  minute.) Used when no neural brain is wired (or when the net's weights fail to load).
- **`BotNet`** (libGDX/Android — NOT in the pure core) — the **neural brain**: the small ~0.53M-param
  self-play PPO net (same architecture + param count as Paper Loop 1's shipped net; only the policy
  head differs — **5 relative-turn buckets** for free movement vs 3 grid turns). `encode()` is a 1:1
  port of the gym's `encode_all` (egocentric crop **rotated so forward=up** for continuous heading;
  6 grid channels + 12 scalars). Two interchangeable backends, like Paper Loop 1: **ONNX**
  (`onnxruntime-android`, the fast primary — ~2.2 ms/bot at `ORT_THREADS`=3) and a **pure-Kotlin** forward
  (no native dep; F-Droid-clean fallback + correctness cross-check). Weights load from `assets/bot.onnx`
  + `assets/bot.plb`; any failure → scripted `Bot`.
  **Action mapping must match the gym:** `targetHeading = heading + (action − 2)·(MAX_TURN_PER_DECISION/2)`.
  - **Inference runs OFF the render/sim thread** (a dedicated daemon `worker`, below-normal priority).
    Each `DECIDE_SUBSTEPS` window the sim thread cheaply **snapshots** what the encoder reads (two 40 KB
    grid copies + per-player kinematics → `work*` arrays, guarded by `inFlight`/`lock`); the worker
    `encodeAt`s all due bots, runs **one batched** forward, and publishes target headings back, which the
    next `apply()` writes to `Player.targetHeading` (a `resetGen` guard discards results from before a
    reset). This is **the** perf win on weak devices: inference can't stall rendering nor multiply across
    sub-steps, so there's no frame-time spiral — the moto g(7) power holds a **locked 60 fps** (was ~37
    fps with synchronous on-thread inference). `encodeAt` reads the snapshot, **not** the live world; it
    stays a 1:1 port of the gym's `encode_all` (keep it that way — gym parity). Player index == id (the
    list is id-ordered, never reordered). ONNX session tuning lives in `BotNet`'s companion
    (`ORT_THREADS`, `ORT_XNNPACK` — XNNPACK is **off**: it can't handle the net's dynamic-batch `Concat`).
    A startup warm-up pass JITs the encode/forward and primes the session so the first decision isn't a
    cold spike. `GameScreen` has a gated frame profiler (`PROFILE`/`BENCH_ON_START` consts, off by
    default; `BotNet.benchmark()` is the latency/cross-check probe) — flip `PROFILE=true` to re-benchmark.
- **`GameScreen`** — the only screen; a `Phase` machine: **READY** (sim frozen, menu, "tap to begin"),
  **PLAYING** (sim runs, HUD), **OVER** (game-over card over a scrim). The floating joystick sets the
  human's **absolute `targetHeading`** via `atan2` (screen y is down → inverted): `steer()`. Start
  menu cycles the opponent type — **AI HARD** (neural `BotNet`) → **EASY SCRIPTED** (the scripted `Bot`)
  → **NO BOTS** (`botMode` 0/1/2, like Paper Loop 1) — plus **RESPAWN** (best score bucketed per
  setting: hard/easy × respawn/norespawn, + a single nobots bucket). `applyBotFlags()` wires the bot
  brain per mode (`world.neuralDecide = neuralLambda` only in AI HARD; null → scripted otherwise);
  AI HARD falls back to scripted if the net failed to load. Wires the world effect seams + the neural
  `BotNet` (crash-safe: any inference error reverts to scripted) in `init`.
- **`WorldRenderer`** — the free-movement look (see below).
- **`Hud`** — small top-left `%` + `#rank · best`, a `Pixmap`→`Texture` minimap (territories + trails),
  the floating joystick, and floating name labels (`(You)` for the human) anchored above each head's
  projected top edge. Reads continuous `p.x`,`p.y` (no grid-cell `+0.5`).
- **`Ui`/`Draw`** — shared screen-space rendering context + `roundRect` helper (unchanged from sibling).

### Rendering & the smooth look (`WorldRenderer`)

World-space `OrthographicCamera` (`viewCells` ≈ 26 cells across); camera rigidly follows the human's
continuous position. Per frame: a `ShapeRenderer` pass for the **circular field + wall ring** (a big
`circle(ARENA_CX, ARENA_CY, ARENA_R, 90)` in `ARENA`, over a slightly larger `WALL` circle), then a
`SpriteBatch`+shader pass for territory, then a `ShapeRenderer` pass for trails + heads. Off-screen
turfs/trails/heads are frustum-culled (see below).

- **Territory = smooth blobs via PER-PLAYER coverage textures + a smoothstep shader.** Each player has
  its own coverage texture (`GW*GH`, alpha = "this player owns this cell"; colour comes from the batch
  vertex colour, not the texture). On rebuild the binary coverage is separable **box-blurred** (`blurR`)
  so the rasterised cog edge reads as a smooth disc (a straight capture edge stays straight); the shader
  does `smoothstep(0.5 ± edge, coverage)` — the **bilinear** 0.5 iso-line is a smooth curve, snapped
  crisply. **Why per-player and not one combined texture:** a single texture can only smooth the
  *coverage*, leaving the per-cell *colour* boundary where two turfs meet jagged. With a texture each,
  every territory has its own anti-aliased edge, so two turfs meeting blend smoothly. Drawn in two
  sweeps: all **rims** first (a darker, downward-shifted copy = 3D slab side, so a rim never darkens
  over another's top), then all **tops** (bots first, human last for the cleanest edges on your turf).
  **Rebuilt only for players whose `area` changed** since the last frame (tracked vs `lastArea`): idle =
  nothing, a capture = 1-2 players, a dissolve = one player per frame. Far cheaper than the old per-frame
  marching squares. **A rebuild touches only the player's owned bbox + blur fringe** (windowed separable
  blur + a `glTexSubImage2D` of just that sub-rect), not the whole 200×200 grid — on a throttled core that
  cut a rebuild from ~12.7 ms to ~1-3 ms. Because the bbox is expand-only, the uploaded region only ever
  *shrinks* on respawn / new game; that shrink is detected and the texture wiped first so old turf can't
  linger (`clearTexture`; also done on texture create). **Off-screen turfs are not rebuilt** — a dirty
  turf outside the camera view rect is left dirty and rebuilds the frame it scrolls into view, so a flurry
  of off-screen captures costs one rebuild on entry, not one each. **Gotcha:** a respawn can land on the
  *same* area as the dead state, so the area dirty-check alone would keep a stale texture → "you spawn with
  no platform". `World.resetGen` bumps on every `reset()`; the renderer force-rebuilds all textures (and
  the shrink-detect wipes their stale region) when it changes. Don't remove that.
- **Frustum culling:** territory quads, trails, and heads whose bbox (owned bbox / trail bbox / head pos,
  + a small `CULL_MARGIN`) don't intersect the camera view rect are **skipped** — the camera shows ~26 of
  the arena's ~188 cells, so most turfs/bots are off-screen. The human turf-redraw-over-trail pass (step 4,
  tucking the ribbon root under your own edge) is also **skipped when you have no trail**.
- **Trail = smooth ribbon, LIGHTER than the turf.** A single-layer **miter triangle strip** from the
  polyline (+ live head), half-width ≈ `TRAIL_R`, drawn in the team colour lerped toward white (so the
  trail reads as a distinct ribbon, not more territory). The polyline starts a couple cells *inside* the
  territory (`TRAIL_ROOT`, set in `World.layTrail`) so the ribbon attaches cleanly under the smooth turf
  edge. (Single-layer, not overlapping circles/rectLines — those composite darker and look beaded.)
- **Head** = a small team-colour square oriented to `heading`, with a **drop shadow shifted down in
  world space** (consistent at any heading — not a rotated-down quad, which read as a wedge). Death = an
  expanding burst. 2× MSAA (`AndroidApplicationConfiguration.numSamples`) cleans the diagonal edges.
- The **minimap** (`Hud`) masks cells outside `ARENA_R` to transparent, so it reads as a disc too. It is
  refreshed ~5×/s (throttled) and packed into an `IntArray` for one **bulk** upload (not per-pixel
  `drawPixel`), so it's cheap even on a weak core.
- There is **no capture/dissolve flash overlay**: the per-player texture melts as `owner` clears during a
  dissolve, which already reads as the ripple. (The old unused `World.flash`/`flashColor` arrays were
  removed — they were written every frame but never rendered, pure waste on the sim thread.)

### `core/` — shared utilities (reused from Paper Loop, package-renamed)

- **`Palette`** — light "paper" (cream) + dark themes sharing 12 vivid team colours (index 0 = human).
  The human's colour is **random each play**; UI accents follow it.
- **`Assets`** — rasterises `BitmapFont`s from the two bundled `.ttf`s via gdx-freetype + a 1px white
  texture. **These fonts are the only bundled assets** (no images/atlases/audio). If you add UI text,
  add its glyphs to `CHARS_UPPER`/`CHARS_FULL` or they render blank.
- **`SoundFx`** — synthesises effects as PCM at startup, plays via libGDX `Sound`, fully defensive.
  Names: `capture`, `place`, `die`, `eliminate`, `tap`, `start`, `ripple`.
- **`Settings`** (sound/haptics/dark/`botMode` 0=AI HARD 1=EASY SCRIPTED 2=NO BOTS/`respawnBots`), **`Scores`** (best per
  id, tenths-of-a-percent), **`Haptics`** (gated by `Settings.hapticsOn`).

## Reinforcement learning — the gym (`gym/`)

The neural bots are trained by **self-play PPO** in a headless, **byte-faithful** re-implementation of
`World.kt`'s rule kernel — the same recipe as Paper Loop 1's gym, adapted for free movement. It is a
separate Python project (`gym/`, managed by `uv`; PyTorch + Numba + numpy; uses the RTX 3090).

- **`paper_loop/env.py`** — batched **Numba** sim (`prange` over `B` envs). Mirrors `World.substep`
  exactly: continuous head integration (steer toward target heading capped at `TURN_RATE`, advance by
  `SPEED`), thick trail **brush** (`stampDisc`/`stampSeg`), grid **capture** (BFS flood + enclosure
  elimination), continuous collisions (head-to-head turf/area priority, trail-cut + credit, geometric
  **self-cross** with `SELF_GRACE`), and the **circular-wall** slide/die. One env step = one *decision*
  = `SUBSTEPS` (=4) fixed sub-steps holding the chosen target heading (matches on-device cadence).
  Permadeath by default; `respawn` knob mirrors the game's RESPAWN toggle. `encode_all` = the
  egocentric crop **rotated forward=up** (6 grid channels + 12 scalars) — **ported 1:1 to `BotNet.encodeAt`**.
- **`paper_loop/rules.py`** — constants mirrored 1:1 from `World.kt` (single source of truth).
- **`paper_loop/scripted.py`** — Numba port of `Bot.kt` (the benchmark opponent).
- **`paper_loop/model.py`** — actor-critic ResNet (stride-2 stem 21→11→6, flatten, scalar MLP, trunk,
  policy head = **`N_ACTIONS`=5** logits, value head). Shipped size = **width 32 / blocks 2 /
  scalar 64 / trunk 128 ≈ 0.53M params** (same as Paper Loop 1's on-device net).
- **`train.py`** — clipped PPO, true self-play, per-agent GAE masked across permadeath gaps; optional
  **league** (`--league_frac`). Reward = `area_scale·Δarea + kill_reward·kills + hold_weight·area_held −
  death/step + win/place`; **`hold_weight` is the anti-turtle lever** (the one lesson that matters).
- **`tournament.py`** crowns a champion by round-robin head-to-head + vs-scripted. **`export.py`** writes
  the champion to the app assets in **both** formats: `bot.onnx` (single self-contained file; onnxruntime
  primary) and `bot.plb` (PL21 named-float32 blob; pure-Kotlin backend). **`run_pipeline.sh`** chains
  base → league → tournament → export unattended.
- **`tests/test_env.py`** — rule-kernel **parity** tests (mirror `SimTest.kt`): capture grows territory,
  area invariant, wall slide-vs-slam, self-cross, long scripted run stable. **Must pass before training.**
  Reproduce: `cd gym && uv sync && uv run python tests/test_env.py && uv run python bench.py`, then
  `uv run python train.py --run base`. The ONNX/Kotlin backends are cross-checked on-device (argmax
  agreement + max-logit-diff; verified identical).

## Where the knobs live

- Grid / circular arena / speed / turn rate / head & trail size / starting disc / grace distances: the
  `const`s at the top of **`World.kt`** (`GW`/`GH`, `ARENA_CX`/`ARENA_CY`/`ARENA_R`, `SUB_DT`, `SPEED`,
  `TURN_RATE`, `HEAD_R`, `TRAIL_R`, `START_R`, `SELF_GRACE`, `HEAD_KILL`, `WALL_MARGIN`, `WALL_DIE_DOT`
  (wall graze/slam threshold), `TRAIL_ROOT` (ribbon attach depth), `TRAIL_RETRACT`, `RIPPLE_SPEED`).
- Number of opponents: `numBots` in `PaperLoopGame`. Scripted bot behaviour: `const`s at the top of
  `Bot.kt`. Neural bots: retrain in `gym/` and re-`export.py`; decision cadence + action mapping in
  `World.kt` (`DECIDE_SUBSTEPS`, `N_ACTIONS`, `MAX_TURN_PER_DECISION`) **must match the gym's `rules.py`**.
- On-device inference perf: `ORT_THREADS` / `ORT_XNNPACK` in `BotNet`'s companion (intra-op threads for
  the off-thread worker; XNNPACK stays off — dynamic-batch `Concat` is unsupported). Frame-profiler
  toggle: `PROFILE` / `BENCH_ON_START` consts in `GameScreen` (logcat tag `PERF` / `BotNetBench`).
- Camera zoom: `viewCells` in `WorldRenderer`. Blob smoothness/rim/ribbon look: `blurR`, `edge`,
  `rimDepth`, `rimDark` and the shader near the top of `WorldRenderer`. Off-screen cull padding:
  `CULL_MARGIN`. Minimap refresh rate: the `mmTimer` reset in `Hud`. Field-ring segments: `seg` in
  `WorldRenderer.render`.
- Launcher icon: `res/drawable/ic_launcher_foreground.xml` + `ic_launcher_background` in `colors.xml`.
- Colours / theme: `Palette`. Font sizes + glyph sets: `Assets`.

## Conventions (don't regress)

- Keep `World`/`Player`/`Bot`/`Geom` free of libGDX/Android imports (headless tests + the effect seam).
  `BotNet` is the libGDX/Android boundary — it reaches `World` only through the `neuralDecide` seam.
- Sound/haptics fire for **human** events only.
- All `owner` writes go through `setOwner()` so each player's `area` **and** owned-cell bbox stay correct
  incrementally. The bbox/cull additions are a pure render+capture speedup with **identical** sim results,
  so they don't affect gym parity (the gym can keep flooding the full grid).
- The trail brush must stay ≥1 cell thick (no flood-fill gaps) and must never paint on own territory.
- **Gym ⇄ game parity:** if you change `World.kt`'s rules, movement, or the obs encoding, mirror it in
  `gym/paper_loop/{env,rules}.py` + `BotNet.encodeAt`, re-run the parity tests, and **retrain** — an
  out-of-distribution net plays badly. The encoder is duplicated in 2 places (gym + `BotNet`); keep them in lockstep.
- Verify gameplay changes with `:app:testDebugUnitTest` **and** an emulator screenshot. For the sim/gym,
  also run `gym/tests/test_env.py`.
