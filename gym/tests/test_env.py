"""Rule-kernel parity tests for the free-movement env — mirrors app SimTest.kt.

Checks the fundamentals: capture grows territory, the incremental AREA matches the
grid, the circular wall slides at a graze but kills on a slam, self-crossing kills,
and a long scripted run stays stable + productive. Run:

    uv run python tests/test_env.py     (or: uv run pytest tests/)

Single-agent tests steer by writing FHEAD directly and stepping the STRAIGHT action
(offset 0), so the head advances along the chosen absolute heading.
"""
import os, sys, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_loop.env import PaperLoop2Env, EnvConfig
from paper_loop.scripted import ScriptedBots
from paper_loop.rules import (
    GW, GH, N_CELLS, ARENA_CX, ARENA_CY, ARENA_R, WALL_MARGIN, START_R,
    N_ACTIONS, HALF_PI, PI_F, MAX_PLAYERS,
    ALIVE, AREA, PEAK, HAS, NPLAY,
    FX, FY, FPX, FPY, FHEAD, FTHEAD,
)

STRAIGHT = (N_ACTIONS - 1) // 2          # center bucket = no turn
HARD_LEFT = 0
HARD_RIGHT = N_ACTIONS - 1


def single_env(seed=0, respawn=0):
    cfg = EnvConfig(B=1, n_min=1, n_max=1, max_steps=100000, seed=seed, respawn=respawn)
    return PaperLoop2Env(cfg)


def drive(env, heading, steps, action=STRAIGHT):
    """Advance `steps` env-steps along absolute `heading` (or with a fixed action)."""
    a = np.full((env.B, env.P), action, np.int32)
    for _ in range(steps):
        if heading is not None:
            env.pf[0, 0, FHEAD] = heading
        env.step(a, auto_reset=0)


def owner_count(env, b, pid):
    return int(np.count_nonzero(env.owner[b] == pid + 1))


# --------------------------------------------------------------------------- #
def test_capture_grows_territory():
    env = single_env(seed=42)
    start = int(env.pi[0, 0, AREA])
    assert start > 0, "starting disc should own cells"
    # a rectangular loop out of and back into the home disc (heading set absolutely)
    drive(env, 0.0, 11)          # east
    drive(env, HALF_PI, 11)      # up
    drive(env, PI_F, 11)         # west
    drive(env, -HALF_PI, 13)     # down, back into home
    assert env.pi[0, 0, ALIVE] == 1, "human should survive a clean loop"
    assert env.pi[0, 0, AREA] > start, f"closing the loop should claim area: {env.pi[0,0,AREA]} vs {start}"
    print(f"  capture: area {start} -> {int(env.pi[0,0,AREA])}  OK")


def test_area_count_matches_grid():
    env = PaperLoop2Env(EnvConfig(B=16, n_min=6, n_max=6, max_steps=100000, seed=7))
    bots = ScriptedBots(env, seed=7)
    mask = np.ones((env.B, env.P), np.uint8)
    for _ in range(300):
        env.step(bots.act(mask), auto_reset=1)
    bad = 0
    for b in range(env.B):
        n = int(env.ei[b, NPLAY])
        counts = np.bincount(env.owner[b].astype(np.int64), minlength=MAX_PLAYERS + 1)
        for p in range(n):
            if counts[p + 1] != env.pi[b, p, AREA]:
                bad += 1
    assert bad == 0, f"AREA drifted from the grid in {bad} (env,seat) lanes"
    print("  area invariant over 300 scripted steps  OK")


def test_long_scripted_run_stable_and_productive():
    env = PaperLoop2Env(EnvConfig(B=64, n_min=6, n_max=8, max_steps=100000, seed=99))
    bots = ScriptedBots(env, seed=99)
    mask = np.ones((env.B, env.P), np.uint8)
    for _ in range(700):                 # ~47s of game time; must not throw
        env.step(bots.act(mask), auto_reset=1)
    # every owner byte is a valid player id
    for b in range(env.B):
        n = int(env.ei[b, NPLAY])
        assert env.owner[b].max() <= n, "stray owner byte"
    best = env.pi[:, :, PEAK].max()
    assert best > 800, f"some scripted bot should capture territory (best peak={best})"
    print(f"  long scripted run stable; best peak {int(best)} cells "
          f"({best/N_CELLS*100:.1f}%)  OK")


def test_wall_kills():
    env = single_env(seed=3)
    # head straight outward from the centre into the ring
    x = env.pf[0, 0, FX]; y = env.pf[0, 0, FY]
    out = math.atan2(y - ARENA_CY, x - ARENA_CX)
    drive(env, out, 200)
    assert env.pi[0, 0, ALIVE] == 0, "running straight into the wall should kill"
    print("  wall slam kills  OK")


def test_wall_graze_slides_but_slam_kills():
    # graze tangentially -> survive and ride the ring
    env = single_env(seed=11)
    hx = ARENA_CX + (ARENA_R - WALL_MARGIN) - 1.0
    hy = ARENA_CY
    # give the head turf around the spot so leaving it can't trigger start-block
    cxI = int(hx); cyI = int(hy)
    for yy in range(cyI - 24, cyI + 25):
        for xx in range(cxI - 24, cxI + 25):
            dx = xx - hx; dy = yy - hy
            if 0 <= xx < GW and 0 <= yy < GH and dx * dx + dy * dy <= 24 * 24:
                env.owner[0, yy * GW + xx] = 1
    # recompute AREA after the manual stamp
    env.pi[0, 0, AREA] = owner_count(env, 0, 0)
    env.pf[0, 0, FX] = hx; env.pf[0, 0, FY] = hy
    env.pf[0, 0, FPX] = hx; env.pf[0, 0, FPY] = hy
    drive(env, HALF_PI, 15)          # up = tangent at the right edge
    assert env.pi[0, 0, ALIVE] == 1, "grazing the wall tangentially must not kill"
    d = math.hypot(env.pf[0, 0, FX] - ARENA_CX, env.pf[0, 0, FY] - ARENA_CY)
    assert d <= ARENA_R and d >= ARENA_R - 8, f"should ride along the ring, was {d:.1f}"

    # slam straight out -> die
    env2 = single_env(seed=12)
    env2.pf[0, 0, FX] = ARENA_CX + ARENA_R - 3.0
    env2.pf[0, 0, FY] = ARENA_CY
    env2.pf[0, 0, FPX] = env2.pf[0, 0, FX]; env2.pf[0, 0, FPY] = env2.pf[0, 0, FY]
    drive(env2, 0.0, 20)             # straight into the wall
    assert env2.pi[0, 0, ALIVE] == 0, "slamming into the wall must kill"
    print(f"  wall graze slides (r={d:.1f}), slam kills  OK")


def test_self_cross_kills():
    env = single_env(seed=5)
    drive(env, 0.0, 18)             # run well clear of home, laying trail
    # a tight max-rate circle that loops back onto the start of its own trail
    a = np.full((env.B, env.P), HARD_LEFT, np.int32)
    alive_end = True
    for _ in range(60):
        env.step(a, auto_reset=0)
        if env.pi[0, 0, ALIVE] == 0:
            alive_end = False
            break
    assert not alive_end, "looping back across our own trail should kill"
    print("  self-cross kills  OK")


ALL = [
    test_capture_grows_territory,
    test_area_count_matches_grid,
    test_long_scripted_run_stable_and_productive,
    test_wall_kills,
    test_wall_graze_slides_but_slam_kills,
    test_self_cross_kills,
]

if __name__ == "__main__":
    print("running free-movement rule-kernel parity tests...")
    fails = 0
    for t in ALL:
        try:
            t()
        except AssertionError as e:
            fails += 1
            print(f"  FAIL {t.__name__}: {e}")
        except Exception as e:
            fails += 1
            print(f"  ERROR {t.__name__}: {e}")
    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'} ({len(ALL)} tests)")
    sys.exit(1 if fails else 0)
