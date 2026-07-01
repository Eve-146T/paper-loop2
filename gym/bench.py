"""Throughput + invariant sanity for the free-movement env. Run:  uv run python bench.py"""
import time
import numpy as np

from paper_loop.env import PaperLoop2Env, EnvConfig
from paper_loop.rules import N_ACTIONS, OBS_C, OBS_S, MAX_PLAYERS
from paper_loop.scripted import ScriptedBots


def area_invariant(env):
    """owner cell counts must equal each player's incremental AREA, every env."""
    bad = 0
    for b in range(env.B):
        n = int(env.ei[b, 0])
        counts = np.bincount(env.owner[b].astype(np.int64), minlength=MAX_PLAYERS + 1)
        for p in range(n):
            from paper_loop.rules import AREA
            if counts[p + 1] != env.pi[b, p, AREA]:
                bad += 1
    return bad


def main():
    print("compiling + running (first call JITs numba)...", flush=True)
    cfg = EnvConfig(B=128, n_min=4, n_max=8, max_steps=2200, seed=0)
    env = PaperLoop2Env(cfg)
    rng = np.random.default_rng(0)

    # warm-up (compile)
    t0 = time.time()
    a = rng.integers(0, N_ACTIONS, size=(env.B, env.P)).astype(np.int32)
    env.step(a)
    print(f"  warm-up (compile) {time.time()-t0:.1f}s", flush=True)

    bad = area_invariant(env)
    print(f"  area invariant after 1 step: {'OK' if bad==0 else f'{bad} MISMATCH'}", flush=True)

    # random throughput
    N = 400
    t0 = time.time()
    for _ in range(N):
        a = rng.integers(0, N_ACTIONS, size=(env.B, env.P)).astype(np.int32)
        env.step(a)
    dt = time.time() - t0
    print(f"  random:   {N*env.B/dt/1e3:7.1f}k env-steps/s  ({N} steps, B={env.B})", flush=True)
    print(f"  area invariant (random):  {'OK' if area_invariant(env)==0 else 'MISMATCH'}", flush=True)

    # scripted throughput + productivity
    env2 = PaperLoop2Env(EnvConfig(B=128, n_min=6, n_max=6, max_steps=3000, seed=1))
    bots = ScriptedBots(env2, seed=1)
    mask = np.ones((env2.B, env2.P), np.uint8)
    t0 = time.time()
    M = 400
    for _ in range(M):
        a = bots.act(mask)
        env2.step(a, auto_reset=1)
    dt = time.time() - t0
    from paper_loop.rules import PEAK, N_CELLS
    peaks = env2.pi[:, :, PEAK].max(axis=1)
    terr = []
    for b in range(env2.B):
        n = int(env2.ei[b, 0])
        for p in range(n):
            terr.append(env2.pi[b, p, 2])  # AREA index = 2
    terr = np.array(terr)
    print(f"  scripted: {M*env2.B/dt/1e3:7.1f}k env-steps/s", flush=True)
    print(f"  area invariant (scripted): {'OK' if area_invariant(env2)==0 else 'MISMATCH'}", flush=True)
    print(f"  scripted max peak area frac: {peaks.max()/N_CELLS*100:.1f}%  "
          f"mean alive territory: {terr.mean()/N_CELLS*100:.2f}%", flush=True)

    # obs sanity
    g, s, al = env.obs_grid, env.obs_scalar, env.alive_start
    print(f"  obs grid {g.shape} dtype {g.dtype}  scalar {s.shape}", flush=True)
    print(f"  obs grid channel means: {g.reshape(-1, OBS_C, g.shape[3], g.shape[4]).mean(axis=(0,2,3))}", flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
