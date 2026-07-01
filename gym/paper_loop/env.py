"""Headless, batched Paper Loop 2 simulation (FREE continuous movement).

Mirrors the logical rule kernel of World.kt: continuous head integration (steer
toward a target heading at a capped turn rate), a thick trail brush stamped onto
the grid, grid capture (BFS flood + enclosure elimination), continuous collisions
(head-to-head by turf/area priority, trail-cut + credit, geometric self-cross), and
the circular-arena wall slide/die. `B` independent games step in parallel across
cores via numba prange.

One env step = one *decision*: the chosen relative-turn action sets the target
heading, then the sim advances SUBSTEPS fixed sub-steps holding it (matches the
on-device bot, which decides every BOT_DECIDE_DT and holds target heading between).

State (all numpy, batch dim first):
  owner [B, N_CELLS] int8     0=empty else playerId+1   (territory; source of truth)
  trail [B, N_CELLS] int8     0=none  else playerId+1
  pi    [B, P, NF]   int32    per-player int fields   (see rules.py)
  pf    [B, P, NFF]  float32  per-player float fields (continuous head + trail anchor)
  polx/poly [B, P, MAXPTS] float32  decimated trail polyline (self-cross geometry)
  ei    [B, ME]      int32    per-env fields (player count, step, fill token, done)

Render-only layers of World.kt (animated dissolve, trail retract, audio/haptics,
interpolation) are NOT ported: death is instant, respawn is an optional knob.
"""
from dataclasses import dataclass
from math import cos, sin, sqrt, hypot, floor, ceil, atan2

import numpy as np
from numba import njit, prange

from .rules import (
    GW, GH, N_CELLS, MAX_PLAYERS,
    ARENA_CX, ARENA_CY, ARENA_R,
    SUB_DT, SPEED, TURN_RATE, HEAD_R, TRAIL_R, START_R, TRAIL_DECIM, TRAIL_ROOT,
    SELF_GRACE, SELF_CUT_R, HEAD_KILL, WALL_MARGIN, WALL_DIE_DOT,
    SUBSTEPS, MAX_TURN_PER_DECISION, PI_F, TWO_PI, HALF_PI,
    ALIVE, HAS, AREA, PEAK, KILLS, TLEN, RESPAWN, POLN, NF,
    FX, FY, FPX, FPY, FHEAD, FTHEAD, FEXITX, FEXITY, FDEATHARC, NFF,
    NPLAY, ESTEP, TOKEN, EDONE, ME, MAXPTS,
    OBS_R_DEFAULT, OBS_C, OBS_S, N_ACTIONS,
)


# --------------------------------------------------------------------------- #
#  small math helpers (mirror Geom.kt)
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _wrap(a):
    # fold any angle into (-PI, PI] (same canonical representative as Kotlin wrapAngle)
    x = a % TWO_PI          # numba float % -> [0, TWO_PI)
    if x > PI_F:
        x -= TWO_PI
    return x


@njit(cache=True, inline="always")
def _dist_sq_point_seg(px, py, ax, ay, bx, by):
    abx = bx - ax; aby = by - ay
    apx = px - ax; apy = py - ay
    len2 = abx * abx + aby * aby
    if len2 <= 1e-6:
        t = 0.0
    else:
        t = (apx * abx + apy * aby) / len2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
    cx = ax + abx * t; cy = ay + aby * t
    dx = px - cx; dy = py - cy
    return dx * dx + dy * dy


@njit(cache=True, inline="always")
def _cellx(fx):
    c = int(floor(fx))
    if c < 0:
        c = 0
    elif c > GW - 1:
        c = GW - 1
    return c


@njit(cache=True, inline="always")
def _celly(fy):
    c = int(floor(fy))
    if c < 0:
        c = 0
    elif c > GH - 1:
        c = GH - 1
    return c


@njit(cache=True, inline="always")
def _owner_at(owner, b, fx, fy):
    return owner[b, _celly(fy) * GW + _cellx(fx)]


# --------------------------------------------------------------------------- #
#  ownership / death (mirror World.setOwner / killPlayer)
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _set_owner(owner, pi, b, cid, newid):
    old = owner[b, cid]
    if old == newid:
        return
    if old != 0:
        pi[b, old - 1, AREA] -= 1
    owner[b, cid] = newid
    if newid != 0:
        pi[b, newid - 1, AREA] += 1
        if pi[b, newid - 1, AREA] > pi[b, newid - 1, PEAK]:
            pi[b, newid - 1, PEAK] = pi[b, newid - 1, AREA]


@njit(cache=True)
def _kill(owner, trail, pi, b, v):
    """Permadeath: clear the dead player's trail + territory instantly."""
    self = v + 1
    for cid in range(N_CELLS):
        if owner[b, cid] == self:
            _set_owner(owner, pi, b, cid, 0)
        if trail[b, cid] == self:
            trail[b, cid] = 0
    pi[b, v, ALIVE] = 0
    pi[b, v, HAS] = 0
    pi[b, v, TLEN] = 0
    pi[b, v, POLN] = 0


# --------------------------------------------------------------------------- #
#  trail brush (mirror World.stampDisc / stampSeg / layTrail)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _stamp_disc(owner, trail, pi, doomed, rew_kill, b, p, cx, cy):
    """Stamp a thick disc of trail; painting over an enemy trail cell cuts that enemy."""
    self = p + 1
    x0 = int(floor(cx - TRAIL_R)); x1 = int(ceil(cx + TRAIL_R))
    y0 = int(floor(cy - TRAIL_R)); y1 = int(ceil(cy + TRAIL_R))
    r2 = TRAIL_R * TRAIL_R
    for yy in range(y0, y1 + 1):
        if yy < 0 or yy >= GH:
            continue
        for xx in range(x0, x1 + 1):
            if xx < 0 or xx >= GW:
                continue
            dx = xx + 0.5 - cx; dy = yy + 0.5 - cy
            if dx * dx + dy * dy > r2:
                continue
            cid = yy * GW + xx
            if owner[b, cid] == self:        # never lay trail on our own land
                continue
            cur = trail[b, cid]
            if cur == self:
                continue
            if cur != 0:                      # our brush crossed an enemy trail -> cut them
                v = cur - 1
                if pi[b, v, ALIVE] == 1 and v != p:
                    doomed[b, v] = 1
                    pi[b, p, KILLS] += 1
                    rew_kill[b, p] += 1.0
            trail[b, cid] = self
            pi[b, p, TLEN] += 1


@njit(cache=True)
def _stamp_seg(owner, trail, pi, doomed, rew_kill, b, p, ax, ay, bx, by):
    dist = hypot(bx - ax, by - ay)
    n = int(ceil(dist / (TRAIL_R * 0.6)))
    if n < 1:
        n = 1
    for i in range(1, n + 1):
        t = i / n
        _stamp_disc(owner, trail, pi, doomed, rew_kill, b, p,
                    ax + (bx - ax) * t, ay + (by - ay) * t)


@njit(cache=True, inline="always")
def _append_point(pf_unused, polx, poly, pi, b, p, x, y):
    n = pi[b, p, POLN]
    if n == 0:
        polx[b, p, 0] = x; poly[b, p, 0] = y; pi[b, p, POLN] = 1
        return
    dx = x - polx[b, p, n - 1]; dy = y - poly[b, p, n - 1]
    if dx * dx + dy * dy >= TRAIL_DECIM * TRAIL_DECIM:
        if n < MAXPTS:
            polx[b, p, n] = x; poly[b, p, n] = y; pi[b, p, POLN] = n + 1


@njit(cache=True)
def _lay_trail(owner, trail, pi, pf, polx, poly, doomed, rew_kill, b, p):
    self = p + 1
    x = pf[b, p, FX]; y = pf[b, p, FY]
    px = pf[b, p, FPX]; py = pf[b, p, FPY]
    inside_now = _owner_at(owner, b, x, y) == self
    if pi[b, p, HAS] == 1:
        _stamp_seg(owner, trail, pi, doomed, rew_kill, b, p, px, py, x, y)
        _append_point(pf, polx, poly, pi, b, p, x, y)
    elif not inside_now:
        # just left our territory: open a trail anchored at the last inside point
        pi[b, p, HAS] = 1
        pf[b, p, FEXITX] = px; pf[b, p, FEXITY] = py
        pi[b, p, TLEN] = 0
        pi[b, p, POLN] = 0
        h = pf[b, p, FHEAD]
        # ribbon starts a couple cells INSIDE (TRAIL_ROOT) — kept for parity with World.layTrail
        _append_point(pf, polx, poly, pi, b, p, px - cos(h) * TRAIL_ROOT, py - sin(h) * TRAIL_ROOT)
        _append_point(pf, polx, poly, pi, b, p, px, py)
        _stamp_seg(owner, trail, pi, doomed, rew_kill, b, p, px, py, x, y)
        _append_point(pf, polx, poly, pi, b, p, x, y)


@njit(cache=True)
def _head_crosses_own_trail(pf, polx, poly, pi, b, p):
    n = pi[b, p, POLN]
    if n < 2:
        return False
    x = pf[b, p, FX]; y = pf[b, p, FY]
    acc = hypot(x - polx[b, p, n - 1], y - poly[b, p, n - 1])
    i = n - 1
    while i >= 1:
        bx = polx[b, p, i]; by = poly[b, p, i]
        ax = polx[b, p, i - 1]; ay = poly[b, p, i - 1]
        seg = hypot(bx - ax, by - ay)
        if acc >= SELF_GRACE and _dist_sq_point_seg(x, y, ax, ay, bx, by) < SELF_CUT_R * SELF_CUT_R:
            return True
        acc += seg
        i -= 1
    return False


# --------------------------------------------------------------------------- #
#  capture (mirror World.capture) — trail -> territory + flood enclose
# --------------------------------------------------------------------------- #
@njit(cache=True, inline="always")
def _visit(owner, fillstamp, bfs, b, x, y, self, token, tail):
    cid = y * GW + x
    if owner[b, cid] != self and fillstamp[b, cid] != token:
        fillstamp[b, cid] = token
        bfs[b, tail] = cid
        tail += 1
    return tail


@njit(cache=True)
def _capture(owner, trail, fillstamp, bfs, pi, pf, ei, doomed, rew_kill, encl, b, p, n):
    self = p + 1
    # trail becomes territory
    for cid in range(N_CELLS):
        if trail[b, cid] == self:
            _set_owner(owner, pi, b, cid, self)
            trail[b, cid] = 0
    pi[b, p, HAS] = 0
    pi[b, p, TLEN] = 0
    pi[b, p, POLN] = 0

    # flood the outside from the border, treating self-owned cells as walls
    token = ei[b, TOKEN] + 1
    ei[b, TOKEN] = token
    tail = 0
    for x in range(GW):
        tail = _visit(owner, fillstamp, bfs, b, x, 0, self, token, tail)
        tail = _visit(owner, fillstamp, bfs, b, x, GH - 1, self, token, tail)
    for y in range(GH):
        tail = _visit(owner, fillstamp, bfs, b, 0, y, self, token, tail)
        tail = _visit(owner, fillstamp, bfs, b, GW - 1, y, self, token, tail)
    h = 0
    while h < tail:
        cur = bfs[b, h]; h += 1
        cx = cur % GW; cy = cur // GW
        if cx > 0:
            tail = _visit(owner, fillstamp, bfs, b, cx - 1, cy, self, token, tail)
        if cx < GW - 1:
            tail = _visit(owner, fillstamp, bfs, b, cx + 1, cy, self, token, tail)
        if cy > 0:
            tail = _visit(owner, fillstamp, bfs, b, cx, cy - 1, self, token, tail)
        if cy < GH - 1:
            tail = _visit(owner, fillstamp, bfs, b, cx, cy + 1, self, token, tail)

    # any non-self cell the flood didn't reach is enclosed -> claim it
    for v in range(n):
        encl[b, v] = 0
    for cid in range(N_CELLS):
        if owner[b, cid] != self and fillstamp[b, cid] != token:
            t = trail[b, cid]
            if t != 0 and t != self:
                encl[b, t - 1] = 1
                trail[b, cid] = 0
            _set_owner(owner, pi, b, cid, self)

    # only a player whose TRAIL was enclosed is cut (no "head engulf")
    for v in range(n):
        if encl[b, v] == 1 and pi[b, v, ALIVE] == 1 and v != p:
            doomed[b, v] = 1
            pi[b, p, KILLS] += 1
            rew_kill[b, p] += 1.0


# --------------------------------------------------------------------------- #
#  collisions (mirror World.resolveCollisions) — run after all heads move + lay
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _resolve_collisions(owner, trail, pi, pf, polx, poly, doomed, rew_kill, b, n):
    # head-to-head within HEAD_KILL: turf-priority then area decides the survivor
    for a in range(n):
        if pi[b, a, ALIVE] == 0 or doomed[b, a] == 1:
            continue
        for q in range(a + 1, n):
            if pi[b, q, ALIVE] == 0 or doomed[b, q] == 1:
                continue
            dx = pf[b, a, FX] - pf[b, q, FX]; dy = pf[b, a, FY] - pf[b, q, FY]
            if dx * dx + dy * dy < HEAD_KILL * HEAD_KILL:
                # pickCrashSurvivor: own turf wins, else bigger area, else tie (both die)
                surv = -1
                if _owner_at(owner, b, pf[b, a, FX], pf[b, a, FY]) == a + 1:
                    surv = a
                elif _owner_at(owner, b, pf[b, q, FX], pf[b, q, FY]) == q + 1:
                    surv = q
                else:
                    aa = pi[b, a, AREA]; aq = pi[b, q, AREA]
                    if aa > aq:
                        surv = a
                    elif aq > aa:
                        surv = q
                if surv == a:
                    doomed[b, q] = 1; pi[b, a, KILLS] += 1; rew_kill[b, a] += 1.0
                elif surv == q:
                    doomed[b, a] = 1; pi[b, q, KILLS] += 1; rew_kill[b, q] += 1.0
                else:
                    doomed[b, a] = 1; doomed[b, q] = 1
    # a head with NO trail (cruising its own land) running over an enemy trail cuts them
    for p in range(n):
        if pi[b, p, ALIVE] == 0 or doomed[b, p] == 1 or pi[b, p, HAS] == 1:
            continue
        cid = _celly(pf[b, p, FY]) * GW + _cellx(pf[b, p, FX])
        t = trail[b, cid]
        if t != 0 and t != p + 1:
            v = t - 1
            if pi[b, v, ALIVE] == 1:
                doomed[b, v] = 1; pi[b, p, KILLS] += 1; rew_kill[b, p] += 1.0
    # crossing your OWN trail (beyond the grace zone) kills you
    for p in range(n):
        if pi[b, p, ALIVE] == 1 and doomed[b, p] == 0 and pi[b, p, HAS] == 1:
            if _head_crosses_own_trail(pf, polx, poly, pi, b, p):
                doomed[b, p] = 1


# --------------------------------------------------------------------------- #
#  one continuous sub-step for env b (mirror World.substep, minus bot/audio)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _substep(owner, trail, pi, pf, polx, poly, ei, fillstamp, bfs,
             doomed, died, rew_kill, encl, b, n):
    # 1) move every head: steer toward target heading (capped), advance, wall slide/die
    for p in range(n):
        if pi[b, p, ALIVE] == 0:
            continue
        doomed[b, p] = 0
        pf[b, p, FPX] = pf[b, p, FX]; pf[b, p, FPY] = pf[b, p, FY]
        diff = _wrap(pf[b, p, FTHEAD] - pf[b, p, FHEAD])
        mt = TURN_RATE * SUB_DT
        if diff > mt:
            diff = mt
        elif diff < -mt:
            diff = -mt
        h = _wrap(pf[b, p, FHEAD] + diff)
        pf[b, p, FHEAD] = h
        pf[b, p, FX] += cos(h) * SPEED * SUB_DT
        pf[b, p, FY] += sin(h) * SPEED * SUB_DT
        ddx = pf[b, p, FX] - ARENA_CX; ddy = pf[b, p, FY] - ARENA_CY
        dist = sqrt(ddx * ddx + ddy * ddy)
        lim = ARENA_R - WALL_MARGIN
        if dist > lim:
            if dist > 1e-4:
                nx = ddx / dist; ny = ddy / dist
            else:
                nx = cos(h); ny = sin(h)
            pf[b, p, FX] = ARENA_CX + nx * lim
            pf[b, p, FY] = ARENA_CY + ny * lim
            into = cos(h) * nx + sin(h) * ny
            if into > WALL_DIE_DOT:
                doomed[b, p] = 1
    # 2) lay trail (brush; may cut enemies whose trail it paints over)
    for p in range(n):
        if pi[b, p, ALIVE] == 1 and doomed[b, p] == 0:
            _lay_trail(owner, trail, pi, pf, polx, poly, doomed, rew_kill, b, p)
    # 3) collisions among moved heads
    _resolve_collisions(owner, trail, pi, pf, polx, poly, doomed, rew_kill, b, n)
    # 4) apply deaths
    for p in range(n):
        if doomed[b, p] == 1 and pi[b, p, ALIVE] == 1:
            _kill(owner, trail, pi, b, p)
            died[b, p] = 1
    # 5) capture (head back on own land with a trail)
    for p in range(n):
        if pi[b, p, ALIVE] == 1 and pi[b, p, HAS] == 1:
            if _owner_at(owner, b, pf[b, p, FX], pf[b, p, FY]) == p + 1:
                _capture(owner, trail, fillstamp, bfs, pi, pf, ei,
                         doomed, rew_kill, encl, b, p, n)
    # 6) start-block: a live trail whose exit cell is no longer ours dies
    for p in range(n):
        if pi[b, p, ALIVE] == 1 and pi[b, p, HAS] == 1:
            ex = _cellx(pf[b, p, FEXITX]); ey = _celly(pf[b, p, FEXITY])
            if owner[b, ey * GW + ex] != p + 1:
                _kill(owner, trail, pi, b, p)
                died[b, p] = 1


# --------------------------------------------------------------------------- #
#  spawn / reset (mirror World.spawnAt / spawnSpots / respawnBot / reset)
# --------------------------------------------------------------------------- #
@njit(cache=True)
def _stamp_start_disc_direct(owner, b, p, x, y):
    """Stamp a round START_R territory disc with a direct owner write (area summed later)."""
    r = int(ceil(START_R))
    cx = int(floor(x)); cy = int(floor(y))
    r2 = START_R * START_R
    for yy in range(cy - r, cy + r + 1):
        if yy < 0 or yy >= GH:
            continue
        for xx in range(cx - r, cx + r + 1):
            if xx < 0 or xx >= GW:
                continue
            dx = xx + 0.5 - x; dy = yy + 0.5 - y
            if dx * dx + dy * dy <= r2:
                owner[b, yy * GW + xx] = p + 1


@njit(cache=True)
def _spawn_fields(pi, pf, b, p, x, y):
    pf[b, p, FX] = x; pf[b, p, FY] = y
    pf[b, p, FPX] = x; pf[b, p, FPY] = y
    h = atan2(ARENA_CY - y, ARENA_CX - x)        # face inward (avoids spawn wall-death)
    pf[b, p, FHEAD] = h; pf[b, p, FTHEAD] = h
    pf[b, p, FEXITX] = x; pf[b, p, FEXITY] = y
    pf[b, p, FDEATHARC] = 1.0
    pi[b, p, ALIVE] = 1
    pi[b, p, HAS] = 0
    pi[b, p, TLEN] = 0
    pi[b, p, POLN] = 0
    pi[b, p, RESPAWN] = 0


@njit(cache=True)
def _respawn(owner, trail, pi, pf, b, p):
    """Respawn player p at the emptiest spot (mirror World.respawnBot + spawnAt)."""
    self = p + 1
    best_cx = int(ARENA_CX); best_cy = int(ARENA_CY); best_empty = -1
    lim = ARENA_R - START_R - 4.0
    for _ in range(50):
        ang = np.random.random() * TWO_PI
        rr = np.random.random() * lim
        cx = int(ARENA_CX + cos(ang) * rr)
        cy = int(ARENA_CY + sin(ang) * rr)
        if cx < 8:
            cx = 8
        elif cx > GW - 9:
            cx = GW - 9
        if cy < 8:
            cy = 8
        elif cy > GH - 9:
            cy = GH - 9
        empty = 0
        for yy in range(cy - 4, cy + 5):
            for xx in range(cx - 4, cx + 5):
                if 0 <= xx < GW and 0 <= yy < GH and owner[b, yy * GW + xx] == 0:
                    empty += 1
        if empty > best_empty:
            best_empty = empty; best_cx = cx; best_cy = cy
    x = best_cx + 0.5; y = best_cy + 0.5
    r = int(ceil(START_R)); r2 = START_R * START_R
    for yy in range(best_cy - r, best_cy + r + 1):
        if yy < 0 or yy >= GH:
            continue
        for xx in range(best_cx - r, best_cx + r + 1):
            if xx < 0 or xx >= GW:
                continue
            dx = xx + 0.5 - x; dy = yy + 0.5 - y
            if dx * dx + dy * dy <= r2:
                cid = yy * GW + xx
                trail[b, cid] = 0
                _set_owner(owner, pi, b, cid, self)
    _spawn_fields(pi, pf, b, p, x, y)
    pi[b, p, PEAK] = pi[b, p, AREA]


@njit(cache=True)
def _reset_inline(owner, trail, pi, pf, ei, b, n_in, n_min, n_max):
    """Re-seed env b for a fresh episode (mirror World.reset / spawnSpots / spawnAt)."""
    for cid in range(N_CELLS):
        owner[b, cid] = 0
        trail[b, cid] = 0
    for p in range(MAX_PLAYERS):
        for f in range(NF):
            pi[b, p, f] = 0
        for f in range(NFF):
            pf[b, p, f] = 0.0
    for f in range(ME):
        ei[b, f] = 0
    if n_in > 0:
        n = n_in
    else:
        n = n_min + np.random.randint(0, n_max - n_min + 1)
    ei[b, NPLAY] = n
    base = np.random.random() * TWO_PI
    ring = ARENA_R * 0.55
    lim = ARENA_R - START_R - 3.0
    for p in range(n):
        ang = base + p * (TWO_PI / n)
        rr = ring * (0.7 + np.random.random() * 0.35)
        if rr > lim:
            rr = lim
        cx = int(ARENA_CX + cos(ang) * rr)
        cy = int(ARENA_CY + sin(ang) * rr)
        if cx < 8:
            cx = 8
        elif cx > GW - 9:
            cx = GW - 9
        if cy < 8:
            cy = 8
        elif cy > GH - 9:
            cy = GH - 9
        x = cx + 0.5; y = cy + 0.5
        _stamp_start_disc_direct(owner, b, p, x, y)
        _spawn_fields(pi, pf, b, p, x, y)
    # area accounting from the stamped discs
    for cid in range(N_CELLS):
        o = owner[b, cid]
        if o > 0:
            pi[b, o - 1, AREA] += 1
    for p in range(n):
        pi[b, p, PEAK] = pi[b, p, AREA]


@njit(cache=True)
def seed_numba(s):
    np.random.seed(s)


# --------------------------------------------------------------------------- #
#  parallel driver: one decision (SUBSTEPS sub-steps) for every env
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def step_all(owner, trail, pi, pf, polx, poly, ei, fillstamp, bfs,
             actions, rewards, dones, alive_start, area_before,
             doomed, died, rew_kill, encl, ep_ended,
             death_penalty, kill_reward, area_scale, win_bonus, place_weight,
             hold_weight, step_cost, max_steps, n_min, n_max, auto_reset,
             respawn, respawn_min, respawn_max):
    B = owner.shape[0]
    P = pi.shape[1]
    half_span = (N_ACTIONS - 1) * 0.5
    turn_step = MAX_TURN_PER_DECISION / half_span
    for b in prange(B):
        n = ei[b, NPLAY]
        ep_ended[b] = 0
        for p in range(P):
            rewards[b, p] = 0.0
            dones[b, p] = 0
            alive_start[b, p] = 0
        # respawn dead players whose timer has elapsed (mirror World.respawnBot)
        if respawn == 1:
            for p in range(n):
                if pi[b, p, ALIVE] == 0 and pi[b, p, RESPAWN] > 0:
                    pi[b, p, RESPAWN] -= 1
                    if pi[b, p, RESPAWN] == 0:
                        _respawn(owner, trail, pi, pf, b, p)
        for p in range(n):
            died[b, p] = 0
            rew_kill[b, p] = 0.0
            alive_start[b, p] = 1 if pi[b, p, ALIVE] == 1 else 0
            area_before[b, p] = pi[b, p, AREA]

        # set the target heading from this decision's relative-turn action
        for p in range(n):
            if pi[b, p, ALIVE] == 1:
                a = actions[b, p]
                off = (a - half_span) * turn_step
                pf[b, p, FTHEAD] = _wrap(pf[b, p, FHEAD] + off)

        # advance SUBSTEPS continuous sub-steps holding the target heading
        for _s in range(SUBSTEPS):
            _substep(owner, trail, pi, pf, polx, poly, ei, fillstamp, bfs,
                     doomed, died, rew_kill, encl, b, n)

        # arm respawn timers for anyone who died this decision
        if respawn == 1:
            for p in range(n):
                if died[b, p] == 1:
                    pi[b, p, RESPAWN] = respawn_min + np.random.randint(0, respawn_max - respawn_min + 1)

        # --- per-decision reward ---
        for p in range(n):
            if alive_start[b, p] == 0:
                continue
            if died[b, p] == 1:
                rewards[b, p] = death_penalty + rew_kill[b, p] * kill_reward
                dones[b, p] = 1
            else:
                gained = pi[b, p, AREA] - area_before[b, p]
                rewards[b, p] = (gained / N_CELLS) * area_scale \
                    + rew_kill[b, p] * kill_reward \
                    + (pi[b, p, AREA] / N_CELLS) * hold_weight - step_cost

        # --- episode end ---
        alive_count = 0
        max_area = 0
        for p in range(n):
            if pi[b, p, ALIVE] == 1:
                alive_count += 1
            if pi[b, p, AREA] > max_area:
                max_area = pi[b, p, AREA]
        ei[b, ESTEP] += 1
        ep_done = 0
        if respawn == 1:
            if max_area >= N_CELLS:
                ep_done = 1
        else:
            if alive_count <= 1:
                ep_done = 1
        if ei[b, ESTEP] >= max_steps:
            ep_done = 1
        if ep_done == 1:
            for p in range(n):
                if pi[b, p, ALIVE] == 1:
                    if respawn == 1:
                        if pi[b, p, AREA] >= N_CELLS:
                            rewards[b, p] += win_bonus
                    elif alive_count == 1:
                        rewards[b, p] += win_bonus
                    rewards[b, p] += (pi[b, p, AREA] / N_CELLS) * place_weight
                    dones[b, p] = 1
            ep_ended[b] = 1
            if auto_reset == 1:
                _reset_inline(owner, trail, pi, pf, ei, b, -1, n_min, n_max)


# --------------------------------------------------------------------------- #
#  observation encoding (egocentric, rotated forward=up; arbitrary heading)
# --------------------------------------------------------------------------- #
@njit(parallel=True, cache=True)
def encode_all(owner, trail, pi, pf, ei, out_grid, out_scalar, max_steps):
    B = owner.shape[0]
    H = out_grid.shape[3]
    W = out_grid.shape[4]
    R = (H - 1) // 2
    wall_lim2 = (ARENA_R - WALL_MARGIN) * (ARENA_R - WALL_MARGIN)
    for b in prange(B):
        n = ei[b, NPLAY]
        alive_count = 0
        for q in range(n):
            if pi[b, q, ALIVE] == 1:
                alive_count += 1
        for p in range(n):
            if pi[b, p, ALIVE] == 0:
                continue
            hx = pf[b, p, FX]; hy = pf[b, p, FY]
            h = pf[b, p, FHEAD]
            self = p + 1
            fx = cos(h); fy = sin(h)       # forward unit
            rx = fy; ry = -fx              # right unit (forward rotated clockwise, y-up)
            # local crop, rotated so forward = up
            for iy in range(H):
                k = R - iy                 # forward distance (image top = ahead)
                for ix in range(W):
                    r = ix - R             # right distance
                    wx = hx + k * fx + r * rx
                    wy = hy + k * fy + r * ry
                    ddx = wx - ARENA_CX; ddy = wy - ARENA_CY
                    if (ddx * ddx + ddy * ddy) >= wall_lim2:
                        out_grid[b, p, 0, iy, ix] = 1
                        continue
                    cx = int(floor(wx)); cy = int(floor(wy))
                    if cx < 0 or cx >= GW or cy < 0 or cy >= GH:
                        out_grid[b, p, 0, iy, ix] = 1
                        continue
                    cid = cy * GW + cx
                    o = owner[b, cid]; t = trail[b, cid]
                    if o == self:
                        out_grid[b, p, 1, iy, ix] = 1
                    elif o != 0:
                        out_grid[b, p, 3, iy, ix] = 1
                    if t == self:
                        out_grid[b, p, 2, iy, ix] = 1
                    elif t != 0:
                        out_grid[b, p, 4, iy, ix] = 1
            # enemy heads inside the crop
            for q in range(n):
                if q == p or pi[b, q, ALIVE] == 0:
                    continue
                ex = pf[b, q, FX] - hx; ey = pf[b, q, FY] - hy
                fcomp = ex * fx + ey * fy
                rcomp = ex * rx + ey * ry
                iy = R - int(round(fcomp))
                ix = R + int(round(rcomp))
                if 0 <= iy < H and 0 <= ix < W:
                    out_grid[b, p, 5, iy, ix] = 1

            # scalars
            area = pi[b, p, AREA]
            out_scalar[b, p, 0] = float(pi[b, p, HAS])
            out_scalar[b, p, 1] = (area / N_CELLS) * 5.0
            tl = pi[b, p, TLEN] / 150.0
            out_scalar[b, p, 2] = tl if tl < 1.5 else 1.5
            out_scalar[b, p, 3] = alive_count / MAX_PLAYERS
            out_scalar[b, p, 4] = ei[b, ESTEP] / max_steps
            if pi[b, p, HAS] == 1:
                ex = pf[b, p, FEXITX] - hx; ey = pf[b, p, FEXITY] - hy
                fcomp = ex * fx + ey * fy
                rcomp = ex * rx + ey * ry
                out_scalar[b, p, 5] = (abs(ex) + abs(ey)) / (GW + GH)
                out_scalar[b, p, 6] = fcomp / GW
                out_scalar[b, p, 7] = rcomp / GW
            # nearest enemy head, egocentric
            best = 1e30; bf = 0.0; br = 0.0
            for q in range(n):
                if q == p or pi[b, q, ALIVE] == 0:
                    continue
                ex = pf[b, q, FX] - hx; ey = pf[b, q, FY] - hy
                md = abs(ex) + abs(ey)
                if md < best:
                    best = md
                    bf = ex * fx + ey * fy
                    br = ex * rx + ey * ry
            if best < 1e30:
                d = best / (GW + GH)
                out_scalar[b, p, 8] = d if d < 1.0 else 1.0
                out_scalar[b, p, 9] = bf / GW
                out_scalar[b, p, 10] = br / GW
            else:
                out_scalar[b, p, 8] = 1.0
            more = 0
            for q in range(n):
                if q == p or pi[b, q, ALIVE] == 0:
                    continue
                if pi[b, q, AREA] > area:
                    more += 1
            out_scalar[b, p, 11] = more / MAX_PLAYERS


# --------------------------------------------------------------------------- #
#  python env wrapper
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    B: int = 256
    n_min: int = 4
    n_max: int = MAX_PLAYERS
    max_steps: int = 2200          # ~147s of game time at DECISION_DT ~= 0.0667s
    obs_r: int = OBS_R_DEFAULT
    seed: int = 0
    # reward weights (hold_weight is the anti-turtle lever)
    death_penalty: float = -1.0
    kill_reward: float = 2.0
    area_scale: float = 100.0
    hold_weight: float = 0.10
    win_bonus: float = 10.0
    place_weight: float = 5.0
    step_cost: float = 0.0
    respawn: int = 1               # 1 = respawn + play-to-fill (like the game); 0 = permadeath/last-alive
    respawn_min: int = 33          # respawn delay range in env-steps (~2.2-4.4s at DECISION_DT)
    respawn_max: int = 66


class PaperLoop2Env:
    """Vectorized multi-agent free-movement env. Lanes are (env b, seat p); dead/
    inactive seats are masked via the returned `alive_start`. Auto-resets on episode end."""

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        B, P = cfg.B, MAX_PLAYERS
        self.B, self.P = B, P
        self.rng = np.random.default_rng(cfg.seed)
        self.max_steps = cfg.max_steps

        self.owner = np.zeros((B, N_CELLS), np.int8)
        self.trail = np.zeros((B, N_CELLS), np.int8)
        self.pi = np.zeros((B, P, NF), np.int32)
        self.pf = np.zeros((B, P, NFF), np.float32)
        self.polx = np.zeros((B, P, MAXPTS), np.float32)
        self.poly = np.zeros((B, P, MAXPTS), np.float32)
        self.ei = np.zeros((B, ME), np.int32)
        self.fillstamp = np.zeros((B, N_CELLS), np.int32)
        self.bfs = np.zeros((B, N_CELLS), np.int32)

        self.rewards = np.zeros((B, P), np.float32)
        self.dones = np.zeros((B, P), np.int8)
        self.alive_start = np.zeros((B, P), np.int8)
        self.area_before = np.zeros((B, P), np.int32)
        self.doomed = np.zeros((B, P), np.uint8)
        self.died = np.zeros((B, P), np.uint8)
        self.rew_kill = np.zeros((B, P), np.float32)
        self.encl = np.zeros((B, P), np.uint8)
        self.ep_ended = np.zeros(B, np.int8)
        seed_numba(cfg.seed)

        R = cfg.obs_r
        self.obs_grid = np.zeros((B, P, OBS_C, 2 * R + 1, 2 * R + 1), np.uint8)
        self.obs_scalar = np.zeros((B, P, OBS_S), np.float32)
        self.reset_all()

    def reset_env(self, b, n=None):
        _reset_inline(self.owner, self.trail, self.pi, self.pf, self.ei, b,
                      -1 if n is None else int(n), self.cfg.n_min, self.cfg.n_max)

    def reset_all(self, n=None):
        for b in range(self.B):
            self.reset_env(b, n)
        self._encode()
        return self.obs_grid, self.obs_scalar, self.alive_start

    def step(self, actions, auto_reset=1):
        c = self.cfg
        step_all(self.owner, self.trail, self.pi, self.pf, self.polx, self.poly, self.ei,
                 self.fillstamp, self.bfs,
                 actions, self.rewards, self.dones, self.alive_start, self.area_before,
                 self.doomed, self.died, self.rew_kill, self.encl, self.ep_ended,
                 c.death_penalty, c.kill_reward, c.area_scale, c.win_bonus,
                 c.place_weight, c.hold_weight, c.step_cost, self.max_steps,
                 c.n_min, c.n_max, int(auto_reset),
                 int(c.respawn), int(c.respawn_min), int(c.respawn_max))
        self._encode()
        return (self.obs_grid, self.obs_scalar, self.rewards, self.dones, self.alive_start)

    def _encode(self):
        self.obs_grid.fill(0)
        self.obs_scalar.fill(0)
        encode_all(self.owner, self.trail, self.pi, self.pf, self.ei,
                   self.obs_grid, self.obs_scalar, self.max_steps)
