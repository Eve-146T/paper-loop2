"""Faithful Numba port of Bot.kt — the free-movement rectangle-venture planner with
a reactive safety filter (wall / own-trail / pocket avoidance). Used as the fixed
benchmark opponent and progress yardstick (win-rate vs scripted).

Bot.kt only ever sets an absolute target heading; here we evaluate the same safety
score over the 5 achievable relative-turn buckets and emit the best one directly as
the env action. Stateful per (env, seat): venture phase / timer / curl sign / goal
heading in `bs`. Run with auto_reset OFF (fixed rosters) so plan state stays coherent.
"""
from math import cos, sin, atan2, hypot, sqrt, floor

import numpy as np
from numba import njit, prange

from .rules import (
    GW, GH, N_CELLS, ARENA_CX, ARENA_CY, ARENA_R, WALL_MARGIN,
    HALF_PI, TWO_PI, PI_F, DECISION_DT, MAX_TURN_PER_DECISION, N_ACTIONS,
    ALIVE, HAS, AREA, TLEN, NPLAY,
    FX, FY, FHEAD, FEXITX, FEXITY,
)

# Bot.kt tunables
OUT_MIN = 0.7; OUT_RND = 1.0
SIDE_MIN = 0.6; SIDE_RND = 0.8
MAX_TRAIL = 78
DANGER_DIST = 6.0
LOOK = 3.4
WALL_LOOK = 5.5
NEEDFREE = 16

# bot-state fields in bs[B, P, 4]
BPHASE, BREM, BSIGN, BGOAL = 0, 1, 2, 3


@njit(cache=True, inline="always")
def _wrap(a):
    x = a % TWO_PI
    if x <= -PI_F:
        x += TWO_PI
    elif x > PI_F:
        x -= TWO_PI
    return x


@njit(cache=True, inline="always")
def _in_arena(x, y):
    dx = x - ARENA_CX; dy = y - ARENA_CY
    lim = ARENA_R - WALL_MARGIN
    return dx * dx + dy * dy < lim * lim


@njit(cache=True, inline="always")
def _owner_at(owner, b, fx, fy):
    cx = int(floor(fx)); cy = int(floor(fy))
    if cx < 0:
        cx = 0
    elif cx > GW - 1:
        cx = GW - 1
    if cy < 0:
        cy = 0
    elif cy > GH - 1:
        cy = GH - 1
    return owner[b, cy * GW + cx]


@njit(cache=True)
def _clear_wall(x, y, ang):
    dx = cos(ang); dy = sin(ang)
    d = 0.5
    while d <= WALL_LOOK:
        if not _in_arena(x + dx * d, y + dy * d):
            return d
        d += 0.5
    return WALL_LOOK


@njit(cache=True)
def _self_trail_ahead(trail, b, p, x, y, ang, has):
    if has == 0:
        return LOOK
    self = p + 1
    dx = cos(ang); dy = sin(ang)
    d = 1.1
    while d <= LOOK:
        wx = x + dx * d; wy = y + dy * d
        cx = int(floor(wx)); cy = int(floor(wy))
        if 0 <= cx < GW and 0 <= cy < GH and trail[b, cy * GW + cx] == self:
            return d
        d += 0.5
    return LOOK


@njit(cache=True)
def _reach_from(trail, fillstamp, bfs, stok, b, sx, sy, selfid, cap):
    """Flood from (sx,sy): count reachable cells (capped), own trail blocks. Mirror World.reachFrom."""
    if sx < 0 or sx >= GW or sy < 0 or sy >= GH:
        return 0
    stok[b] += 1
    tok = stok[b]
    head = 0; tail = 0; cnt = 0
    sid = sy * GW + sx
    fillstamp[b, sid] = tok; bfs[b, tail] = sid; tail += 1
    while head < tail and cnt < cap:
        c = bfs[b, head]; head += 1; cnt += 1
        cx = c % GW; cy = c // GW
        if cx > 0:
            cc = c - 1
            if fillstamp[b, cc] != tok and trail[b, cc] != selfid:
                fillstamp[b, cc] = tok
                if tail < N_CELLS:
                    bfs[b, tail] = cc; tail += 1
        if cx < GW - 1:
            cc = c + 1
            if fillstamp[b, cc] != tok and trail[b, cc] != selfid:
                fillstamp[b, cc] = tok
                if tail < N_CELLS:
                    bfs[b, tail] = cc; tail += 1
        if cy > 0:
            cc = c - GW
            if fillstamp[b, cc] != tok and trail[b, cc] != selfid:
                fillstamp[b, cc] = tok
                if tail < N_CELLS:
                    bfs[b, tail] = cc; tail += 1
        if cy < GH - 1:
            cc = c + GW
            if fillstamp[b, cc] != tok and trail[b, cc] != selfid:
                fillstamp[b, cc] = tok
                if tail < N_CELLS:
                    bfs[b, tail] = cc; tail += 1
    return cnt


@njit(cache=True)
def _outward_heading(owner, b, p, x, y):
    self = p + 1
    best = 0.0; bs = -1e30
    for k in range(12):
        ang = k * (TWO_PI / 12.0)
        wall = _clear_wall(x, y, ang)
        if wall < 3.0:
            continue
        empty = 0
        for s in range(1, 6):
            px = x + cos(ang) * s; py = y + sin(ang) * s
            if _in_arena(px, py) and _owner_at(owner, b, px, py) != self:
                empty += 1
        sc = empty + wall * 0.1 + np.random.random() * 0.6
        if sc > bs:
            bs = sc; best = ang
    return best


@njit(cache=True)
def _best_action(owner, trail, fillstamp, bfs, stok, b, p, x, y, h, has, desired):
    """Score each of the 5 relative-turn buckets (Bot.kt bestHeading over achievable headings)."""
    self = p + 1
    half_span = (N_ACTIONS - 1) * 0.5
    turn_step = MAX_TURN_PER_DECISION / half_span
    best_a = int(half_span); best_sc = -1e30
    for a in range(N_ACTIONS):
        off = (a - half_span) * turn_step
        ang = _wrap(h + off)
        sc = -abs(_wrap(ang - desired)) * 1.3
        wall = _clear_wall(x, y, ang)
        if wall < 1.3:
            sc -= 200.0
        else:
            sc += (wall if wall < WALL_LOOK else WALL_LOOK) * 0.22
        st = _self_trail_ahead(trail, b, p, x, y, ang, has)
        if st < LOOK:
            sc -= 80.0 * (1.0 - st / LOOK)
        lx = x + cos(ang) * LOOK; ly = y + sin(ang) * LOOK
        if _in_arena(lx, ly):
            cx = int(floor(lx)); cy = int(floor(ly))
            if _reach_from(trail, fillstamp, bfs, stok, b, cx, cy, self, NEEDFREE + 6) < NEEDFREE:
                sc -= 60.0
        else:
            sc -= 80.0
        if sc > best_sc:
            best_sc = sc; best_a = a
    return best_a


@njit(parallel=True, cache=True)
def scripted_decide_all(owner, trail, pi, pf, ei, bs, mask,
                        fillstamp, bfs, stok, out_actions):
    B = owner.shape[0]
    half = int((N_ACTIONS - 1) // 2)
    for b in prange(B):
        n = ei[b, NPLAY]
        for p in range(n):
            out_actions[b, p] = half          # default: straight
            if mask[b, p] == 0 or pi[b, p, ALIVE] == 0:
                continue
            x = pf[b, p, FX]; y = pf[b, p, FY]; h = pf[b, p, FHEAD]
            self = p + 1
            on_land = _owner_at(owner, b, x, y) == self
            has = pi[b, p, HAS]
            # start a fresh venture when safely home with no trail
            if on_land and has == 0:
                bs[b, p, BPHASE] = 0.0
                bs[b, p, BREM] = OUT_MIN + np.random.random() * OUT_RND
                bs[b, p, BSIGN] = 1.0 if np.random.random() < 0.5 else -1.0
                bs[b, p, BGOAL] = _outward_heading(owner, b, p, x, y)
            bs[b, p, BREM] -= DECISION_DT
            # nearest enemy head
            nearest = 1e30
            for q in range(n):
                if q != p and pi[b, q, ALIVE] == 1:
                    dx = x - pf[b, q, FX]; dy = y - pf[b, q, FY]
                    dd = sqrt(dx * dx + dy * dy)
                    if dd < nearest:
                        nearest = dd
            if has == 1 and (pi[b, p, TLEN] >= MAX_TRAIL or nearest < DANGER_DIST):
                bs[b, p, BPHASE] = 2.0
            if bs[b, p, BREM] <= 0.0:
                ph = bs[b, p, BPHASE]
                if ph == 0.0:
                    bs[b, p, BPHASE] = 1.0
                    bs[b, p, BREM] = SIDE_MIN + np.random.random() * SIDE_RND
                elif ph == 1.0:
                    bs[b, p, BPHASE] = 2.0
                    bs[b, p, BREM] = 999.0
            ph = bs[b, p, BPHASE]
            if ph == 0.0:
                desired = bs[b, p, BGOAL]
            elif ph == 1.0:
                desired = _wrap(bs[b, p, BGOAL] + bs[b, p, BSIGN] * HALF_PI)
            else:
                desired = atan2(pf[b, p, FEXITY] - y, pf[b, p, FEXITX] - x)
            out_actions[b, p] = _best_action(owner, trail, fillstamp, bfs, stok,
                                             b, p, x, y, h, has, desired)


class ScriptedBots:
    """Drives masked seats with the scripted policy; emits relative actions for the env."""

    def __init__(self, env, seed=0):
        self.env = env
        self.B, self.P = env.B, env.P
        self.bs = np.zeros((self.B, self.P, 4), np.float32)
        self.actions = np.zeros((self.B, self.P), np.int32)
        self.fillstamp = np.zeros((self.B, N_CELLS), np.int32)
        self.bfs = np.zeros((self.B, N_CELLS), np.int32)
        self.stok = np.zeros(self.B, np.int32)
        np.random.seed(seed)

    def reset(self):
        self.bs.fill(0.0)
        self.fillstamp.fill(0)
        self.stok.fill(0)

    def act(self, mask):
        e = self.env
        scripted_decide_all(e.owner, e.trail, e.pi, e.pf, e.ei, self.bs,
                            mask.astype(np.uint8), self.fillstamp, self.bfs, self.stok,
                            self.actions)
        return self.actions
