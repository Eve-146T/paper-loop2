"""Match / evaluation harness for the free-movement game. Runs games to completion
(auto_reset OFF) with a mix of policy-controlled and scripted seats, reports
outcomes, and records replays (continuous float heads + trail polylines).
"""
import numpy as np
import torch

from .env import PaperLoop2Env, EnvConfig
from .scripted import ScriptedBots
from .rules import (NPLAY, ALIVE, PEAK, AREA, KILLS, POLN, N_CELLS,
                    FX, FY, FHEAD)


@torch.no_grad()
def policy_actions(model, env, device, greedy=False):
    BP = env.B * env.P
    g = torch.from_numpy(env.obs_grid).to(device).view(BP, *env.obs_grid.shape[2:]).float()
    s = torch.from_numpy(env.obs_scalar).to(device).view(BP, env.obs_scalar.shape[2])
    action, _, _ = model.act(g, s, greedy=greedy)
    return action.view(env.B, env.P).to(torch.int32).cpu().numpy()


def eval_vs_scripted(model, device, n_players=8, n_policy=4, B=64, max_steps=1200,
                     seed=12345, greedy=True):
    """Policy holds seats [0:n_policy], scripted holds [n_policy:n_players].
    Win = a policy seat ends with the most territory in its env; also mean peak %."""
    model.eval()
    env = PaperLoop2Env(EnvConfig(B=B, n_min=n_players, n_max=n_players,
                                  max_steps=max_steps, seed=seed))
    bots = ScriptedBots(env, seed=seed + 1)
    pol_mask = np.zeros((B, env.P), np.int32); pol_mask[:, :n_policy] = 1
    bot_mask = np.zeros((B, env.P), np.int32); bot_mask[:, n_policy:n_players] = 1
    done = np.zeros(B, bool)
    for _ in range(max_steps):
        pa = policy_actions(model, env, device, greedy)
        ba = bots.act(bot_mask)
        acts = np.where(pol_mask == 1, pa, ba).astype(np.int32)
        env.step(acts, auto_reset=0)
        done |= (env.ep_ended == 1)
        if done.all():
            break
    peak = env.pi[:, :, PEAK].astype(np.float64)
    alive = env.pi[:, :, ALIVE]
    pol_best = peak[:, :n_policy].max(1)
    bot_best = peak[:, n_policy:n_players].max(1)
    win = float((pol_best > bot_best).mean())
    surv = float(alive[:, :n_policy].sum() / (alive[:, :n_players].sum() + 1e-9))
    return dict(
        win=win,
        pol_terr=float(peak[:, :n_policy].mean() / N_CELLS * 100),
        bot_terr=float(peak[:, n_policy:n_players].mean() / N_CELLS * 100),
        pol_survival_share=surv,
    )


@torch.no_grad()
def _two_policy_actions(modelA, modelB, env, device, maskA_flat, greedy=True):
    BP = env.B * env.P
    g = torch.from_numpy(env.obs_grid).to(device).view(BP, *env.obs_grid.shape[2:]).float()
    s = torch.from_numpy(env.obs_scalar).to(device).view(BP, env.obs_scalar.shape[2])
    aA, _, _ = modelA.act(g, s, greedy=greedy)
    aB, _, _ = modelB.act(g, s, greedy=greedy)
    a = torch.where(maskA_flat.bool(), aA, aB)
    return a.view(env.B, env.P).to(torch.int32).cpu().numpy()


def eval_h2h(modelA, modelB, device, n_players=8, B=64, max_steps=1500, seed=4242, greedy=True):
    """Head-to-head: A on even seats, B on odd seats. A's win rate + each side's mean peak %."""
    modelA.eval(); modelB.eval()
    env = PaperLoop2Env(EnvConfig(B=B, n_min=n_players, n_max=n_players,
                                  max_steps=max_steps, seed=seed))
    maskA = np.zeros((B, env.P), np.int32)
    maskA[:, 0:n_players:2] = 1
    maskA_flat = torch.from_numpy(maskA.reshape(B * env.P)).to(device)
    done = np.zeros(B, bool)
    for _ in range(max_steps):
        acts = _two_policy_actions(modelA, modelB, env, device, maskA_flat, greedy)
        env.step(acts, auto_reset=0)
        done |= (env.ep_ended == 1)
        if done.all():
            break
    peak = env.pi[:, :, PEAK].astype(np.float64)
    a_seats = list(range(0, n_players, 2))
    b_seats = list(range(1, n_players, 2))
    a_best = peak[:, a_seats].max(1)
    b_best = peak[:, b_seats].max(1)
    return dict(a_win=float((a_best > b_best).mean()),
                a_terr=float(peak[:, a_seats].mean() / N_CELLS * 100),
                b_terr=float(peak[:, b_seats].mean() / N_CELLS * 100))


def snapshot(env, b):
    """Compact per-frame state for replay (one env b): owner + trail grids + continuous heads
    + trail polylines (so the libGDX-style renderer can draw smooth ribbons)."""
    n = int(env.ei[b, NPLAY])
    heads = np.array([[env.pf[b, p, FX], env.pf[b, p, FY], env.pf[b, p, FHEAD],
                       float(env.pi[b, p, ALIVE])] for p in range(n)], np.float32)
    polys = []
    for p in range(n):
        m = int(env.pi[b, p, POLN])
        polys.append(np.stack([env.polx[b, p, :m].copy(), env.poly[b, p, :m].copy()], axis=1)
                     if m > 0 else np.zeros((0, 2), np.float32))
    return env.owner[b].copy(), env.trail[b].copy(), heads, polys


def record_match(model, device, n_players=8, n_policy=4, max_steps=1500, seed=999,
                 greedy=False, all_policy=False):
    """Play ONE match (B=1); return (frames, meta). frames: list of (owner, trail, heads, polys)."""
    model.eval()
    env = PaperLoop2Env(EnvConfig(B=1, n_min=n_players, n_max=n_players,
                                  max_steps=max_steps, seed=seed))
    bots = ScriptedBots(env, seed=seed + 1)
    pol_mask = np.zeros((1, env.P), np.int32)
    bot_mask = np.zeros((1, env.P), np.int32)
    if all_policy:
        pol_mask[0, :n_players] = 1
    else:
        pol_mask[0, :n_policy] = 1
        bot_mask[0, n_policy:n_players] = 1
    frames = []
    for _ in range(max_steps):
        frames.append(snapshot(env, 0))
        pa = policy_actions(model, env, device, greedy)
        ba = bots.act(bot_mask)
        acts = np.where(pol_mask == 1, pa, ba).astype(np.int32)
        env.step(acts, auto_reset=0)
        if env.ep_ended[0] == 1:
            frames.append(snapshot(env, 0))
            break
    meta = dict(n_players=n_players, n_policy=(n_players if all_policy else n_policy),
                seed=seed, controllers=["policy" if pol_mask[0, p] else "scripted"
                                        for p in range(n_players)])
    return frames, meta
