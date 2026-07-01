"""Self-play PPO trainer for Paper Loop 2 (free movement).

Shared policy controls all seats (parameter sharing). Rollout over B parallel
envs x P seats; per-agent GAE handles permadeath (a lane terminates at its
agent's death; dead-waiting steps are masked out). Checkpoints are pruned to stay
well under the disk budget. Run:

    python train.py --run base --updates 6000

Resumes automatically from checkpoints/<run>/latest.pt if present.
"""
import argparse, os, sys, time, json, glob, math
from contextlib import nullcontext
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_loop.env import PaperLoop2Env, EnvConfig
from paper_loop.model import ActorCritic, count_params
from paper_loop.match import eval_vs_scripted
from paper_loop.rules import OBS_C, OBS_S, OBS_R_DEFAULT, MAX_PLAYERS


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="base")
    p.add_argument("--updates", type=int, default=6000)
    p.add_argument("--B", type=int, default=256)         # parallel envs
    p.add_argument("--T", type=int, default=64)          # rollout length
    p.add_argument("--n_min", type=int, default=4)
    p.add_argument("--n_max", type=int, default=12)
    p.add_argument("--max_steps", type=int, default=2200)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--gradnorm", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--width", type=int, default=96)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--scalar_dim", type=int, default=128)
    p.add_argument("--trunk_dim", type=int, default=512)
    # reward weights
    p.add_argument("--death_penalty", type=float, default=-1.0)
    p.add_argument("--kill_reward", type=float, default=2.0)
    p.add_argument("--area_scale", type=float, default=100.0)
    p.add_argument("--win_bonus", type=float, default=10.0)
    p.add_argument("--place_weight", type=float, default=5.0)
    p.add_argument("--hold_weight", type=float, default=0.10)
    p.add_argument("--step_cost", type=float, default=0.0)
    p.add_argument("--respawn", type=int, default=1)
    # bookkeeping
    p.add_argument("--ckpt_every", type=int, default=100)
    p.add_argument("--milestone_every", type=int, default=500)
    p.add_argument("--keep_last", type=int, default=5)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--max_ckpt_gb", type=float, default=35.0)
    # league (self-play vs past checkpoints); 0 = pure self-play
    p.add_argument("--league_frac", type=float, default=0.0)
    p.add_argument("--snapshot_every", type=int, default=150)
    p.add_argument("--league_size", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_from", default="")   # fine-tune: load weights only, fresh opt/steps
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def dir_size_gb(path):
    tot = 0
    for f in glob.glob(os.path.join(path, "**", "*"), recursive=True):
        if os.path.isfile(f):
            tot += os.path.getsize(f)
    return tot / 1e9


def prune_checkpoints(ckpt_dir, keep_last, max_gb):
    # milestones (ms_*) are always kept; step_*.pt pruned to keep_last newest
    steps = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")), key=os.path.getmtime)
    for f in steps[:-keep_last] if keep_last > 0 else steps:
        try:
            os.remove(f)
        except OSError:
            pass
    # hard size guard: drop oldest non-milestone, non-latest until under budget
    while dir_size_gb(ckpt_dir) > max_gb:
        steps = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")), key=os.path.getmtime)
        if not steps:
            break
        try:
            os.remove(steps[0])
        except OSError:
            break


def main():
    args = parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    amp = (device == "cuda")
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if amp else nullcontext()
    B, T, P = args.B, args.T, MAX_PLAYERS   # env always allocates MAX_PLAYERS seats
    BP = B * P

    ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", args.run)
    run_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", args.run)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(run_dir)
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    cfg = EnvConfig(B=B, n_min=args.n_min, n_max=args.n_max, max_steps=args.max_steps,
                    obs_r=OBS_R_DEFAULT, seed=args.seed,
                    death_penalty=args.death_penalty, kill_reward=args.kill_reward,
                    area_scale=args.area_scale, win_bonus=args.win_bonus,
                    place_weight=args.place_weight, hold_weight=args.hold_weight,
                    step_cost=args.step_cost, respawn=args.respawn)
    env = PaperLoop2Env(cfg)
    C, H, W = OBS_C, 2 * OBS_R_DEFAULT + 1, 2 * OBS_R_DEFAULT + 1

    model = ActorCritic(width=args.width, blocks=args.blocks,
                        scalar_dim=args.scalar_dim, trunk_dim=args.trunk_dim).to(device)
    print(f"model params: {count_params(model)/1e6:.2f}M  device={device}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    if args.compile:
        model = torch.compile(model)
    raw = model._orig_mod if hasattr(model, "_orig_mod") else model

    # league: a fraction of envs put past-checkpoint opponents on odd seats; we train
    # only on learner (even) seats. league_frac=0 -> pure self-play (identical path).
    league = []
    opp_model = None
    learner_mask = torch.ones(B, P, device=device)
    if args.league_frac > 0:
        opp_model = ActorCritic(width=args.width, blocks=args.blocks,
                                scalar_dim=args.scalar_dim, trunk_dim=args.trunk_dim).to(device).eval()
        is_league = torch.rand(B, device=device) < args.league_frac
        odd = (torch.arange(P, device=device) % 2 == 1)
        learner_mask[is_league] = (~odd).float()
    learner_flat = learner_mask.reshape(BP)

    # rollout buffers (GPU)
    obs_g = torch.zeros((T, BP, C, H, W), dtype=torch.uint8, device=device)
    obs_s = torch.zeros((T, BP, OBS_S), dtype=torch.float32, device=device)
    acts = torch.zeros((T, BP), dtype=torch.long, device=device)
    logps = torch.zeros((T, BP), device=device)
    vals = torch.zeros((T, BP), device=device)
    rews = torch.zeros((T, BP), device=device)
    dones = torch.zeros((T, BP), device=device)
    alive = torch.zeros((T, BP), device=device)

    start_update = 0
    global_steps = 0
    best_win = -1.0
    latest = os.path.join(ckpt_dir, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device)
        (model._orig_mod if hasattr(model, "_orig_mod") else model).load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_update = ck["update"]
        global_steps = ck.get("global_steps", 0)
        best_win = ck.get("best_win", -1.0)
        print(f"resumed from update {start_update} ({global_steps} steps)", flush=True)
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device)
        raw.load_state_dict(ck["model"])
        print(f"init weights from {args.init_from} (fresh opt/steps)", flush=True)
        if opp_model is not None:   # seed the league pool with the init weights
            league.append({k: v.detach().cpu().clone() for k, v in raw.state_dict().items()})

    def save(path, update):
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save(dict(model=raw.state_dict(), opt=opt.state_dict(), update=update,
                        global_steps=global_steps, best_win=best_win,
                        args=vars(args)), path)

    def to_dev(np_arr):
        return torch.from_numpy(np_arr).to(device)

    gamma, lam = args.gamma, args.lam
    t_start = time.time()
    for update in range(start_update, args.updates):
        model.eval()
        roll_reward = 0.0
        ep_count = 0
        t0 = time.time()
        # ---- rollout ----
        if opp_model is not None:
            opp_model.load_state_dict(
                league[np.random.randint(len(league))] if league else raw.state_dict())
        for t in range(T):
            g_u8 = to_dev(env.obs_grid).view(BP, C, H, W)
            s = to_dev(env.obs_scalar).view(BP, OBS_S)
            with torch.no_grad(), autocast():
                a, lp, v = model.act(g_u8.float(), s, greedy=False)
            v = v.float()
            obs_g[t] = g_u8
            obs_s[t] = s
            acts[t] = a
            logps[t] = lp
            vals[t] = v
            if opp_model is not None:
                with torch.no_grad(), autocast():
                    oa, _, _ = opp_model.act(g_u8.float(), s, greedy=False)
                a_use = torch.where(learner_flat.bool(), a, oa)
            else:
                a_use = a
            a_np = a_use.view(B, P).to(torch.int32).cpu().numpy()
            env.step(a_np)
            rews[t] = to_dev(env.rewards.reshape(BP))
            dones[t] = to_dev(env.dones.reshape(BP).astype(np.float32))
            alive[t] = to_dev(env.alive_start.reshape(BP).astype(np.float32)) * learner_flat
            roll_reward += float(env.rewards[env.alive_start == 1].sum())
            ep_count += int(env.ep_ended.sum())
        global_steps += T * B

        # ---- bootstrap + per-agent GAE ----
        with torch.no_grad(), autocast():
            g_u8 = to_dev(env.obs_grid).view(BP, C, H, W)
            s = to_dev(env.obs_scalar).view(BP, OBS_S)
            _, last_v = model(g_u8.float(), s)
        last_v = last_v.float()
        adv = torch.zeros_like(rews)
        lastgae = torch.zeros(BP, device=device)
        for t in reversed(range(T)):
            nonterm = 1.0 - dones[t]
            nextv = last_v if t == T - 1 else vals[t + 1]
            delta = rews[t] + gamma * nextv * nonterm - vals[t]
            lastgae = delta + gamma * lam * nonterm * lastgae
            adv[t] = lastgae
        returns = adv + vals

        # ---- flatten alive transitions ----
        flat_alive = alive.reshape(-1).bool()
        idx = flat_alive.nonzero(as_tuple=False).squeeze(1)
        N = idx.numel()
        if N == 0:
            continue
        b_g = obs_g.reshape(-1, C, H, W)
        b_s = obs_s.reshape(-1, OBS_S)
        b_a = acts.reshape(-1)
        b_lp = logps.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = returns.reshape(-1)
        b_val = vals.reshape(-1)
        adv_sel = b_adv[idx]
        adv_norm = (adv_sel - adv_sel.mean()) / (adv_sel.std() + 1e-8)

        # ---- PPO update ----
        model.train()
        last_kl = last_pg = last_vl = last_ent = 0.0
        clipfrac = 0.0
        for epoch in range(args.epochs):
            perm = idx[torch.randperm(N, device=device)]
            adv_perm = None
            for start in range(0, N, args.minibatch):
                mb = perm[start:start + args.minibatch]
                # advantage for this mb (recompute normalized via gather on full)
                a_mb = b_adv[mb]
                a_mb = (a_mb - adv_sel.mean()) / (adv_sel.std() + 1e-8)
                g = b_g[mb].float()
                s = b_s[mb]
                with autocast():
                    newlp, ent, value = model.evaluate(g, s, b_a[mb])
                    ratio = (newlp - b_lp[mb]).exp()
                    pg1 = -a_mb * ratio
                    pg2 = -a_mb * torch.clamp(ratio, 1 - args.clip, 1 + args.clip)
                    pg = torch.max(pg1, pg2).mean()
                    v_clip = b_val[mb] + (value - b_val[mb]).clamp(-args.clip, args.clip)
                    vl = 0.5 * torch.max((value - b_ret[mb]) ** 2,
                                         (v_clip - b_ret[mb]) ** 2).mean()
                    ent_m = ent.mean()
                    loss = pg + args.vf * vl - args.ent * ent_m
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.gradnorm)
                opt.step()
                with torch.no_grad():
                    last_kl = float((b_lp[mb] - newlp).mean())
                    clipfrac = float(((ratio - 1).abs() > args.clip).float().mean())
                last_pg, last_vl, last_ent = float(pg.detach()), float(vl.detach()), float(ent_m.detach())

        dt = time.time() - t0
        sps = T * B / dt
        mean_r = roll_reward / max(1, int(alive.sum().item()))
        if update % 5 == 0:
            print(f"upd {update:5d} | steps {global_steps/1e6:6.2f}M | {sps:6.0f} sps | "
                  f"R/agent {mean_r:+.4f} | ent {last_ent:.3f} | kl {last_kl:+.4f} | "
                  f"pg {last_pg:+.4f} vl {last_vl:.4f} | eps {ep_count} | N {N}", flush=True)
        writer.add_scalar("train/reward_per_agent", mean_r, global_steps)
        writer.add_scalar("train/entropy", last_ent, global_steps)
        writer.add_scalar("train/approx_kl", last_kl, global_steps)
        writer.add_scalar("train/pg_loss", last_pg, global_steps)
        writer.add_scalar("train/value_loss", last_vl, global_steps)
        writer.add_scalar("train/clipfrac", clipfrac, global_steps)
        writer.add_scalar("train/sps", sps, global_steps)
        writer.add_scalar("train/episodes", ep_count, global_steps)

        # ---- eval ----
        if update % args.eval_every == 0 and update > start_update:
            ev = eval_vs_scripted(model._orig_mod if hasattr(model, "_orig_mod") else model,
                                  device, n_players=8, n_policy=4, B=64)
            writer.add_scalar("eval/win_vs_scripted", ev["win"], global_steps)
            writer.add_scalar("eval/policy_territory", ev["pol_terr"], global_steps)
            writer.add_scalar("eval/scripted_territory", ev["bot_terr"], global_steps)
            writer.add_scalar("eval/policy_survival_share", ev["pol_survival_share"], global_steps)
            print(f"  [eval] win {ev['win']*100:5.1f}%  pol_terr {ev['pol_terr']:.2f}%  "
                  f"bot_terr {ev['bot_terr']:.2f}%  survshare {ev['pol_survival_share']:.2f}", flush=True)
            # win-dominant with territory as tie-breaker (win saturates at 100%)
            score = ev["win"] * 100 + ev["pol_terr"]
            if score > best_win:
                best_win = score
                save(os.path.join(ckpt_dir, "best.pt"), update)

        # ---- league snapshot ----
        if opp_model is not None and update % args.snapshot_every == 0 and update > start_update:
            league.append({k: v.detach().cpu().clone() for k, v in raw.state_dict().items()})
            if len(league) > args.league_size:
                league.pop(0)

        # ---- checkpoint ----
        if update % args.ckpt_every == 0 and update > start_update:
            save(latest, update)
            save(os.path.join(ckpt_dir, f"step_{update:06d}.pt"), update)
            prune_checkpoints(ckpt_dir, args.keep_last, args.max_ckpt_gb)
        if update % args.milestone_every == 0 and update > start_update:
            save(os.path.join(ckpt_dir, f"ms_{update:06d}.pt"), update)

    save(latest, args.updates)
    writer.close()
    print(f"done. total {global_steps/1e6:.1f}M steps in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
