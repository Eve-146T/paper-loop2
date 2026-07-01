"""Round-robin tournament among checkpoints to crown the strongest. Also reports
each one's territory vs the scripted baseline.

  python tournament.py checkpoints/base/best.pt checkpoints/aggro/best.pt ...
  python tournament.py 'checkpoints/*/best.pt'
"""
import argparse, glob, os, sys, itertools
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_loop.model import ActorCritic
from paper_loop.match import eval_h2h, eval_vs_scripted


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    a = ck.get("args", {})
    m = ActorCritic(width=a.get("width", 96), blocks=a.get("blocks", 4),
                    scalar_dim=a.get("scalar_dim", 128), trunk_dim=a.get("trunk_dim", 512)).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m


def label(path):
    parts = path.replace("\\", "/").split("/")
    run = parts[-2] if len(parts) > 1 else "?"
    return f"{run}/{os.path.basename(path).replace('.pt','')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--n_players", type=int, default=8)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    paths = []
    for c in args.ckpts:
        paths.extend(sorted(glob.glob(c)) if any(ch in c for ch in "*?[") else [c])
    paths = [p for p in paths if os.path.exists(p)]
    labels = [label(p) for p in paths]
    models = [load(p, device) for p in paths]
    K = len(models)
    print(f"loaded {K}: {labels}\n")

    print("vs scripted (territory %, win%):")
    scripted_terr = []
    for lab, m in zip(labels, models):
        ev = eval_vs_scripted(m, device, n_players=args.n_players, n_policy=args.n_players // 2,
                              B=args.B, max_steps=args.max_steps)
        scripted_terr.append(ev["pol_terr"])
        print(f"  {lab:24s} terr {ev['pol_terr']:5.2f}%  (bots {ev['bot_terr']:4.2f}%)  win {ev['win']*100:5.1f}%")

    win = np.full((K, K), np.nan)
    print("\nhead-to-head (row A win% vs col B):")
    for i, j in itertools.combinations(range(K), 2):
        r = eval_h2h(models[i], models[j], device, n_players=args.n_players,
                     B=args.B, max_steps=args.max_steps)
        win[i, j] = r["a_win"] * 100
        win[j, i] = (1 - r["a_win"]) * 100

    hdr = "  " + " ".join(f"{l.split('/')[0][:8]:>8s}" for l in labels)
    print(hdr)
    for i, lab in enumerate(labels):
        row = " ".join((f"{win[i,j]:8.1f}" if not np.isnan(win[i, j]) else f"{'--':>8s}")
                       for j in range(K))
        print(f"{lab.split('/')[0][:8]:>8s} {row}")

    avg = np.nanmean(win, axis=1)
    order = np.argsort(-avg)
    print("\nranking (avg head-to-head win%):")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank}. {labels[idx]:24s}  avg_win {avg[idx]:5.1f}%  vs_scripted {scripted_terr[idx]:5.2f}%")
    print(f"\nCHAMPION: {labels[order[0]]}")


if __name__ == "__main__":
    main()
