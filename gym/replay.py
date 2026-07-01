"""Render a champion match to mp4/gif (eye-candy + behaviour sanity check).

    uv run python replay.py --ckpt checkpoints/league/best.pt --all_policy --out replays/champ.mp4
    uv run python replay.py --ckpt checkpoints/base/best.pt   --out replays/base.gif

Draws the circular arena, per-player territory + trails (lighter), and heads, from
match.record_match snapshots (owner/trail grids + continuous heads).
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_loop.model import ActorCritic
from paper_loop.match import record_match
from paper_loop.rules import GW, GH, ARENA_CX, ARENA_CY, ARENA_R, MAX_PLAYERS

# 12 vivid team colours (index 0 = human-ish), RGB
TEAM = np.array([
    [ 90, 200, 245], [255, 159,  64], [ 80, 200, 120], [201,  90, 230],
    [255,  99, 132], [ 75, 192, 192], [255, 205,  86], [120, 130, 245],
    [ 60, 200, 180], [240, 110,  90], [160, 210,  70], [230, 120, 200],
], np.uint8)
BG = np.array([238, 235, 220], np.uint8)
OUT = np.array([250, 248, 240], np.uint8)


def load(ckpt, device):
    ck = torch.load(ckpt, map_location=device)
    a = ck.get("args", {})
    m = ActorCritic(width=a.get("width", 32), blocks=a.get("blocks", 2),
                    scalar_dim=a.get("scalar_dim", 64), trunk_dim=a.get("trunk_dim", 128)).to(device)
    m.load_state_dict(ck["model"]); m.eval()
    return m


def _light(col):
    return tuple(int(c * 0.45 + 255 * 0.55) for c in col)


def render(frame, scale):
    """Anti-aliased: territory upscaled with LANCZOS (smooth blob edges), then ribbon trails
    (polylines) + oriented heads drawn on top with PIL, supersampled 2x and downsampled."""
    from PIL import Image, ImageDraw
    import math
    owner, trail, heads, polys = frame
    # base territory image at grid resolution
    img = np.broadcast_to(BG, (GH, GW, 3)).copy()
    yy, xx = np.mgrid[0:GH, 0:GW]
    outside = (xx + 0.5 - ARENA_CX) ** 2 + (yy + 0.5 - ARENA_CY) ** 2 > ARENA_R ** 2
    img[outside] = OUT
    o = owner.reshape(GH, GW)
    for pid in range(1, MAX_PLAYERS + 1):
        img[o == pid] = TEAM[(pid - 1) % len(TEAM)]
    img = np.flipud(img)                                  # world y-up -> image y-down
    ss = 2                                                # supersample factor
    W = GW * scale * ss; H = GH * scale * ss
    im = Image.fromarray(img, "RGB").resize((W, H), Image.LANCZOS)
    d = ImageDraw.Draw(im, "RGBA")

    def pt(x, y):                                         # world -> supersampled image px
        return (x * scale * ss, (GH - y) * scale * ss)

    # ribbon trails (lighter team colour) from the polylines
    for pid in range(1, MAX_PLAYERS + 1):
        poly = polys[pid - 1] if pid - 1 < len(polys) else None
        if poly is None or len(poly) < 2:
            continue
        col = _light(TEAM[(pid - 1) % len(TEAM)])
        wpx = max(2, int(1.7 * scale * ss))
        d.line([pt(px, py) for px, py in poly], fill=col + (235,), width=wpx, joint="curve")
    # heads: oriented square + dark outline
    for hd in heads:
        hx, hy, hh, alive = hd
        if alive < 0.5:
            continue
        cx, cy = pt(hx, hy)
        r = 1.4 * scale * ss
        ca, sa = math.cos(hh), math.sin(hh)
        corners = []
        for dx, dy in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            wx = dx * r; wy = dy * r
            rx = wx * ca - wy * sa; ry = wx * sa + wy * ca
            corners.append((cx + rx, cy - ry))           # -ry: image y is down
        d.polygon(corners, fill=(30, 32, 40, 255))
    return np.asarray(im.resize((GW * scale, GH * scale), Image.LANCZOS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/league/best.pt")
    ap.add_argument("--out", default="replays/champ.mp4")
    ap.add_argument("--n_players", type=int, default=8)
    ap.add_argument("--n_policy", type=int, default=4)
    ap.add_argument("--all_policy", action="store_true")
    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    args = ap.parse_args()
    import imageio.v2 as imageio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = load(args.ckpt, device)
    frames, meta = record_match(m, device, n_players=args.n_players, n_policy=args.n_policy,
                                max_steps=args.max_steps, seed=args.seed, all_policy=args.all_policy)
    print(f"recorded {len(frames)} frames; controllers={meta['controllers']}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    imgs = [render(f, args.scale) for f in frames[::args.stride]]
    if args.out.endswith(".gif"):
        imageio.mimsave(args.out, imgs, fps=args.fps, loop=0)
    else:
        imageio.mimsave(args.out, imgs, fps=args.fps, quality=8, macro_block_size=None)
    print(f"wrote {args.out}  ({len(imgs)} frames @ {args.fps}fps)")


if __name__ == "__main__":
    main()
